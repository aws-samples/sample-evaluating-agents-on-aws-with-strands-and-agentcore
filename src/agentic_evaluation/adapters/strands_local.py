# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Adapter for a local Strands ``Agent`` instance.

Usage::

    from strands import Agent
    from agentic_evaluation.adapters.strands_local import make_task_fn
    from agentic_evaluation import run_all_layers

    agent = Agent(model="us.anthropic.claude-sonnet-4-6", tools=[...])
    results = run_all_layers(task_fn=make_task_fn(agent))
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import UTC, datetime
from typing import Any

from strands_evals import Case
from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.adapters._conversation import ConversationScope, conversation_key
from agentic_evaluation.adapters._session import build_session
from agentic_evaluation.adapters.metrics import (
    DEFAULT_PRICING,
    TokenPricing,
    TokenUsage,
    base_metrics,
)
from agentic_evaluation.types import TaskFnResult


def _extract_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each assistant ``toolUse`` with its ``toolResult`` from history."""
    results_by_id: dict[str, str] = {}
    for message in messages:
        for block in message.get("content", []):
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            tool_result = block["toolResult"]
            use_id = tool_result.get("toolUseId", "")
            content = tool_result.get("content", [])
            text = ""
            if isinstance(content, list):
                text = "".join(
                    c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
                )
            results_by_id[use_id] = text or json.dumps(content, default=str)

    trajectory: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict) or "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            use_id = tool_use.get("toolUseId", "")
            trajectory.append(
                {
                    "name": tool_use.get("name", ""),
                    "arguments": tool_use.get("input", {}),
                    "result": results_by_id.get(use_id, ""),
                    "tool_use_id": use_id,
                }
            )
    return trajectory


def _extract_output(result: Any) -> str:
    """Read the assistant's text out of a Strands ``AgentResult``.

    Args:
        result: Whatever the agent returned.

    Returns:
        The concatenated text blocks, or an empty string when the result carries
        no textual reply.
    """
    message = getattr(result, "message", None)
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return "".join(
            block["text"]
            for block in message.get("content", [])
            if isinstance(block, dict) and "text" in block
        )
    return ""


def _extract_usage(result: Any) -> TokenUsage:
    """Read token counts off a Strands ``AgentResult``.

    Args:
        result: Whatever the agent returned.

    Returns:
        The turn's usage; all zeros when the agent reported no metrics.
    """
    usage = getattr(getattr(result, "metrics", None), "accumulated_usage", None) or {}
    return TokenUsage.from_counts(
        usage.get("inputTokens"),
        usage.get("outputTokens"),
        usage.get("totalTokens"),
    )


@contextmanager
def _isolated_history(
    agent: Any,
    *,
    preserve: bool,
    lock: AbstractContextManager[Any],
) -> Iterator[int]:
    """Run one invocation against a clean message history, then restore it.

    A shared ``Agent`` is not thread-safe while its history is temporarily
    cleared, so the whole invocation is serialised on ``lock``.

    Args:
        agent: The agent about to be invoked.
        preserve: Leave the history alone. Set for a multi-turn conversation,
            which needs the earlier turns to still be there.
        lock: Held for the duration of the invocation. Factory-built agents
            share no state, so they pass a :func:`~contextlib.nullcontext`.

    Yields:
        The index the turn's new messages start at, for slicing them out
        afterwards.
    """
    with lock:
        history = getattr(agent, "messages", None)
        original_messages = list(history or [])
        if isinstance(history, list) and not preserve:
            history.clear()
        try:
            yield len(history or [])
        finally:
            current_history = getattr(agent, "messages", None)
            if isinstance(current_history, list) and not preserve:
                current_history[:] = original_messages


def make_task_fn(
    agent: Any | None = None,
    *,
    agent_factory: Callable[[], Any] | None = None,
    pricing: TokenPricing = DEFAULT_PRICING,
) -> Callable[[Case], TaskFnResult]:
    """Wrap a Strands Agent so it satisfies the ``task_fn`` contract.

    The returned callable:
        - Sends ``case.input`` as the prompt.
        - Extracts the assistant's text response.
        - Rebuilds a Strands ``Session`` from the conversation history so the
          Session-level judges (Helpfulness, GoalSuccessRate) run for real.
        - Surfaces latency/token/cost via ``environment_state`` for the domain
          evaluators (the channel ``strands_evals`` propagates).

    Pass ``agent_factory`` for the strongest isolation, especially when an
    agent has a session manager or other external state. For backward
    compatibility, a supplied ``agent`` has its in-memory messages cleared for
    each case and restored after the invocation.

    Args:
        agent: An existing agent, reused for every case.
        agent_factory: Builds a fresh agent per conversation. Mutually exclusive
            with ``agent``.
        pricing: Per-1K-token rates used to estimate cost for
            :class:`~agentic_evaluation.evaluators.CostEvaluator`.

    Returns:
        A ``task_fn`` that invokes the agent once per case.

    Raises:
        ValueError: Neither or both of ``agent`` and ``agent_factory`` given.

    .. versionchanged:: 0.4.0
        ``input_cost_per_1k`` / ``output_cost_per_1k`` are replaced by the
        ``pricing`` value object.
    """
    if (agent is None) == (agent_factory is None):
        raise ValueError("Pass exactly one of agent or agent_factory")
    # Normalising both forms to one factory collapses the shared-agent and
    # per-conversation paths below into a single lookup. With no factory the
    # "factory" hands back the one shared agent every time.
    reuses_history = agent_factory is not None
    factory: Callable[[], Any] = agent_factory if agent_factory is not None else lambda: agent
    agents: ConversationScope[Any] = ConversationScope(factory)

    def task_fn(case: Case) -> TaskFnResult:
        metadata = case.metadata or {}
        key = conversation_key(metadata)
        active_agent = agents.acquire(key)

        # Factory-built agents own their history, so a multi-turn conversation
        # keeps it. A shared agent is isolated per case and needs the lock.
        with _isolated_history(
            active_agent,
            preserve=reuses_history and key is not None,
            lock=nullcontext() if reuses_history else agents.lock,
        ) as message_start:
            start_dt = datetime.now(UTC)
            start = time.perf_counter()
            result = active_agent(case.input)
            elapsed_ms = (time.perf_counter() - start) * 1000
            end_dt = datetime.now(UTC)
            messages = list(getattr(active_agent, "messages", []) or [])[message_start:]

        agents.release(key, metadata)

        output = _extract_output(result)
        session = build_session(
            session_id=f"local-{uuid.uuid4().hex}",
            user_prompt=str(case.input),
            agent_response=output,
            tool_calls=_extract_trajectory(messages),
            available_tools=[str(name) for name in getattr(active_agent, "tool_names", []) or []],
            start=start_dt,
            end=end_dt,
        )

        metrics_state = {
            **base_metrics(elapsed_ms, _extract_usage(result), pricing),
            "stop_reason": getattr(result, "stop_reason", None),
        }

        return {
            "output": output,
            "trajectory": session,
            "environment_state": [EnvironmentState(name="metrics", state=metrics_state)],
            "metadata": metrics_state,
        }

    return task_fn
