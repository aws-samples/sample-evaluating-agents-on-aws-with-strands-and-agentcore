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
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from strands_evals import Case
from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.adapters._session import build_session
from agentic_evaluation.types import TaskFnResult

# Per-token USD pricing for the agent's model. Defaults match Anthropic Claude
# Sonnet 4.x list pricing ($3 / 1M input, $15 / 1M output).
_DEFAULT_INPUT_COST_PER_1K = 0.003
_DEFAULT_OUTPUT_COST_PER_1K = 0.015


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


def make_task_fn(
    agent: Any | None = None,
    *,
    agent_factory: Callable[[], Any] | None = None,
    input_cost_per_1k: float = _DEFAULT_INPUT_COST_PER_1K,
    output_cost_per_1k: float = _DEFAULT_OUTPUT_COST_PER_1K,
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
    """
    if (agent is None) == (agent_factory is None):
        raise ValueError("Pass exactly one of agent or agent_factory")
    shared_agent_lock = threading.Lock()
    conversation_agents: dict[str, Any] = {}

    def task_fn(case: Case) -> TaskFnResult:
        metadata = case.metadata or {}
        conversation_id = metadata.get("conversation_id")
        run_id = metadata.get("evaluation_run_id", "default")
        conversation_key = f"{run_id}:{conversation_id}" if conversation_id else None
        if agent_factory is not None and conversation_key:
            with shared_agent_lock:
                active_agent = conversation_agents.get(conversation_key)
                if active_agent is None:
                    active_agent = agent_factory()
                    conversation_agents[conversation_key] = active_agent
        else:
            active_agent = agent_factory() if agent_factory is not None else agent
        if active_agent is None:  # pragma: no cover - guarded above
            raise RuntimeError("Agent construction failed")

        # A shared Agent is not thread-safe while its history is temporarily
        # cleared. Serialize that compatibility path; factory-created agents do
        # not share state and need no lock.
        lock = shared_agent_lock if agent_factory is None else threading.Lock()
        with lock:
            history = getattr(active_agent, "messages", None)
            original_messages = list(history or [])
            preserve_conversation = conversation_key is not None and agent_factory is not None
            if isinstance(history, list) and not preserve_conversation:
                history.clear()
            message_start = len(history or [])
            try:
                start_dt = datetime.now(timezone.utc)
                start = time.perf_counter()
                result = active_agent(case.input)
                elapsed_ms = (time.perf_counter() - start) * 1000
                end_dt = datetime.now(timezone.utc)
                messages = list(getattr(active_agent, "messages", []) or [])[message_start:]
            finally:
                current_history = getattr(active_agent, "messages", None)
                if isinstance(current_history, list) and not preserve_conversation:
                    current_history[:] = original_messages
        if (
            conversation_key
            and metadata.get("turn_index") == metadata.get("turn_count")
            and agent_factory is not None
        ):
            with shared_agent_lock:
                conversation_agents.pop(conversation_key, None)

        output = ""
        if hasattr(result, "message"):
            msg = result.message
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and "text" in block:
                        output += block["text"]
            elif isinstance(msg, str):
                output = msg

        tool_calls = _extract_trajectory(messages)
        available_tools = [str(name) for name in getattr(active_agent, "tool_names", []) or []]

        usage = dict(getattr(getattr(result, "metrics", None), "accumulated_usage", {}) or {})
        input_tokens = int(usage.get("inputTokens", 0))
        output_tokens = int(usage.get("outputTokens", 0))
        total_tokens = int(usage.get("totalTokens", input_tokens + output_tokens))
        estimated_cost = (
            input_tokens / 1000 * input_cost_per_1k + output_tokens / 1000 * output_cost_per_1k
        )

        session = build_session(
            session_id=f"local-{uuid.uuid4().hex}",
            user_prompt=str(case.input),
            agent_response=output,
            tool_calls=tool_calls,
            available_tools=available_tools,
            start=start_dt,
            end=end_dt,
        )

        metrics_state = {
            "latency_ms": elapsed_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "stop_reason": getattr(result, "stop_reason", None),
        }

        return {
            "output": output,
            "trajectory": session,
            "environment_state": [EnvironmentState(name="metrics", state=metrics_state)],
            "metadata": metrics_state,
        }

    return task_fn
