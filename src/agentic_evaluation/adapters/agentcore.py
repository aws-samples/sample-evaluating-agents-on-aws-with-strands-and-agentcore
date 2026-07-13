# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Adapter for a deployed Amazon Bedrock AgentCore runtime.

Usage::

    from agentic_evaluation.adapters.agentcore import make_task_fn
    from agentic_evaluation import run_all_layers

    results = run_all_layers(
        task_fn=make_task_fn(
            runtime_arn="arn:aws:bedrock-agentcore:eu-west-1:123:runtime/my-agent",
            region="eu-west-1",
        )
    )

The deployed agent returns ordered tool calls and token usage alongside its
text reply (see ``agent/app.py``). This adapter turns that into a real Strands
``Session`` so the Session-level judges (Helpfulness, GoalSuccessRate) score the
genuine trajectory, and surfaces live latency/token/cost figures through
``environment_state`` — the only task-result channel ``strands_evals`` actually
propagates onto ``EvaluationData`` (task-result ``metadata`` is dropped).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import boto3
from strands_evals import Case
from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.adapters._session import build_session
from agentic_evaluation.types import TaskFnResult

# Per-token USD pricing for the deployed agent's model. Defaults match Anthropic
# Claude Sonnet 4.x list pricing ($3 / 1M input, $15 / 1M output) — override via
# ``make_task_fn`` if you deploy a different model or have negotiated rates.
_DEFAULT_INPUT_COST_PER_1K = 0.003
_DEFAULT_OUTPUT_COST_PER_1K = 0.015


def _extract_text(result: Any) -> str:
    """Best-effort extraction of the agent's textual reply.

    BedrockAgentCoreApp default shape is::

        {"result": {"role": "assistant",
                    "content": [{"text": "..."}], ...}}
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            joined = "".join(parts)
            if joined:
                return joined
        if "text" in result:
            return str(result["text"])
    return json.dumps(result) if result is not None else ""


def _normalize_trajectory(raw: Any) -> list[dict[str, Any]]:
    """Coerce the agent's emitted trajectory into call dicts.

    The current agent emits ``[{name, arguments, result, tool_use_id}, ...]``.
    A list of bare tool-name strings (older agent builds, or other runtimes) is
    accepted too so the adapter degrades gracefully instead of erroring.
    """
    if not isinstance(raw, list):
        return []
    calls: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            calls.append(
                {
                    "name": str(item.get("name", "")),
                    "arguments": item.get("arguments", {}) or {},
                    "result": item.get("result", ""),
                    "tool_use_id": str(item.get("tool_use_id") or f"call_{idx}"),
                }
            )
        elif isinstance(item, str):
            calls.append(
                {"name": item, "arguments": {}, "result": "", "tool_use_id": f"call_{idx}"}
            )
    return calls


def make_task_fn(
    runtime_arn: str,
    region: str = "us-east-1",
    session_prefix: str = "eval",
    *,
    payload_extra: dict[str, Any] | None = None,
    input_cost_per_1k: float = _DEFAULT_INPUT_COST_PER_1K,
    output_cost_per_1k: float = _DEFAULT_OUTPUT_COST_PER_1K,
) -> Callable[[Case], TaskFnResult]:
    """Wrap a deployed Amazon Bedrock AgentCore runtime as a ``task_fn``.

    The agent's response payload should include ``result`` (string or
    Bedrock-style ``{role, content:[{text}]}``), ``trajectory`` (ordered tool
    calls), ``available_tools`` (tool names the agent could call), and ``usage``
    (token counts). The adapter rebuilds a Strands ``Session`` from these so the
    Session-level judges run for real, and returns per-turn latency/token/cost
    via ``environment_state`` for the domain evaluators.

    Each invocation gets a fresh runtime session id. AgentCore requires
    ``runtimeSessionId`` to be 33-256 chars; ``"{prefix}-{uuid4hex}"`` is at
    least 1 + 1 + 32 = 34 chars, so any non-empty prefix is valid. Pass
    ``session_prefix`` to scope sessions to a run.

    Args:
        runtime_arn: ARN of the deployed AgentCore runtime.
        region: AWS region of the runtime.
        session_prefix: Prefix for generated ``runtimeSessionId`` values.
        payload_extra: Extra fields merged into every request payload alongside
            ``prompt`` (e.g. ``{"dealer_id": "DLR24946"}`` to supply the dealer
            context the reference agent reads). ``prompt`` always wins on key
            collision so a case input can never be overridden.
        input_cost_per_1k: USD per 1K input tokens (for CostEvaluator).
        output_cost_per_1k: USD per 1K output tokens (for CostEvaluator).
    """
    client = boto3.client("bedrock-agentcore", region_name=region)
    base_payload = dict(payload_extra or {})

    def task_fn(case: Case) -> TaskFnResult:
        start_dt = datetime.now(timezone.utc)
        start = time.perf_counter()
        session_id = f"{session_prefix}-{uuid.uuid4().hex}"

        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            runtimeSessionId=session_id,
            payload=json.dumps({**base_payload, "prompt": case.input}).encode(),
            contentType="application/json",
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        end_dt = datetime.now(timezone.utc)
        body = json.loads(response["response"].read())

        output_text = _extract_text(body.get("result"))
        tool_calls = _normalize_trajectory(body.get("trajectory", []))
        available_tools = [t for t in body.get("available_tools", []) if isinstance(t, str)]

        usage = body.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        estimated_cost = (
            input_tokens / 1000 * input_cost_per_1k + output_tokens / 1000 * output_cost_per_1k
        )

        session = build_session(
            session_id=session_id,
            user_prompt=str(case.input),
            agent_response=output_text,
            tool_calls=tool_calls,
            available_tools=available_tools,
            start=start_dt,
            end=end_dt,
        )

        # environment_state is the channel that survives onto EvaluationData;
        # task-result metadata is dropped by strands_evals. LatencyEvaluator and
        # CostEvaluator read these keys from actual_environment_state.
        metrics_state = {
            "latency_ms": elapsed_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "runtime_arn": runtime_arn,
            "session_id": session_id,
        }

        return {
            "output": output_text,
            "trajectory": session,
            "environment_state": [EnvironmentState(name="metrics", state=metrics_state)],
            "metadata": metrics_state,
        }

    return task_fn
