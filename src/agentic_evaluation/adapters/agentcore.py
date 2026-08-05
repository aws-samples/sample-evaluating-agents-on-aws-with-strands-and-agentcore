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
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
from strands_evals import Case
from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.adapters._session import build_session
from agentic_evaluation.exceptions import TaskFnError
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
    runtime_user_id: str | None = None,
    evaluation_token: str | None = None,
    evaluation_secret_id: str | None = None,
    payload_extra: dict[str, Any] | None = None,
    input_cost_per_1k: float = _DEFAULT_INPUT_COST_PER_1K,
    output_cost_per_1k: float = _DEFAULT_OUTPUT_COST_PER_1K,
    boto3_session: boto3.Session | None = None,
    boto_client_config: BotocoreConfig | None = None,
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
        runtime_user_id: User/dealer identity resolved by a trusted upstream
            service. AgentCore sends it through the dedicated ``runtimeUserId``
            channel rather than the caller-controlled JSON body. Production
            end-user deployments should prefer the JWT-authorizer path.
        evaluation_token: Privileged token authorizing trajectory and usage
            telemetry. Prefer ``evaluation_secret_id`` outside tests.
        evaluation_secret_id: Secrets Manager secret containing the privileged
            evaluation token. The adapter retrieves it with the selected region.
        payload_extra: Extra fields merged into every request payload alongside
            ``prompt``. Identity fields are rejected; ``prompt`` always wins on
            key collision so a case input can never be overridden.
        input_cost_per_1k: USD per 1K input tokens (for CostEvaluator).
        output_cost_per_1k: USD per 1K output tokens (for CostEvaluator).
        boto3_session: Optional explicit AWS session used for runtime and secret
            clients.
        boto_client_config: Optional botocore client configuration. The default
            bounds connection/read time and retries so a stalled invocation
            fails instead of blocking an evaluation indefinitely.
    """
    client_factory = boto3_session.client if boto3_session is not None else boto3.client
    client_config = boto_client_config or BotocoreConfig(
        connect_timeout=5,
        read_timeout=120,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    client = client_factory("bedrock-agentcore", region_name=region, config=client_config)
    if evaluation_token and evaluation_secret_id:
        raise ValueError("Pass only one of evaluation_token or evaluation_secret_id")
    if evaluation_secret_id:
        secret_response = client_factory(
            "secretsmanager",
            region_name=region,
            config=client_config,
        ).get_secret_value(SecretId=evaluation_secret_id)
        evaluation_token = secret_response.get("SecretString")
        if not isinstance(evaluation_token, str) or not evaluation_token:
            raise ValueError("Evaluation secret has no SecretString")
    base_payload = dict(payload_extra or {})
    forbidden_identity_fields = {"actor_id", "dealer_id"} & base_payload.keys()
    if forbidden_identity_fields:
        fields = ", ".join(sorted(forbidden_identity_fields))
        raise ValueError(f"Identity fields must use runtime_user_id, not payload_extra: {fields}")
    conversation_sessions: dict[str, str] = {}
    conversation_lock = threading.Lock()

    def task_fn(case: Case) -> TaskFnResult:
        start_dt = datetime.now(timezone.utc)
        start = time.perf_counter()
        metadata = case.metadata or {}
        conversation_id = metadata.get("conversation_id")
        run_id = metadata.get("evaluation_run_id", "default")
        if conversation_id:
            conversation_key = f"{run_id}:{conversation_id}"
            with conversation_lock:
                session_id = conversation_sessions.setdefault(
                    conversation_key, f"{session_prefix}-{uuid.uuid4().hex}"
                )
        else:
            conversation_key = None
            session_id = f"{session_prefix}-{uuid.uuid4().hex}"

        request_payload = {**base_payload, "prompt": case.input}
        if evaluation_token:
            request_payload["evaluation_token"] = evaluation_token
        invocation: dict[str, Any] = {
            "agentRuntimeArn": runtime_arn,
            "runtimeSessionId": session_id,
            "payload": json.dumps(request_payload).encode(),
            "contentType": "application/json",
        }
        if runtime_user_id:
            invocation["runtimeUserId"] = runtime_user_id

        response = client.invoke_agent_runtime(
            **invocation,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        end_dt = datetime.now(timezone.utc)
        body = json.loads(response["response"].read())
        required_trace_fields = {"trajectory", "available_tools", "usage"}
        missing_trace_fields = required_trace_fields - body.keys()
        if missing_trace_fields:
            missing = ", ".join(sorted(missing_trace_fields))
            raise TaskFnError(
                f"Runtime did not return authorized evaluation telemetry; missing fields: {missing}"
            )

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
        last_refresh_time = body.get("last_refresh_time")
        if isinstance(last_refresh_time, str) and last_refresh_time:
            metrics_state["last_refresh_time"] = last_refresh_time
        lancedb_version = body.get("lancedb_version")
        if isinstance(lancedb_version, str) and lancedb_version:
            metrics_state["lancedb_version"] = lancedb_version
        if conversation_key and metadata.get("turn_index") == metadata.get("turn_count"):
            with conversation_lock:
                conversation_sessions.pop(conversation_key, None)

        return {
            "output": output_text,
            "trajectory": session,
            "environment_state": [EnvironmentState(name="metrics", state=metrics_state)],
            "metadata": metrics_state,
        }

    return task_fn
