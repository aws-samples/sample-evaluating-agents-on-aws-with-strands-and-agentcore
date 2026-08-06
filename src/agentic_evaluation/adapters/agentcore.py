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
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config as BotocoreConfig
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
from agentic_evaluation.exceptions import TaskFnError
from agentic_evaluation.types import TaskFnResult

# Telemetry the runtime must return for the evaluators to have anything real to
# score. Absent fields mean the caller was not authorized for it, so the adapter
# fails loudly rather than silently evaluating a degraded trace.
_REQUIRED_TRACE_FIELDS = frozenset({"trajectory", "available_tools", "usage"})

# Identity must travel through AgentCore's dedicated ``runtimeUserId`` channel,
# never the caller-controlled JSON body.
_IDENTITY_FIELDS = frozenset({"actor_id", "dealer_id"})

_ClientFactory = Callable[..., Any]


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


def _resolve_evaluation_token(
    token: str | None,
    secret_id: str | None,
    client_factory: _ClientFactory,
    region: str,
    config: BotocoreConfig,
) -> str | None:
    """Resolve the privileged evaluation token from its literal or secret form.

    Args:
        token: A literal token, or None.
        secret_id: A Secrets Manager secret id, or None.
        client_factory: Builds the Secrets Manager client.
        region: Region holding the secret.
        config: Botocore config for the secrets client.

    Returns:
        The token, or None when neither form was supplied.

    Raises:
        ValueError: Both forms were supplied, or the secret has no
            ``SecretString``.
    """
    if token and secret_id:
        raise ValueError("Pass only one of evaluation_token or evaluation_secret_id")
    if not secret_id:
        return token
    secret = client_factory(
        "secretsmanager",
        region_name=region,
        config=config,
    ).get_secret_value(SecretId=secret_id)
    resolved = secret.get("SecretString")
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("Evaluation secret has no SecretString")
    return resolved


def _validate_payload_extra(payload_extra: dict[str, Any] | None) -> dict[str, Any]:
    """Copy the caller's extra payload fields after rejecting identity claims.

    Args:
        payload_extra: Extra fields to merge into every request.

    Returns:
        A copy safe to use as the base payload.

    Raises:
        ValueError: The extras carry an identity field, which would let the
            caller impersonate a user through the JSON body.
    """
    base = dict(payload_extra or {})
    forbidden = _IDENTITY_FIELDS & base.keys()
    if forbidden:
        fields = ", ".join(sorted(forbidden))
        raise ValueError(f"Identity fields must use runtime_user_id, not payload_extra: {fields}")
    return base


def _read_telemetry(body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], TokenUsage]:
    """Extract the evaluation telemetry from a runtime response body.

    Args:
        body: The decoded response payload.

    Returns:
        The normalised tool calls, the tool names the agent could call, and the
        turn's token usage.

    Raises:
        TaskFnError: The runtime withheld telemetry, so there is no genuine
            trajectory to evaluate.
    """
    missing_fields = _REQUIRED_TRACE_FIELDS - body.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise TaskFnError(
            f"Runtime did not return authorized evaluation telemetry; missing fields: {missing}"
        )
    usage = body.get("usage", {}) or {}
    return (
        _normalize_trajectory(body.get("trajectory", [])),
        [tool for tool in body.get("available_tools", []) if isinstance(tool, str)],
        TokenUsage.from_counts(
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        ),
    )


def _data_freshness(body: dict[str, Any]) -> dict[str, str]:
    """Pull the optional data-freshness fields the domain evaluators can use.

    Args:
        body: The decoded response payload.

    Returns:
        Only the freshness keys the runtime actually reported, so
        :class:`~agentic_evaluation.evaluators.DataFreshnessEvaluator` can tell
        "not reported" from "reported as stale".
    """
    reported = {
        "last_refresh_time": body.get("last_refresh_time"),
        "lancedb_version": body.get("lancedb_version"),
    }
    return {key: value for key, value in reported.items() if isinstance(value, str) and value}


# PLR0913 (10 > 5 args): all but ``runtime_arn`` are independent, documented
# deployment knobs, and every one past ``session_prefix`` is keyword-only — the
# argument-transposition bug the rule guards against cannot occur. Bundling them
# into a config object would relocate the same ten values behind one more
# indirection without removing any of them.
def make_task_fn(  # noqa: PLR0913
    runtime_arn: str,
    region: str = "us-east-1",
    session_prefix: str = "eval",
    *,
    runtime_user_id: str | None = None,
    evaluation_token: str | None = None,
    evaluation_secret_id: str | None = None,
    payload_extra: dict[str, Any] | None = None,
    pricing: TokenPricing = DEFAULT_PRICING,
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
        pricing: Per-1K-token rates used to estimate cost for
            :class:`~agentic_evaluation.evaluators.CostEvaluator`.
        boto3_session: Optional explicit AWS session used for runtime and secret
            clients.
        boto_client_config: Optional botocore client configuration. The default
            bounds connection/read time and retries so a stalled invocation
            fails instead of blocking an evaluation indefinitely.

    Returns:
        A ``task_fn`` that invokes the runtime once per case.

    .. versionchanged:: 0.4.0
        ``input_cost_per_1k`` / ``output_cost_per_1k`` are replaced by the
        ``pricing`` value object.
    """
    client_factory = boto3_session.client if boto3_session is not None else boto3.client
    client_config = boto_client_config or BotocoreConfig(
        connect_timeout=5,
        read_timeout=120,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    client = client_factory("bedrock-agentcore", region_name=region, config=client_config)
    token = _resolve_evaluation_token(
        evaluation_token, evaluation_secret_id, client_factory, region, client_config
    )
    base_payload = _validate_payload_extra(payload_extra)
    sessions: ConversationScope[str] = ConversationScope(
        lambda: f"{session_prefix}-{uuid.uuid4().hex}"
    )

    def build_invocation(case: Case, session_id: str) -> dict[str, Any]:
        """Build the ``invoke_agent_runtime`` kwargs for one case."""
        request_payload = {**base_payload, "prompt": case.input}
        if token:
            request_payload["evaluation_token"] = token
        invocation: dict[str, Any] = {
            "agentRuntimeArn": runtime_arn,
            "runtimeSessionId": session_id,
            "payload": json.dumps(request_payload).encode(),
            "contentType": "application/json",
        }
        if runtime_user_id:
            invocation["runtimeUserId"] = runtime_user_id
        return invocation

    def task_fn(case: Case) -> TaskFnResult:
        metadata = case.metadata or {}
        key = conversation_key(metadata)
        session_id = sessions.acquire(key)

        start_dt = datetime.now(UTC)
        start = time.perf_counter()
        response = client.invoke_agent_runtime(**build_invocation(case, session_id))
        elapsed_ms = (time.perf_counter() - start) * 1000
        end_dt = datetime.now(UTC)

        body = json.loads(response["response"].read())
        tool_calls, available_tools, usage = _read_telemetry(body)
        output_text = _extract_text(body.get("result"))

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
            **base_metrics(elapsed_ms, usage, pricing),
            **_data_freshness(body),
            "runtime_arn": runtime_arn,
            "session_id": session_id,
        }
        sessions.release(key, metadata)

        return {
            "output": output_text,
            "trajectory": session,
            "environment_state": [EnvironmentState(name="metrics", state=metrics_state)],
            "metadata": metrics_state,
        }

    return task_fn
