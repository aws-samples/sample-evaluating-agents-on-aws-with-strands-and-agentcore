# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared helper for rebuilding a Strands ``Session`` from a single agent turn.

Both the live AgentCore adapter and the in-process Strands adapter need to turn
a turn's tool calls into a ``strands_evals.Session`` so the Session-level judges
(Helpfulness, GoalSuccessRate) score the genuine trajectory rather than a
degraded tool-name list. Keeping the construction here means the two adapters
emit identical, judge-compatible traces.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


def build_session(
    *,
    session_id: str,
    user_prompt: str,
    agent_response: str,
    tool_calls: list[dict[str, Any]],
    available_tools: list[str],
    start: datetime | None = None,
    end: datetime | None = None,
) -> Session:
    """Assemble a single-turn Strands ``Session`` from one agent invocation.

    Args:
        session_id: Identifier shared by the trace and all spans.
        user_prompt: The user's message for the turn.
        agent_response: The agent's final textual reply.
        tool_calls: Ordered call dicts with ``name``, ``arguments``, ``result``,
            and ``tool_use_id`` keys.
        available_tools: Names of tools the agent could call this turn.
        start: Turn start time (defaults to now, UTC).
        end: Turn end time (defaults to now, UTC).

    Returns:
        A ``Session`` with one trace containing an ``AgentInvocationSpan`` (so
        the trace/session judges have a turn to score) followed by one
        ``ToolExecutionSpan`` per call in invocation order (so the deterministic
        order-graders see the real sequence).
    """
    now = datetime.now(timezone.utc)
    span_info = SpanInfo(
        trace_id=uuid.uuid4().hex,
        span_id=uuid.uuid4().hex[:16],
        session_id=session_id,
        start_time=start or now,
        end_time=end or now,
    )

    spans: list[Any] = [
        AgentInvocationSpan(
            span_info=span_info,
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=[ToolConfig(name=name) for name in available_tools],
        )
    ]
    for call in tool_calls:
        use_id = str(call.get("tool_use_id") or uuid.uuid4().hex[:16])
        result = call.get("result", "")
        spans.append(
            ToolExecutionSpan(
                span_info=span_info,
                tool_call=ToolCall(
                    name=str(call.get("name", "")),
                    arguments=call.get("arguments", {}) or {},
                    tool_call_id=use_id,
                ),
                tool_result=ToolResult(
                    content=result if isinstance(result, str) else json.dumps(result, default=str),
                    tool_call_id=use_id,
                ),
            )
        )

    trace = Trace(spans=spans, trace_id=span_info.trace_id, session_id=session_id)
    return Session(traces=[trace], session_id=session_id)
