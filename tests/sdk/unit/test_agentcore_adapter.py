# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the deployed AgentCore task adapter."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from strands_evals import Case

from agentic_evaluation.adapters.agentcore import make_task_fn
from agentic_evaluation.exceptions import TaskFnError


def test_rejects_identity_in_payload_extra() -> None:
    with (
        patch("agentic_evaluation.adapters.agentcore.boto3.client"),
        pytest.raises(ValueError, match="runtime_user_id"),
    ):
        make_task_fn(
            "arn:aws:bedrock-agentcore:eu-west-1:111122223333:runtime/test",
            region="eu-west-1",
            payload_extra={"dealer_id": "DLR24946"},
        )


def test_sends_identity_through_runtime_user_id() -> None:
    client = MagicMock()
    client.invoke_agent_runtime.return_value = {
        "response": BytesIO(
            json.dumps(
                {
                    "result": {"role": "assistant", "content": [{"text": "ok"}]},
                    "trajectory": [],
                    "available_tools": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "last_refresh_time": "2026-07-21T01:00:00+00:00",
                    "lancedb_version": "a" * 64,
                }
            ).encode()
        )
    }
    with patch(
        "agentic_evaluation.adapters.agentcore.boto3.client", return_value=client
    ) as client_factory:
        task_fn = make_task_fn(
            "arn:aws:bedrock-agentcore:eu-west-1:111122223333:runtime/test",
            region="eu-west-1",
            runtime_user_id="DLR24946",
            evaluation_token="eval-token",
        )

    result = task_fn(Case[str, str](name="identity", input="hello"))

    invocation = client.invoke_agent_runtime.call_args.kwargs
    assert invocation["runtimeUserId"] == "DLR24946"
    payload = json.loads(invocation["payload"])
    assert "dealer_id" not in payload
    assert payload["evaluation_token"] == "eval-token"
    assert result["metadata"]["last_refresh_time"] == "2026-07-21T01:00:00+00:00"
    assert result["metadata"]["lancedb_version"] == "a" * 64
    runtime_client_config = client_factory.call_args_list[0].kwargs["config"]
    assert runtime_client_config.connect_timeout == 5
    assert runtime_client_config.read_timeout == 120


def test_fails_closed_without_authorized_trace() -> None:
    client = MagicMock()
    client.invoke_agent_runtime.return_value = {
        "response": BytesIO(json.dumps({"result": "public answer"}).encode())
    }
    with patch("agentic_evaluation.adapters.agentcore.boto3.client", return_value=client):
        task_fn = make_task_fn(
            "arn:aws:bedrock-agentcore:eu-west-1:111122223333:runtime/test",
            region="eu-west-1",
        )

    with pytest.raises(TaskFnError, match="authorized evaluation telemetry"):
        task_fn(Case[str, str](name="trace", input="hello"))
