# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for isolated local Strands evaluation."""

from types import SimpleNamespace

import pytest
from strands_evals import Case

from agentic_evaluation.adapters.strands_local import make_task_fn


class _FakeAgent:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.tool_names = ["search_vehicles"]

    def __call__(self, prompt: str) -> SimpleNamespace:
        use_id = f"use-{prompt}"
        self.messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": use_id,
                                "name": "search_vehicles",
                                "input": {"query": prompt},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": use_id,
                                "content": [{"text": prompt}],
                            }
                        }
                    ],
                },
            ]
        )
        return SimpleNamespace(
            message={"content": [{"text": f"answer:{prompt}"}]},
            metrics=SimpleNamespace(
                accumulated_usage={"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
            ),
            stop_reason="end_turn",
        )


def test_shared_agent_history_is_isolated_and_restored() -> None:
    agent = _FakeAgent()
    task_fn = make_task_fn(agent)

    first = task_fn(Case[str, str](name="first", input="diesel"))
    second = task_fn(Case[str, str](name="second", input="electric"))

    assert first["output"] == "answer:diesel"
    assert second["output"] == "answer:electric"
    assert agent.messages == []
    second_trace = second["trajectory"].traces[0]
    assert second_trace.spans[1].tool_call.arguments == {"query": "electric"}


def test_agent_factory_creates_one_agent_per_case() -> None:
    created: list[_FakeAgent] = []

    def factory() -> _FakeAgent:
        created.append(_FakeAgent())
        return created[-1]

    task_fn = make_task_fn(agent_factory=factory)
    task_fn(Case[str, str](name="first", input="one"))
    task_fn(Case[str, str](name="second", input="two"))

    assert len(created) == 2


def test_requires_exactly_one_agent_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        make_task_fn()
