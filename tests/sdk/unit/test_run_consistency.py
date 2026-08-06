# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Regression tests for layer routing and single-run evaluation artifacts."""

import logging
from collections import Counter
from typing import Any

import pytest
from strands_evals import Case

from agentic_evaluation.run_experiment import build_cases_from_registry, run_all_layers
from agentic_evaluation.test_cases import (
    EvaluationLayer,
)
from agentic_evaluation.test_cases import (
    TestCase as EvaluationTestCase,
)
from agentic_evaluation.test_cases import (
    TestCaseRegistry as EvaluationTestCaseRegistry,
)
from agentic_evaluation.test_cases import (
    TestCategory as EvaluationTestCategory,
)


def _case(
    case_id: str,
    layers: list[EvaluationLayer],
    *,
    reference_solution: dict[str, Any] | None = None,
) -> EvaluationTestCase:
    return EvaluationTestCase(
        id=case_id,
        query=f"query:{case_id}",
        category=EvaluationTestCategory.HAPPY_PATH,
        expected_tools=["search_vehicles"],
        expected_behavior="returns results",
        evaluation_layers=layers,
        tags=[],
        reference_solution=reference_solution,
    )


def test_each_case_executes_once_and_routes_by_declared_layer() -> None:
    registry = EvaluationTestCaseRegistry.from_config(
        [
            _case(
                "both",
                [
                    EvaluationLayer.LAYER_1_TOOL_USAGE,
                    EvaluationLayer.LAYER_2_REASONING,
                ],
            ),
            _case("reasoning_only", [EvaluationLayer.LAYER_2_REASONING]),
        ]
    )
    calls: Counter[str] = Counter()

    def task_fn(case: Case) -> dict[str, Any]:
        calls[str(case.name)] += 1
        return {"output": "ok", "trajectory": ["search_vehicles"]}

    results = run_all_layers(
        task_fn,
        registry=registry,
        judge_backend="noop",
        layers=["layer_1", "layer_2"],
    )

    assert calls == {"both": 1, "reasoning_only": 1}
    assert len(results["layer_1"]["reports"][0].scores) == 1
    assert len(results["layer_2"]["reports"][0].scores) == 2


def test_reference_solution_expands_and_executes_in_turn_order() -> None:
    registry = EvaluationTestCaseRegistry.from_config(
        [
            _case(
                "conversation",
                [EvaluationLayer.LAYER_1_TOOL_USAGE],
                reference_solution={
                    "turn_2_query": "second",
                    "turn_2_expected_tools": ["search_vehicles"],
                    "turn_2_behavior": "second result",
                    "turn_3_query": "third",
                    "turn_3_expected_tools": ["search_vehicles"],
                    "turn_3_behavior": "third result",
                },
            )
        ]
    )
    cases = build_cases_from_registry(registry)
    assert [case.input for case in cases] == ["query:conversation", "second", "third"]
    assert [case.metadata["turn_index"] for case in cases] == [1, 2, 3]

    seen: list[str] = []

    def task_fn(case: Case) -> dict[str, Any]:
        seen.append(str(case.input))
        return {"output": "ok", "trajectory": ["search_vehicles"]}

    results = run_all_layers(
        task_fn,
        registry=registry,
        judge_backend="noop",
        layers=["layer_1"],
    )

    assert seen == ["query:conversation", "second", "third"]
    assert len(results["layer_1"]["reports"][0].scores) == 3


def test_each_trial_gets_one_fresh_execution_per_case() -> None:
    registry = EvaluationTestCaseRegistry.from_config(
        [_case("trial_case", [EvaluationLayer.LAYER_1_TOOL_USAGE])]
    )
    calls = 0

    def task_fn(case: Case) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"output": "ok", "trajectory": ["search_vehicles"]}

    run_all_layers(
        task_fn,
        registry=registry,
        judge_backend="noop",
        layers=["layer_1"],
        num_trials=3,
    )

    assert calls == 3


def test_multi_trial_aggregation_gates_on_every_trial() -> None:
    """A run whose *last* trial passes must still fail the gate if an earlier one failed."""
    registry = EvaluationTestCaseRegistry.from_config(
        [_case("flaky", [EvaluationLayer.LAYER_1_TOOL_USAGE])]
    )
    trial = 0

    def task_fn(case: Case) -> dict[str, Any]:
        # One case, so one call per trial: trial 1 picks the wrong tool.
        nonlocal trial
        trial += 1
        tool = "get_schema" if trial == 1 else "search_vehicles"
        return {"output": "ok", "trajectory": [tool]}

    results = run_all_layers(
        task_fn,
        registry=registry,
        judge_backend="noop",
        layers=["layer_1"],
        num_trials=3,
    )

    layer_1 = results["layer_1"]
    assert layer_1["passed"] is True  # the final trial
    assert layer_1["pass_at_k"] is True  # at least one trial
    assert layer_1["pass_all_k"] is False  # not every trial
    assert layer_1["pass_rate"] == 2 / 3
    assert results["num_trials"] == 3
    # pass^k, not the last trial, is the CI gate.
    assert results["all_passed"] is False
    # Reports sample the first trial, which is the one that picked the wrong tool.
    assert 0.0 in layer_1["reports"][0].scores


def test_num_trials_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_trials must be at least 1"):
        run_all_layers(lambda case: {"output": "", "trajectory": []}, num_trials=0)


def test_run_reports_case_and_layer_progress(caplog) -> None:
    registry = EvaluationTestCaseRegistry.from_config(
        [_case("progress", [EvaluationLayer.LAYER_1_TOOL_USAGE])]
    )

    def task_fn(case: Case) -> dict[str, Any]:
        return {"output": "ok", "trajectory": ["search_vehicles"]}

    with caplog.at_level(logging.INFO, logger="agentic_evaluation.run_experiment"):
        run_all_layers(
            task_fn,
            registry=registry,
            judge_backend="noop",
            layers=["layer_1"],
        )

    assert "Executing evaluation case 1/1: progress" in caplog.text
    assert "Completed evaluation case 1/1: progress" in caplog.text
    assert "Completed layer_1 evaluation: passed=True" in caplog.text
