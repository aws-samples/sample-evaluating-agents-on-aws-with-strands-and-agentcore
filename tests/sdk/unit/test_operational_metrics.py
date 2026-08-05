# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Operational evaluators must fail when required measurements are absent."""

from datetime import datetime, timezone

from strands_evals import Case, Experiment
from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.adapters._session import build_session
from agentic_evaluation.evaluators import (
    CostEvaluator,
    DataFreshnessEvaluator,
    LatencyEvaluator,
    ToolParameterGrader,
)


def _task_without_metrics(case: Case) -> dict:
    return {"output": "ok", "trajectory": []}


def test_missing_freshness_fails() -> None:
    report = Experiment(
        cases=[Case(name="missing", input="query")],
        evaluators=[DataFreshnessEvaluator()],
    ).run_evaluations(_task_without_metrics)[0]
    assert report.test_passes == [False]
    assert "unavailable" in report.reasons[0]


def test_missing_latency_fails() -> None:
    report = Experiment(
        cases=[Case(name="missing", input="query")],
        evaluators=[LatencyEvaluator()],
    ).run_evaluations(_task_without_metrics)[0]
    assert report.test_passes == [False]
    assert "unavailable" in report.reasons[0]


def test_missing_cost_fails() -> None:
    report = Experiment(
        cases=[Case(name="missing", input="query")],
        evaluators=[CostEvaluator()],
    ).run_evaluations(_task_without_metrics)[0]
    assert report.test_passes == [False]
    assert "unavailable" in report.reasons[0]


def test_complete_metrics_pass() -> None:
    now = datetime.now(timezone.utc).isoformat()

    def task(case: Case) -> dict:
        return {
            "output": "ok",
            "trajectory": [],
            "environment_state": [
                EnvironmentState(
                    name="metrics",
                    state={
                        "last_refresh_time": now,
                        "latency_ms": 10,
                        "total_tokens": 2,
                        "estimated_cost_usd": 0.001,
                    },
                )
            ],
        }

    reports = Experiment(
        cases=[Case(name="complete", input="query")],
        evaluators=[DataFreshnessEvaluator(), LatencyEvaluator(), CostEvaluator()],
    ).run_evaluations(task)
    assert all(report.test_passes == [True] for report in reports)


def test_tool_parameter_mismatch_fails() -> None:
    case = Case(
        name="parameters",
        input="diesel SUVs",
        expected_trajectory=["search_vehicles"],
        metadata={
            "expected_tool_parameters": {
                "search_vehicles": {"fuel_type": "diesel", "body_type": "SUV"}
            }
        },
    )

    def task(case: Case) -> dict:
        session = build_session(
            session_id="parameters",
            user_prompt=str(case.input),
            agent_response="results",
            tool_calls=[
                {
                    "name": "search_vehicles",
                    "arguments": {"fuel_type": "petrol", "body_type": "SUV"},
                    "result": "[]",
                }
            ],
            available_tools=["search_vehicles"],
        )
        return {"output": "results", "trajectory": session}

    report = Experiment(
        cases=[case],
        evaluators=[ToolParameterGrader(threshold=0.95)],
    ).run_evaluations(task)[0]
    assert report.test_passes == [False]
    assert report.scores == [0.5]
