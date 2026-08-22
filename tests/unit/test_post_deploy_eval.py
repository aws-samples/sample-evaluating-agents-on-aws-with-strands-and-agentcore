# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the deployed-runtime smoke evaluation summary.

This script is the deploy gate: it decides, from a live runtime, whether the
deployment is good. Its own summary step used to read a report-level
`evaluator_name` that 1.x removed, which raised `AttributeError` only after the
runtime had been invoked -- so the failure was unreachable in CI and guaranteed
on the deploy path. These tests drive the real layer pipeline with a stub
`task_fn` so that path is exercised without AWS credentials.
"""

from datetime import UTC, datetime

from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.run_experiment import run_all_layers
from scripts.post_deploy_eval import _smoke_cases, _summarize


def _compliant_task_fn(case):
    """Stand in for the deployed runtime: refuses both safety cases, cheap and fast."""
    metrics = {
        "latency_ms": 100,
        "total_tokens": 50,
        "estimated_cost_usd": 0.0001,
        # Freshness is measured against now, so a fixed literal would rot.
        "last_refresh_time": datetime.now(UTC).isoformat(),
    }
    return {
        "output": "I cannot place bids or predict winning chances. Hello.",
        "trajectory": [],
        "environment_state": [EnvironmentState(name="metrics", state=metrics)],
        "metadata": metrics,
    }


def _run_smoke_summary():
    return _summarize(
        run_all_layers(
            task_fn=_compliant_task_fn,
            registry=_smoke_cases(),
            judge_backend="noop",
            layers=["layer_1", "domain"],
        )
    )


def test_summary_names_every_evaluator_that_ran():
    # The names come from a per-row tag, not a report attribute. Asserting the
    # exact set is what makes a silent rename or a missing tag fail here rather
    # than in a deploy log.
    summary = _run_smoke_summary()

    assert [e["name"] for e in summary["layers"]["layer_1"]["evaluators"]] == [
        "ToolSelectionGrader",
        "ToolParameterGrader",
        "TrajectoryOrderGrader",
    ]
    assert [e["name"] for e in summary["layers"]["domain"]["evaluators"]] == [
        "DataFreshnessEvaluator",
        "SafetyGuardrailEvaluator",
        "LatencyEvaluator",
        "CostEvaluator",
    ]


def test_summary_reports_a_clean_run_as_passing():
    summary = _run_smoke_summary()

    assert summary["all_passed"] is True
    assert summary["layers"]["layer_1"]["passed"] is True
    assert summary["layers"]["domain"]["passed"] is True
    # Two of the three smoke cases declare layer_1; all three reach the domain
    # layer, which is non-functional and applies to every case.
    assert all(e["test_passes"] == [True, True] for e in summary["layers"]["layer_1"]["evaluators"])
    assert all(
        e["test_passes"] == [True, True, True] for e in summary["layers"]["domain"]["evaluators"]
    )


def test_summary_omits_layers_that_were_not_run():
    # `--layers layer_1,domain` is the default precisely so no Bedrock judge
    # permissions are needed; the summary must not invent empty judge layers,
    # which would read as "the judge ran and found nothing".
    summary = _run_smoke_summary()

    assert set(summary["layers"]) == {"layer_1", "domain"}


def test_a_bidding_agent_fails_the_safety_gate():
    # The positive control for the gate itself: if the deployed agent does the
    # two things eval_config.yaml forbids -- calling a bid tool and claiming
    # winning odds -- the script has to exit non-zero. A gate that has never
    # been seen to go red is not a gate.
    def violating_task_fn(case):
        result = _compliant_task_fn(case)
        result["output"] = "Placed. Your chances of winning are excellent."
        result["trajectory"] = ["place_bid"]
        return result

    summary = _summarize(
        run_all_layers(
            task_fn=violating_task_fn,
            registry=_smoke_cases(),
            judge_backend="noop",
            layers=["layer_1", "domain"],
        )
    )

    assert summary["all_passed"] is False
    # Both layers catch it, for different reasons: an unexpected tool call and a
    # forbidden action plus a forbidden phrase.
    assert summary["layers"]["layer_1"]["passed"] is False
    assert summary["layers"]["domain"]["passed"] is False
    by_name = {e["name"]: e for layer in summary["layers"].values() for e in layer["evaluators"]}
    assert by_name["ToolSelectionGrader"]["test_passes"] == [False, False]
    assert by_name["SafetyGuardrailEvaluator"]["test_passes"] == [False, False, False]
    assert "place_bid" in by_name["SafetyGuardrailEvaluator"]["reasons"][0]
    assert "chances of winning" in by_name["SafetyGuardrailEvaluator"]["reasons"][0]
