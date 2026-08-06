# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Real end-to-end evaluation against the deployed AgentCore runtime.

Invokes the actual agent through the *production* adapter
(:func:`agentic_evaluation.adapters.agentcore.make_task_fn`), so the trajectory,
token usage, latency and cost scored here are the genuine values the agent
emits — not fixtures. The same adapter powers ``scripts/post_deploy_eval.py``,
so this test exercises the exact code path a real deploy uses.

Usage:
    AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:... \\
    EVALUATION_TRACE_SECRET_ID=arn:aws:secretsmanager:... \\
        pytest tests/integration/test_real_agent_eval.py -v -s

Both variables are required. The runtime only returns the privileged trajectory,
tool inventory and token usage to a caller that presents the evaluation token, so
without ``EVALUATION_TRACE_SECRET_ID`` the adapter raises ``TaskFnError`` rather
than scoring a degraded trace. Take the ARN from the ``EvaluationTraceSecretArn``
output of the ``agent-eval-<env>-agent-runtime`` stack.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from strands_evals import Case

from agentic_evaluation.adapters.agentcore import make_task_fn
from agentic_evaluation.run_experiment import (
    build_cases_from_registry,
    build_domain_experiment,
    build_layer1_experiment,
    build_layer2_experiment,
    build_layer3_experiment,
    run_all_layers,
)
from agentic_evaluation.thresholds import EVALUATION_THRESHOLDS
from agentic_evaluation.types import TaskFnResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Resolve the deployed runtime ARN from the environment. The default is a
# documentation placeholder — set AGENT_RUNTIME_ARN (e.g. in a local .env or
# CI secret) to point at your own deployed AgentCore runtime. See .env.example.
_PLACEHOLDER_RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:eu-west-1:123456789012:runtime/agent_eval_runtime_dev-EXAMPLE"
)
RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", _PLACEHOLDER_RUNTIME_ARN)
REGION = os.environ.get("AWS_REGION", "eu-west-1")

# Dealer context for the location-aware cases ("near my dealership"). The
# trusted evaluation caller passes it through AgentCore's runtimeUserId channel,
# never through the JSON body. DLR24946 is seeded by scripts/seed_dealer_data.py.
DEALER_ID = os.environ.get("EVAL_DEALER_ID", "DLR24946")
EVALUATION_SECRET_ID = os.environ.get("EVALUATION_TRACE_SECRET_ID")

pytestmark = pytest.mark.skipif(
    RUNTIME_ARN == _PLACEHOLDER_RUNTIME_ARN,
    reason="Set AGENT_RUNTIME_ARN to a deployed runtime to run real end-to-end tests",
)


# ---------------------------------------------------------------------------
# Task function: the REAL production adapter
# ---------------------------------------------------------------------------


def _real_task_fn():
    """Build the production AgentCore task_fn used by every layer below.

    This is the same adapter ``scripts/post_deploy_eval.py`` ships with: it
    invokes the live runtime, rebuilds a genuine Strands ``Session`` from the
    agent's emitted trajectory, and surfaces real latency/token/cost via
    ``environment_state``. No part of the trajectory or metrics is faked.
    """
    return make_task_fn(
        runtime_arn=RUNTIME_ARN,
        region=REGION,
        session_prefix="pytest-e2e",
        runtime_user_id=DEALER_ID,
        evaluation_secret_id=EVALUATION_SECRET_ID,
    )


def _invoke_once(prompt: str) -> TaskFnResult:
    """Invoke the deployed runtime once via the production adapter.

    Returns the adapter's :class:`~agentic_evaluation.types.TaskFnResult` so callers
    can inspect the real output, trajectory (a Strands ``Session``) and the
    ``metrics`` environment-state entry.
    """
    task_fn = _real_task_fn()
    return task_fn(Case[str, str](name="adhoc", input=prompt))


def _print_case_scores(report) -> None:
    """Print enough case-level evidence to diagnose a live quality failure."""
    for index, (score, passed, reason) in enumerate(
        zip(report.scores, report.test_passes, report.reasons, strict=True)
    ):
        case = report.cases[index]
        case_name = case.get("name", f"case_{index}") if isinstance(case, dict) else f"case_{index}"
        status = "PASS" if passed else "FAIL"
        print(f"    {case_name}: {score:.2f} [{status}] {(reason or '')[:500]}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_task_fn():
    """Production AgentCore task_fn, shared across tests in the module."""
    return _real_task_fn()


@pytest.fixture(scope="module")
def cases():
    """Load all test cases from eval_config.yaml."""
    return build_cases_from_registry()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealAgentInvocation:
    """Verify the agent responds to basic queries."""

    @pytest.mark.deployed
    def test_agent_responds(self) -> None:
        """Validity check: agent returns a non-empty response with real metrics."""
        result = _invoke_once("Find me diesel SUVs under 25k")
        output = result.get("output", "")
        assert output, "Agent returned empty response"

        metrics = _metrics_state(result)
        assert metrics.get("latency_ms", 0) > 0, "Adapter did not record latency"
        assert metrics.get("total_tokens", 0) > 0, "Agent reported zero tokens"
        print(
            f"\nAgent response ({metrics['latency_ms']:.0f}ms, "
            f"{metrics['total_tokens']} tokens): {output[:200]}"
        )


class TestRealLayer1:
    """Layer 1: Tool selection and trajectory (deterministic, against real agent)."""

    @pytest.mark.deployed
    def test_layer1_real_agent(self, cases, real_task_fn) -> None:
        """Run Layer 1 graders against the agent's real tool trajectory.

        The trajectory is the genuine sequence of tool calls the agent made
        (rebuilt from its emitted ``trajectory``), so tool-selection accuracy
        here is a real signal — not tautological and not zero-by-construction.
        """
        exp = build_layer1_experiment(cases)
        reports = exp.run_evaluations(real_task_fn)

        print("\n=== Layer 1: Tool Usage (Real Agent) ===")
        for report in reports:
            print(f"  Evaluator: {type(report).__name__}")
            print(f"  Overall score: {report.overall_score:.2f}")
            _print_case_scores(report)

        tool_report = reports[0]
        print(f"\n  Tool selection score: {tool_report.overall_score:.2f}")
        print(f"  Threshold: {EVALUATION_THRESHOLDS.tool_selection_accuracy}")

        # Structural assertions — these fail if the evaluation pipeline breaks.
        assert reports, "Layer 1 produced no reports"
        for report in reports:
            assert 0.0 <= report.overall_score <= 1.0, (
                f"overall_score {report.overall_score!r} is outside [0.0, 1.0]"
            )


class TestRealLayer2:
    """Layer 2: Reasoning quality via LLM judge (against real agent)."""

    @pytest.mark.deployed
    def test_layer2_real_agent(self, cases, real_task_fn) -> None:
        """Run Layer 2 LLM evaluators against real agent responses."""
        exp = build_layer2_experiment(cases)
        reports = exp.run_evaluations(real_task_fn)

        print("\n=== Layer 2: Reasoning Quality (Real Agent) ===")
        for report in reports:
            print(f"  Overall score: {report.overall_score:.2f}")
            _print_case_scores(report)

        helpfulness_report = reports[0]
        print(f"\n  Helpfulness score: {helpfulness_report.overall_score:.2f}")
        print(f"  Threshold: {EVALUATION_THRESHOLDS.helpfulness_score}")

        assert reports, "Layer 2 produced no reports"
        assert all(0.0 <= r.overall_score <= 1.0 for r in reports), (
            "A Layer 2 overall_score is outside [0.0, 1.0]"
        )


class TestRealLayer3:
    """Layer 3: Output quality via LLM judge (against real agent)."""

    @pytest.mark.deployed
    def test_layer3_real_agent(self, cases, real_task_fn) -> None:
        """Run Layer 3 LLM evaluators against real agent responses."""
        exp = build_layer3_experiment(cases)
        reports = exp.run_evaluations(real_task_fn)

        print("\n=== Layer 3: Output Quality (Real Agent) ===")
        for report in reports:
            print(f"  Overall score: {report.overall_score:.2f}")
            _print_case_scores(report)

        output_report = reports[0]
        print(f"\n  Output quality score: {output_report.overall_score:.2f}")
        print(f"  Threshold: {EVALUATION_THRESHOLDS.output_quality_score}")

        assert reports, "Layer 3 produced no reports"
        assert all(0.0 <= r.overall_score <= 1.0 for r in reports), (
            "A Layer 3 overall_score is outside [0.0, 1.0]"
        )


class TestRealDomain:
    """Domain evaluators against real agent responses."""

    @pytest.mark.deployed
    def test_domain_real_agent(self, cases, real_task_fn) -> None:
        """Run domain evaluators (latency/cost/safety/freshness) on real metrics."""
        exp = build_domain_experiment(cases)
        reports = exp.run_evaluations(real_task_fn)

        print("\n=== Domain Evaluators (Real Agent) ===")
        for report in reports:
            print(f"  Overall score: {report.overall_score:.2f}")

        assert reports, "Domain layer produced no reports"
        assert all(0.0 <= r.overall_score <= 1.0 for r in reports), (
            "A domain overall_score is outside [0.0, 1.0]"
        )


class TestRealFullPipeline:
    """Full pipeline: run_all_layers against the real deployed agent."""

    @pytest.mark.deployed
    def test_run_all_layers_real(self) -> None:
        """The real quality gate — all 4 layers against the live agent."""
        results = run_all_layers(task_fn=_real_task_fn(), num_trials=1)

        print("\n" + "=" * 60)
        print("FULL PIPELINE RESULTS (Real Agent)")
        print("=" * 60)
        for layer_name in ["layer_1", "layer_2", "layer_3", "domain"]:
            layer = results[layer_name]
            status = "PASS" if layer["passed"] else "FAIL"
            print(f"\n  {layer_name}: {status}")
            for report in layer["reports"]:
                print(f"    score: {report.overall_score:.2f}")
                if not report.test_passes or not all(report.test_passes):
                    _print_case_scores(report)

        print(f"\n  ALL LAYERS PASSED: {results['all_passed']}")
        print("=" * 60)

        # Structural assertions verify the pipeline contract, while all_passed
        # makes this test a real release gate instead of a false-green smoke test.
        assert "all_passed" in results, "run_all_layers result missing 'all_passed' key"
        for layer_name in ["layer_1", "layer_2", "layer_3", "domain"]:
            assert isinstance(results[layer_name]["passed"], bool), (
                f"results['{layer_name}']['passed'] is not a bool"
            )
        failed_layers = [
            layer_name
            for layer_name in ["layer_1", "layer_2", "layer_3", "domain"]
            if not results[layer_name]["passed"]
        ]
        assert results["all_passed"] is True, "Live quality gate failed for layers: " + ", ".join(
            failed_layers
        )


def _metrics_state(task_result: TaskFnResult) -> dict[str, Any]:
    """Pull the ``metrics`` environment-state dict out of a TaskFnResult."""
    for env in task_result.get("environment_state", []):
        if getattr(env, "name", None) == "metrics":
            return env.state
    return {}
