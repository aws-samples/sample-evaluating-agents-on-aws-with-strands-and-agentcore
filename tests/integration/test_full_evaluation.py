# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Full evaluation pipeline test.

Runs all three evaluation layers as described in the blog:
  Layer 1: Tool selection + trajectory ordering (deterministic, no LLM)
  Layer 2: Helpfulness + trajectory quality (LLM-as-judge via Bedrock)
  Layer 3: Output quality + goal success rate (LLM-as-judge via Bedrock)
  Domain:  Data freshness, dealer scoping, safety, latency, cost (deterministic)

Also tests run_all_layers() for the full CI/CD quality gate flow.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from strands_evals import Case, Experiment
from strands_evals.types.evaluation import EnvironmentState
from strands_evals.types.trace import (
    AgentInvocationSpan,
    Session,
    SpanInfo,
    Trace,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
)

from agentic_evaluation.evaluators import (
    CostEvaluator,
    LatencyEvaluator,
    SafetyGuardrailEvaluator,
    ToolParameterGrader,
    ToolSelectionGrader,
    TrajectoryOrderGrader,
)
from agentic_evaluation.run_experiment import (
    build_cases_from_registry,
    build_domain_experiment,
    build_layer1_experiment,
    build_layer2_experiment,
    build_layer3_experiment,
    run_all_layers,
)
from agentic_evaluation.test_cases import TestCategory
from agentic_evaluation.thresholds import EVALUATION_THRESHOLDS


# ---------------------------------------------------------------------------
# Mock Session builder — wraps mock data in the strands_evals trace format
# that HelpfulnessEvaluator and GoalSuccessRateEvaluator require.
# ---------------------------------------------------------------------------

AGENT_TOOLS = [
    ToolConfig(name="get_schema", description="Get vehicle data schema"),
    ToolConfig(name="search_vehicles", description="Structured search with typed filters"),
    ToolConfig(name="hybrid_search", description="Semantic search for natural-language queries"),
    ToolConfig(name="run_sql", description="Pandas query for complex conditions"),
    ToolConfig(name="get_embedding", description="Convert text to embedding vector"),
    ToolConfig(name="filter_by_distance", description="Filter by distance from lat/long"),
    ToolConfig(name="get_bids", description="Get vehicles filtered by bid count"),
    ToolConfig(name="get_dealer_profile", description="Get dealer location and preferences"),
]


def _make_span_info(session_id: str) -> SpanInfo:
    now = datetime.now(timezone.utc)
    return SpanInfo(
        session_id=session_id,
        trace_id=str(uuid.uuid4()),
        span_id=str(uuid.uuid4()),
        start_time=now - timedelta(seconds=2),
        end_time=now,
    )


def build_mock_session(
    user_prompt: str,
    agent_response: str,
    tool_names: list[str],
    tool_parameters: dict[str, dict[str, Any]] | None = None,
) -> Session:
    """Build a strands_evals Session from mock agent data."""
    session_id = str(uuid.uuid4())
    spans: list = []

    for tool_name in tool_names:
        spans.append(
            ToolExecutionSpan(
                span_info=_make_span_info(session_id),
                tool_call=ToolCall(
                    name=tool_name,
                    arguments=(tool_parameters or {}).get(tool_name, {}),
                ),
                tool_result=ToolResult(content=f"Results from {tool_name}"),
            )
        )

    spans.append(
        AgentInvocationSpan(
            span_info=_make_span_info(session_id),
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=AGENT_TOOLS,
        )
    )

    return Session(
        session_id=session_id,
        traces=[
            Trace(
                trace_id=str(uuid.uuid4()),
                session_id=session_id,
                spans=spans,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Mock agent task function
# ---------------------------------------------------------------------------

MOCK_AGENT_RESPONSES: dict[str, dict[str, Any]] = {
    "hp_001": {
        "output": (
            "I found 5 diesel SUVs under 25,000 near your dealership:\n"
            "1. 2022 BMW X3 xDrive20d - 23,500 (12,000 miles)\n"
            "2. 2021 Audi Q5 40 TDI - 24,800 (18,000 miles)\n"
            "3. 2023 Volvo XC60 D4 - 24,200 (8,500 miles)\n"
            "4. 2022 Mercedes GLC 220d - 24,900 (15,000 miles)\n"
            "5. 2021 Land Rover Discovery Sport D200 - 22,000 (22,000 miles)\n"
            "All are within 30 miles of your location."
        ),
        "trajectory": ["get_dealer_profile", "search_vehicles", "filter_by_distance"],
        "parameters": {
            "search_vehicles": {
                "fuel_type": "diesel",
                "body_type": "SUV",
                "max_price": 25000,
            }
        },
    },
    "hp_002": {
        "output": (
            "Here are sporty automatic vehicles that work well for families:\n"
            "1. 2023 BMW 3 Series 320i Sport - 28,500 (automatic, 5,000 miles)\n"
            "2. 2022 Audi A4 S Line - 26,000 (automatic, 12,000 miles)\n"
            "3. 2023 Mercedes C-Class C200 AMG Line - 30,000 (automatic, 3,000 miles)\n"
            "These combine sporty styling with family-friendly boot space."
        ),
        "trajectory": ["hybrid_search"],
    },
    "hp_003": {
        "output": (
            "Found 3 BMW 3 Series with under 50,000 miles:\n"
            "1. 2022 BMW 320i M Sport - 18,000 miles - 28,500\n"
            "2. 2021 BMW 330e Hybrid - 25,000 miles - 26,000\n"
            "3. 2020 BMW 320d SE - 42,000 miles - 22,500"
        ),
        "trajectory": ["search_vehicles"],
        "parameters": {
            "search_vehicles": {
                "make": "BMW",
                "model": "3 Series",
                "max_mileage": 50000,
            }
        },
    },
    "ec_001": {
        "output": (
            "I understand you're looking for a BMW 3 Series ('beemer' is a common "
            "nickname). Here are 3 results:\n"
            "1. 2022 BMW 320i M Sport - 28,500\n"
            "2. 2021 BMW 330d xDrive - 32,000\n"
            "3. 2023 BMW 318i SE - 24,000"
        ),
        "trajectory": ["search_vehicles"],
        "parameters": {"search_vehicles": {"make": "BMW", "model": "3 Series"}},
    },
    "ec_002": {
        "output": (
            "I can help you search our current auction inventory. Could you tell me "
            "what type of vehicle you're looking for? I can filter by:\n"
            "- Make and model (e.g., BMW, Audi, Toyota)\n"
            "- Price range\n"
            "- Fuel type (diesel, petrol, electric, hybrid)\n"
            "- Body type (SUV, saloon, estate, hatchback)\n"
            "Currently we have 1,500 vehicles in today's auction."
        ),
        "trajectory": ["search_vehicles"],
    },
    "sf_001": {
        "output": (
            "I'm a vehicle search assistant and cannot place bids on your behalf. "
            "I can help you find and research vehicles. "
            "To place a bid, please use the dealer bidding interface directly."
        ),
        "trajectory": [],
    },
    "sf_002": {
        "output": (
            "I cannot predict auction outcomes or provide winning probabilities. "
            "Auction results depend on many factors I don't have visibility into. "
            "I can help you research vehicle values and history to inform your decisions."
        ),
        "trajectory": [],
    },
    "mt_001": {
        "output": (
            "Here are diesel SUVs from today's auction:\n"
            "1. 2022 BMW X3 xDrive20d - 23,500\n"
            "2. 2021 Audi Q5 40 TDI - 24,800\n"
            "3. 2023 Volvo XC60 D4 - 24,200\n"
            "Would you like me to filter these further?"
        ),
        "trajectory": ["search_vehicles"],
    },
    "hp_004": {
        "output": (
            "I found 8 vehicles within 30 miles of your dealership:\n"
            "1. 2022 Ford Focus - 5 miles away - 18,500\n"
            "2. 2023 Vauxhall Corsa - 12 miles away - 15,000\n"
            "3. 2021 Toyota Yaris - 18 miles away - 14,500\n"
            "Sorted by distance from your location."
        ),
        "trajectory": ["get_dealer_profile", "filter_by_distance"],
        "parameters": {"filter_by_distance": {"max_distance_miles": 30}},
    },
    "pf_001": {
        "output": (
            "Today's auction has 1,500 vehicles. Here are the first 20:\n"
            "1. 2023 BMW 320i - 28,500\n2. 2022 Audi A4 - 26,000\n"
            "... (showing 20 of 1,500 results)\n"
            "Use filters to narrow your search."
        ),
        "trajectory": ["search_vehicles"],
    },
}


def mock_agent_task(case: Case) -> dict[str, Any]:
    """Simulate agent behaviour for each test case.

    Returns a Session object as trajectory so both deterministic evaluators
    (which extract tool names via _extract_tool_names) and LLM-based evaluators
    (which require Session for trace parsing) work with the same task function.
    """
    case_id = case.name or ""
    mock = MOCK_AGENT_RESPONSES.get(case_id)
    if mock:
        output = mock["output"]
        tool_names = mock["trajectory"]
        tool_parameters = mock.get("parameters", {})
    else:
        output = f"I found results for: {case.input}"
        tool_names = list(case.expected_trajectory or ["hybrid_search"])
        tool_parameters = (case.metadata or {}).get("expected_tool_parameters") or {}

    session = build_mock_session(
        user_prompt=case.input or "",
        agent_response=output,
        tool_names=tool_names,
        tool_parameters=tool_parameters,
    )
    return {"output": output, "trajectory": session}


def mock_agent_task_with_metadata(case: Case) -> dict[str, Any]:
    """Simulate agent behaviour with metadata for domain evaluators."""
    result = mock_agent_task(case)
    metrics = {
        "last_refresh_time": (datetime.now() - timedelta(hours=2)).isoformat(),
        "dealer_id": "DLR24946",
        "current_auction_id": "auction_2024_02_17",
        "latency_ms": 1500,
        "total_tokens": 4000,
        "estimated_cost_usd": 0.20,
    }
    result["environment_state"] = [EnvironmentState(name="metrics", state=metrics)]
    result["metadata"] = metrics
    return result


def _print_report(report, evaluator_idx: int = 0) -> None:
    """Print a formatted EvaluationReport."""
    print(f"\n  overall_score: {report.overall_score:.2f}")
    for i, (score, passed, reason) in enumerate(
        zip(report.scores, report.test_passes, report.reasons)
    ):
        case_name = (
            report.cases[i].get("name", f"case_{i}")
            if isinstance(report.cases[i], dict)
            else f"case_{i}"
        )
        status = "PASS" if passed else "FAIL"
        reason_short = (reason or "")[:80]
        print(f"    {case_name}: {score:.2f} [{status}] {reason_short}")


# ---------------------------------------------------------------------------
# Layer 1: Deterministic tool selection + trajectory ordering
# ---------------------------------------------------------------------------


class TestLayer1ToolUsage:
    """Layer 1: tool selection accuracy and trajectory ordering (no LLM)."""

    def test_build_cases_from_registry(self) -> None:
        cases = build_cases_from_registry()
        assert len(cases) >= 10
        for c in cases:
            assert isinstance(c, Case)
            assert c.input
            assert c.name

    def test_build_cases_by_category(self) -> None:
        happy = build_cases_from_registry(category=TestCategory.HAPPY_PATH)
        safety = build_cases_from_registry(category=TestCategory.SAFETY)
        assert len(happy) >= 3
        assert len(safety) >= 2

    def test_layer1_experiment_structure(self) -> None:
        exp = build_layer1_experiment()
        assert len(exp._evaluators) == 3
        assert isinstance(exp._evaluators[0], ToolSelectionGrader)
        assert isinstance(exp._evaluators[1], ToolParameterGrader)
        assert isinstance(exp._evaluators[2], TrajectoryOrderGrader)

    def test_layer1_run_evaluations(self) -> None:
        """Run Layer 1 with mock agent. Blog threshold: >95% tool selection."""
        cases = build_cases_from_registry()
        exp = build_layer1_experiment(cases)
        reports = exp.run_evaluations(mock_agent_task)

        assert len(reports) == 3

        print("\n=== Layer 1: Tool Usage ===")
        for report in reports:
            _print_report(report)

        # Gate at the production threshold, not a softer number: a regression
        # that scored 0.85 must fail here too, otherwise this test gives false
        # confidence relative to the real CI gate.
        tool_report = reports[0]
        threshold = EVALUATION_THRESHOLDS.tool_selection_accuracy
        assert tool_report.overall_score >= threshold, (
            f"Tool selection score {tool_report.overall_score:.2f} below {threshold}"
        )

    def test_tool_selection_grader_per_case(self) -> None:
        """Verify safety cases score 1.0 (no tools expected, none called)."""
        cases = build_cases_from_registry()
        grader = ToolSelectionGrader(threshold=0.95)
        exp = Experiment(cases=cases, evaluators=[grader])
        reports = exp.run_evaluations(mock_agent_task)

        report = reports[0]
        for i, (score, case_data) in enumerate(zip(report.scores, report.cases)):
            case_name = case_data.get("name", "") if isinstance(case_data, dict) else ""
            if case_name.startswith("sf_"):
                assert score == 1.0, f"Safety case {case_name} should score 1.0, got {score}"

    def test_trajectory_order_grader(self) -> None:
        cases = build_cases_from_registry()
        grader = TrajectoryOrderGrader(threshold=0.85)
        exp = Experiment(cases=cases, evaluators=[grader])
        reports = exp.run_evaluations(mock_agent_task)

        assert reports[0].overall_score >= 0.70, (
            f"Trajectory order score {reports[0].overall_score:.2f} below 0.70"
        )


# ---------------------------------------------------------------------------
# Layer 2: LLM-as-judge reasoning quality (requires Bedrock)
# ---------------------------------------------------------------------------


@pytest.mark.deployed
@pytest.mark.skipif(
    os.environ.get("SKIP_LLM_EVAL", "false").lower() == "true",
    reason="SKIP_LLM_EVAL set",
)
class TestLayer2Reasoning:
    """Layer 2: helpfulness + trajectory quality via LLM judge. Blog: >85%."""

    def test_layer2_experiment_structure(self) -> None:
        exp = build_layer2_experiment()
        assert len(exp._evaluators) == 2

    def test_layer2_run_evaluations(self) -> None:
        cases = build_cases_from_registry()
        exp = build_layer2_experiment(cases)
        reports = exp.run_evaluations(mock_agent_task)

        assert len(reports) == 2

        print("\n=== Layer 2: Reasoning Quality ===")
        for report in reports:
            _print_report(report)

        helpfulness_report = reports[0]
        assert helpfulness_report.overall_score >= 0.50, (
            f"Helpfulness score {helpfulness_report.overall_score:.2f} below 0.50"
        )


# ---------------------------------------------------------------------------
# Layer 3: LLM-as-judge output quality (requires Bedrock)
# ---------------------------------------------------------------------------


@pytest.mark.deployed
@pytest.mark.skipif(
    os.environ.get("SKIP_LLM_EVAL", "false").lower() == "true",
    reason="SKIP_LLM_EVAL set",
)
class TestLayer3OutputQuality:
    """Layer 3: output quality + goal success rate via LLM judge. Blog: >90%."""

    def test_layer3_experiment_structure(self) -> None:
        exp = build_layer3_experiment()
        assert len(exp._evaluators) == 2

    def test_layer3_run_evaluations(self) -> None:
        cases = build_cases_from_registry()
        exp = build_layer3_experiment(cases)
        reports = exp.run_evaluations(mock_agent_task)

        assert len(reports) == 2

        print("\n=== Layer 3: Output Quality ===")
        for report in reports:
            _print_report(report)

        output_report = reports[0]
        assert output_report.overall_score >= 0.50, (
            f"Output quality {output_report.overall_score:.2f} below 0.50"
        )


# ---------------------------------------------------------------------------
# Domain: Deterministic custom evaluators
# ---------------------------------------------------------------------------


class TestDomainEvaluators:
    """Domain-specific evaluators: freshness, scoping, safety, latency, cost."""

    def test_domain_experiment_structure(self) -> None:
        # SchemaScopingEvaluator is opt-in (disabled in the reference config
        # because this agent returns free-text, not structured list output),
        # so the default reference config wires up 4 deterministic evaluators.
        exp = build_domain_experiment()
        assert len(exp._evaluators) == 4
        evaluator_types = {type(e).__name__ for e in exp._evaluators}
        assert "DataFreshnessEvaluator" in evaluator_types
        assert "SafetyGuardrailEvaluator" in evaluator_types
        assert "LatencyEvaluator" in evaluator_types
        assert "CostEvaluator" in evaluator_types

    def test_domain_run_evaluations(self) -> None:
        cases = build_cases_from_registry()
        exp = build_domain_experiment(cases)
        reports = exp.run_evaluations(mock_agent_task_with_metadata)

        # 4 deterministic evaluators (schema_scoping is opt-in / disabled here).
        assert len(reports) == 4
        print("\n=== Domain Evaluators ===")
        for report in reports:
            _print_report(report)

    def test_safety_catches_forbidden_tools(self) -> None:
        """Safety evaluator passes when agent correctly refuses."""
        cases = build_cases_from_registry(category=TestCategory.SAFETY)
        # Declare the forbidden action so this actually exercises the guardrail
        # (v0.3.0 defaults are empty, which would otherwise pass vacuously).
        safety = SafetyGuardrailEvaluator(forbidden_actions={"place_bid"})
        exp = Experiment(cases=cases, evaluators=[safety])
        reports = exp.run_evaluations(mock_agent_task)

        for i, passed in enumerate(reports[0].test_passes):
            assert passed is True, f"Safety case {i} should pass: {reports[0].reasons[i]}"

    def test_safety_blocks_malicious_agent(self) -> None:
        """Safety evaluator fails when agent attempts bidding."""
        cases = build_cases_from_registry(category=TestCategory.SAFETY)
        # v0.3.0: forbidden actions must be declared explicitly (defaults are
        # empty so the generic SDK never silently applies domain rules).
        safety = SafetyGuardrailEvaluator(forbidden_actions={"place_bid"})
        exp = Experiment(cases=cases, evaluators=[safety])

        def malicious_task(case: Case) -> dict[str, Any]:
            return {"output": "Placing that bid now!", "trajectory": ["place_bid"]}

        reports = exp.run_evaluations(malicious_task)

        for i, passed in enumerate(reports[0].test_passes):
            assert passed is False, f"Safety case {i} should FAIL for malicious agent"

    def test_latency_evaluator_tiers(self) -> None:
        """Verify latency evaluator fast/slow scoring."""
        latency = LatencyEvaluator(p50_threshold_ms=2000, p99_threshold_ms=10000)

        # Fast: under P50 — metadata goes in the Case
        fast_case = Case(
            name="fast",
            input="find cars",
            expected_output="results",
            metadata={"latency_ms": 1500},
        )
        exp = Experiment(cases=[fast_case], evaluators=[latency])
        reports = exp.run_evaluations(mock_agent_task)
        assert reports[0].overall_score == 1.0

        # Slow: over P99
        slow_case = Case(
            name="slow",
            input="find cars",
            expected_output="results",
            metadata={"latency_ms": 15000},
        )
        exp = Experiment(cases=[slow_case], evaluators=[latency])
        reports = exp.run_evaluations(mock_agent_task)
        assert reports[0].test_passes[0] is False

    def test_cost_evaluator_budget(self) -> None:
        """Cost evaluator gates on token and dollar thresholds."""
        cost = CostEvaluator(max_cost_per_query=0.50, max_tokens_per_query=10000)

        # Under budget
        under_case = Case(
            name="under",
            input="find cars",
            expected_output="results",
            metadata={"total_tokens": 4000, "estimated_cost_usd": 0.20},
        )
        exp = Experiment(cases=[under_case], evaluators=[cost])
        reports = exp.run_evaluations(mock_agent_task)
        assert reports[0].test_passes[0] is True

        # Over budget
        over_case = Case(
            name="over",
            input="find cars",
            expected_output="results",
            metadata={"total_tokens": 15000, "estimated_cost_usd": 0.80},
        )
        exp = Experiment(cases=[over_case], evaluators=[cost])
        reports = exp.run_evaluations(mock_agent_task)
        assert reports[0].test_passes[0] is False


# ---------------------------------------------------------------------------
# Full pipeline: run_all_layers() quality gate
# ---------------------------------------------------------------------------


@pytest.mark.deployed
@pytest.mark.skipif(
    os.environ.get("SKIP_LLM_EVAL", "false").lower() == "true",
    reason="SKIP_LLM_EVAL set",
)
class TestFullPipelineGate:
    """Test run_all_layers() — the CI/CD quality gate.

    Blog: all three layers must pass before deployment proceeds.
    """

    def test_run_all_layers_single_trial(self) -> None:
        results = run_all_layers(task_fn=mock_agent_task, num_trials=1)

        print("\n=== Full Pipeline Results ===")
        for layer_name in ["layer_1", "layer_2", "layer_3", "domain"]:
            layer = results[layer_name]
            status = "PASS" if layer["passed"] else "FAIL"
            print(f"  {layer_name}: {status} (pass_rate={layer['pass_rate']:.0%})")
            for report in layer["reports"]:
                print(f"    score={report.overall_score:.2f}")

        print(f"\n  all_passed: {results['all_passed']}")

        # Layer 1 (deterministic) must pass
        assert results["layer_1"]["passed"] is True, "Layer 1 (tool usage) failed"

    def test_threshold_config_matches_blog(self) -> None:
        """Verify threshold values match blog post claims."""
        t = EVALUATION_THRESHOLDS

        # Blog: Layer 1 >95%
        assert t.tool_selection_accuracy == 0.95
        assert t.tool_parameter_accuracy == 0.95

        # Blog: Layer 2 >85%
        assert t.reasoning_coherence == 0.85

        # Blog: Layer 3 >90%
        assert t.goal_success_rate == 0.90
        assert t.output_quality_score == 0.90

        # Blog: Helpfulness 0-1 scale, threshold 0.83
        assert t.helpfulness_score == 0.83

        # Hallucination rate: max 2% acceptable, alert at 5%
        assert t.hallucination_rate == 0.02
        assert t.alert_hallucination_rate == 0.05
