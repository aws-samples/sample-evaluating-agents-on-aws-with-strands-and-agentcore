# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Evaluation thresholds for quality gates.

Following the three-layer evaluation framework from the blog post:
- Layer 1: Tool Usage (>95%)
- Layer 2: Reasoning (>85%)
- Layer 3: Output Quality (>90%)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationThresholds:
    """Threshold configuration for evaluation quality gates."""

    # Layer 1: Tool Usage
    tool_selection_accuracy: float = 0.95
    tool_parameter_accuracy: float = 0.95

    # Layer 2: Reasoning
    helpfulness_score: float = 0.83  # 0-1 scale, "Very Helpful" level
    reasoning_coherence: float = 0.85

    # Layer 3: Output Quality
    goal_success_rate: float = 0.90
    output_quality_score: float = 0.90

    # Domain layer: mean-score bar for operational evaluators (latency, cost,
    # freshness, safety, scoping). Pass/fail is driven primarily by each
    # evaluator's per-case ``test_pass``; this is the secondary aggregate bar.
    domain_aggregate_score: float = 0.90

    # Production monitoring thresholds
    task_completion_rate: float = 0.95
    hallucination_rate: float = 0.02  # Maximum acceptable
    # Latency budget for an *agentic* request: multiple sequential tool calls
    # (search/profile/distance) plus Claude reasoning over the inventory. A
    # single round-trip is ~7-10s; multi-tool cases reach ~18s and the heaviest
    # (cold start + full inventory) ~21s client-side. Tuned to observed live
    # AgentCore latency, not a single-call REST API.
    response_latency_p50_ms: int = 12000
    response_latency_p99_ms: int = 25000

    # Alert thresholds (when to trigger alerts)
    alert_task_completion: float = 0.80
    alert_tool_selection: float = 0.90
    alert_helpfulness: float = 0.58
    alert_latency_p99_ms: int = 30000
    alert_hallucination_rate: float = 0.05

    def to_dict(self) -> dict[str, float | int]:
        """Convert thresholds to dictionary for serialization."""
        return {
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "tool_parameter_accuracy": self.tool_parameter_accuracy,
            "helpfulness_score": self.helpfulness_score,
            "reasoning_coherence": self.reasoning_coherence,
            "goal_success_rate": self.goal_success_rate,
            "output_quality_score": self.output_quality_score,
            "domain_aggregate_score": self.domain_aggregate_score,
            "task_completion_rate": self.task_completion_rate,
            "hallucination_rate": self.hallucination_rate,
            "response_latency_p50_ms": self.response_latency_p50_ms,
            "response_latency_p99_ms": self.response_latency_p99_ms,
        }

    # NOTE: The methods below are a convenience API for callers that have
    # already reduced a layer to scalar metrics (e.g. an external dashboard or
    # a custom CI script). They are NOT the offline-eval gate — that lives in
    # ``run_experiment._layer_passed``, which combines each evaluator's
    # per-case ``test_pass`` with the mean-score threshold so a single failing
    # case (e.g. a safety violation) cannot be averaged away.
    def validate_layer_1(self, tool_accuracy: float, param_accuracy: float) -> bool:
        """Check if Layer 1 (Tool Usage) thresholds are met."""
        return (
            tool_accuracy >= self.tool_selection_accuracy
            and param_accuracy >= self.tool_parameter_accuracy
        )

    def validate_layer_2(self, helpfulness: float, coherence: float) -> bool:
        """Check if Layer 2 (Reasoning) thresholds are met."""
        return helpfulness >= self.helpfulness_score and coherence >= self.reasoning_coherence

    def validate_layer_3(self, goal_success: float, output_quality: float) -> bool:
        """Check if Layer 3 (Output Quality) thresholds are met."""
        return (
            goal_success >= self.goal_success_rate and output_quality >= self.output_quality_score
        )

    # PLR0913 (6 > 5 args): the three layers measure two scores each, and every
    # call site passes them by name. Grouping them into per-layer tuples would
    # satisfy the rule but hide which number is which at the call site, so the
    # six named floats are kept deliberately.
    def validate_all_layers(  # noqa: PLR0913
        self,
        tool_accuracy: float,
        param_accuracy: float,
        helpfulness: float,
        coherence: float,
        goal_success: float,
        output_quality: float,
    ) -> dict[str, bool]:
        """Validate all three layers and return detailed results.

        Args:
            tool_accuracy: Layer 1 tool-selection accuracy.
            param_accuracy: Layer 1 tool-parameter accuracy.
            helpfulness: Layer 2 helpfulness score.
            coherence: Layer 2 reasoning-coherence score.
            goal_success: Layer 3 goal-success rate.
            output_quality: Layer 3 output-quality score.

        Returns:
            ``layer_N_passed`` for each layer plus ``all_passed``, true only
            when all three pass.
        """
        layer_1 = self.validate_layer_1(tool_accuracy, param_accuracy)
        layer_2 = self.validate_layer_2(helpfulness, coherence)
        layer_3 = self.validate_layer_3(goal_success, output_quality)
        return {
            "layer_1_passed": layer_1,
            "layer_2_passed": layer_2,
            "layer_3_passed": layer_3,
            "all_passed": layer_1 and layer_2 and layer_3,
        }


# Default thresholds instance
EVALUATION_THRESHOLDS = EvaluationThresholds()


# Environment-specific threshold configurations
DEV_THRESHOLDS = EvaluationThresholds(
    tool_selection_accuracy=0.90,  # Lower for dev
    tool_parameter_accuracy=0.90,
    helpfulness_score=0.75,
    reasoning_coherence=0.80,
    goal_success_rate=0.85,
    output_quality_score=0.85,
    domain_aggregate_score=0.85,
)

STAGING_THRESHOLDS = EvaluationThresholds(
    tool_selection_accuracy=0.93,  # Between dev and prod
    tool_parameter_accuracy=0.93,
    helpfulness_score=0.80,
    reasoning_coherence=0.83,
    goal_success_rate=0.88,
    output_quality_score=0.88,
    domain_aggregate_score=0.88,
)

PRODUCTION_THRESHOLDS = EVALUATION_THRESHOLDS  # Use default (strict)
