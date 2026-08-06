# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Evaluators for what a run cost in time and money.

Both read the measured values a task_fn reports rather than timing anything
themselves, and both change with your SLOs and model pricing rather than with
the agent's logic. Deterministic — no LLM calls.
"""

from typing import TypeIs

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT

from agentic_evaluation.evaluators._trajectory import case_metrics
from agentic_evaluation.evaluators._verdict import verdict
from agentic_evaluation.thresholds import EVALUATION_THRESHOLDS


def _is_number(value: object) -> TypeIs[float]:
    """Report whether a metric is a usable number.

    The metrics channel is untyped (a task_fn fills it from a JSON payload), so
    this is the boundary check that turns an unknown into a number. Returning
    ``TypeIs`` rather than ``bool`` lets the arithmetic below the guard type-check
    instead of being read as "maybe None".

    Args:
        value: The value read from the metrics channel.

    Returns:
        True for ``int``/``float``, excluding ``bool`` — ``True`` is an ``int``
        in Python, and a boolean latency or token count means the task_fn
        reported the wrong thing, not a measurement of one.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class LatencyEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates response latency for production performance.

    Uses P50/P99 thresholds from EvaluationThresholds.
    Deterministic evaluator — no LLM calls.
    """

    def __init__(
        self,
        p50_threshold_ms: int = EVALUATION_THRESHOLDS.response_latency_p50_ms,
        p99_threshold_ms: int = EVALUATION_THRESHOLDS.response_latency_p99_ms,
    ) -> None:
        """Set the latency budget.

        Args:
            p50_threshold_ms: Latency at or below which the score is 1.0.
            p99_threshold_ms: Latency above which the case fails.
        """
        super().__init__()
        self.p50_threshold_ms = p50_threshold_ms
        self.p99_threshold_ms = p99_threshold_ms

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score the measured latency against the P50/P99 budget.

        Args:
            evaluation_case: The case, carrying ``latency_ms`` in its metrics
                channel.

        Returns:
            1.0 at or under P50, degrading linearly to 0.7 at P99, then
            decaying proportionally beyond it. A zero-score failure when no
            latency was measured.
        """
        latency_ms = case_metrics(evaluation_case).get("latency_ms")
        if not _is_number(latency_ms):
            return verdict(0.0, passed=False, reason="Latency measurement is unavailable")

        if latency_ms <= self.p50_threshold_ms:
            score = 1.0
        elif latency_ms <= self.p99_threshold_ms:
            over_p50 = latency_ms - self.p50_threshold_ms
            p50_to_p99 = self.p99_threshold_ms - self.p50_threshold_ms
            score = 1.0 - (over_p50 / p50_to_p99) * 0.3
        else:
            score = 0.7 * (self.p99_threshold_ms / latency_ms) if latency_ms > 0 else 0.0

        return verdict(
            score,
            passed=latency_ms <= self.p99_threshold_ms,
            reason=(
                f"Latency: {latency_ms:.0f}ms "
                f"(P50: {self.p50_threshold_ms}ms, P99: {self.p99_threshold_ms}ms)"
            ),
        )


class CostEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates cost per interaction for economic viability.

    Deterministic evaluator — no LLM calls.
    """

    def __init__(
        self,
        max_cost_per_query: float = 0.50,
        max_tokens_per_query: int = 10000,
    ) -> None:
        """Set the per-query cost and token ceilings.

        Args:
            max_cost_per_query: Highest acceptable estimated cost, in USD.
            max_tokens_per_query: Highest acceptable total token count.
        """
        super().__init__()
        self.max_cost_per_query = max_cost_per_query
        self.max_tokens_per_query = max_tokens_per_query

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score the measured token count and cost against their ceilings.

        Args:
            evaluation_case: The case, carrying ``total_tokens`` and
                ``estimated_cost_usd`` in its metrics channel.

        Returns:
            A pass when both are within budget; otherwise the ratio of budget to
            actual for whichever ceiling was exceeded. A zero-score failure when
            either value is missing.
        """
        metrics = case_metrics(evaluation_case)
        total_tokens = metrics.get("total_tokens")
        estimated_cost = metrics.get("estimated_cost_usd")
        if not _is_number(total_tokens) or not _is_number(estimated_cost):
            return verdict(0.0, passed=False, reason="Token or cost measurement is unavailable")

        token_passed = total_tokens <= self.max_tokens_per_query
        cost_passed = estimated_cost <= self.max_cost_per_query

        if token_passed and cost_passed:
            score = 1.0
        elif not token_passed and total_tokens > 0:
            score = min(self.max_tokens_per_query / total_tokens, 1.0)
        elif not cost_passed and estimated_cost > 0:
            score = min(self.max_cost_per_query / estimated_cost, 1.0)
        else:
            # Over a ceiling, but the measured value is zero or negative, so the
            # ratios above are meaningless. Nothing was spent; nothing to penalise.
            score = 1.0

        return verdict(
            score,
            passed=token_passed and cost_passed,
            reason=(
                f"Cost: ${estimated_cost:.4f} (max: ${self.max_cost_per_query}), "
                f"Tokens: {total_tokens} (max: {self.max_tokens_per_query})"
            ),
        )
