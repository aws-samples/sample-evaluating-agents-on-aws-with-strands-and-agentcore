# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Strands-agents-evals compatible evaluators.

Wraps domain-specific custom evaluators as strands_evals.Evaluator subclasses
so they integrate with the Experiment/Case framework while keeping
deterministic, code-based evaluation logic (no LLM calls).

Also re-exports library LLM-based evaluators pre-configured for our thresholds.
"""

from datetime import datetime, timezone
from typing import Any

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from strands_evals.types.trace import Session, ToolExecutionSpan

from agentic_evaluation.thresholds import EVALUATION_THRESHOLDS


def _extract_tool_names(trajectory: Any) -> list[str]:
    """Extract tool names from either a list[str] or a Session object.

    Deterministic evaluators store trajectories as list[str] for simplicity.
    LLM-based evaluators (HelpfulnessEvaluator, GoalSuccessRateEvaluator)
    require a Session object. This helper lets our custom evaluators work
    with both formats so run_all_layers() can use a single task function.
    """
    if trajectory is None:
        return []
    if isinstance(trajectory, list):
        return [t for t in trajectory if isinstance(t, str)]
    if isinstance(trajectory, Session):
        return [
            span.tool_call.name
            for trace in trajectory.traces
            for span in trace.spans
            if isinstance(span, ToolExecutionSpan)
        ]
    return []


def _extract_tool_calls(trajectory: Any) -> list[tuple[str, dict[str, Any]]]:
    """Extract ordered tool names and arguments from supported trajectories."""
    if isinstance(trajectory, list):
        calls = []
        for item in trajectory:
            if isinstance(item, dict):
                name = str(item.get("name", ""))
                arguments = item.get("arguments", {})
                calls.append((name, arguments if isinstance(arguments, dict) else {}))
        return calls
    if isinstance(trajectory, Session):
        if not trajectory.traces:
            return []
        # Multi-turn sessions accumulate traces; parameter expectations belong
        # to the current turn, which is always the final trace.
        return [
            (span.tool_call.name, span.tool_call.arguments or {})
            for span in trajectory.traces[-1].spans
            if isinstance(span, ToolExecutionSpan)
        ]
    return []


def _metrics(evaluation_case: EvaluationData[Any, Any]) -> dict[str, Any]:
    """Read per-turn operational metrics for the case.

    ``strands_evals`` drops a task's result ``metadata`` but propagates
    ``environment_state``, so live adapters (e.g. Bedrock AgentCore) surface
    latency/token/cost there under an ``EnvironmentState(name="metrics")``.
    Prefer that; fall back to ``metadata`` for cases constructed directly with
    ``metadata=`` (tests, static fixtures).
    """
    for env in evaluation_case.actual_environment_state or []:
        if getattr(env, "name", None) == "metrics" and isinstance(env.state, dict):
            return env.state
    return evaluation_case.metadata or {}


class DataFreshnessEvaluator(Evaluator[InputT, OutputT]):
    """Ensures agent is querying fresh data, not stale cached results.

    Generic evaluator for any agent that queries time-sensitive data.
    Default max_age_hours=24 suits daily refresh cycles; override for
    your domain (e.g., 1h for real-time feeds, 72h for weekly reports).

    Reads ``last_refresh_time`` (ISO 8601 string) via the same ``_metrics()``
    channel as LatencyEvaluator/CostEvaluator: ``actual_environment_state``
    first (the only channel ``strands_evals`` propagates from a live task_fn),
    falling back to case ``metadata`` for static fixtures. Deterministic
    evaluator, no LLM calls.
    """

    def __init__(self, max_age_hours: int = 24) -> None:
        super().__init__()
        self.max_age_hours = max_age_hours

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        last_refresh_iso = _metrics(evaluation_case).get("last_refresh_time")

        # Normalize to aware UTC: ingestion writes tz-aware ISO; naive now() would raise TypeError.
        now = datetime.now(timezone.utc)
        if last_refresh_iso:
            try:
                last_refresh = datetime.fromisoformat(str(last_refresh_iso))
            except ValueError:
                return [
                    EvaluationOutput(
                        score=0.0,
                        test_pass=False,
                        reason="Data freshness measurement is invalid",
                    )
                ]
            if last_refresh.tzinfo is None:
                last_refresh = last_refresh.replace(tzinfo=timezone.utc)
        else:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason="Data freshness measurement is unavailable",
                )
            ]

        hours_since = (now - last_refresh).total_seconds() / 3600

        if hours_since > self.max_age_hours:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=(
                        f"Data is {hours_since:.1f}h old "
                        f"(threshold: {self.max_age_hours}h). Stale data."
                    ),
                )
            ]

        return [
            EvaluationOutput(
                score=1.0,
                test_pass=True,
                reason=f"Data is fresh ({hours_since:.1f}h old)",
            )
        ]


class SchemaScopingEvaluator(Evaluator[InputT, OutputT]):
    """Generic data-scoping evaluator driven entirely by config.

    Verifies that every item in a list-valued output field carries the
    expected scope value sourced from the case metadata. Optionally
    verifies a single secondary object too (e.g. a profile/account).

    Use cases:
        - Multi-tenant data isolation (every record carries tenant_id).
        - Time-window scoping (every record belongs to current_period).
        - Per-user RBAC checks (returned profile matches caller).

    Args:
        list_field: Key in actual_output holding a list of items to check.
        scope_field: Field on each item that must match the expected scope.
        metadata_key: Key in case metadata holding the expected scope value.
        secondary_field: Optional key in actual_output holding a single object.
        secondary_scope: Field on that object that must match expected scope.
        secondary_metadata_key: Metadata key for the secondary scope value.
        max_violations_in_reason: Cap on violations listed in the reason text.

    Deterministic evaluator — no LLM calls.
    """

    def __init__(
        self,
        list_field: str,
        scope_field: str,
        metadata_key: str,
        *,
        secondary_field: str | None = None,
        secondary_scope: str | None = None,
        secondary_metadata_key: str | None = None,
        max_violations_in_reason: int = 5,
    ) -> None:
        super().__init__()
        self.list_field = list_field
        self.scope_field = scope_field
        self.metadata_key = metadata_key
        self.secondary_field = secondary_field
        self.secondary_scope = secondary_scope
        self.secondary_metadata_key = secondary_metadata_key
        self.max_violations_in_reason = max_violations_in_reason

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        metadata = evaluation_case.metadata or {}
        expected_scope = metadata.get(self.metadata_key)

        output = evaluation_case.actual_output
        if not isinstance(output, dict):
            return [
                EvaluationOutput(
                    score=1.0,
                    test_pass=True,
                    reason="Non-dict output, no scoping check applicable",
                )
            ]

        if expected_scope in (None, ""):
            return [
                EvaluationOutput(
                    score=1.0,
                    test_pass=True,
                    reason=f"metadata['{self.metadata_key}'] not provided, scoping skipped",
                )
            ]

        violations: list[str] = []
        items = output.get(self.list_field, [])
        if not isinstance(items, list):
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"Field '{self.list_field}' is not a list (got {type(items).__name__})",
                )
            ]

        checked = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            checked += 1
            actual_scope = item.get(self.scope_field)
            if actual_scope != expected_scope:
                violations.append(
                    f"{self.list_field}[{checked - 1}].{self.scope_field}={actual_scope!r} "
                    f"(expected {expected_scope!r})"
                )

        if self.secondary_field and self.secondary_scope and self.secondary_metadata_key:
            secondary_expected = metadata.get(self.secondary_metadata_key)
            secondary_obj = output.get(self.secondary_field)
            if (
                secondary_expected
                and isinstance(secondary_obj, dict)
                and secondary_obj.get(self.secondary_scope) != secondary_expected
            ):
                violations.append(
                    f"{self.secondary_field}.{self.secondary_scope}="
                    f"{secondary_obj.get(self.secondary_scope)!r} "
                    f"(expected {secondary_expected!r})"
                )

        if violations:
            shown = violations[: self.max_violations_in_reason]
            extra = len(violations) - len(shown)
            tail = f" ... and {extra} more" if extra > 0 else ""
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"Scoping violations: {'; '.join(shown)}{tail}",
                )
            ]

        return [
            EvaluationOutput(
                score=1.0,
                test_pass=True,
                reason=f"All {checked} items correctly scoped to {self.metadata_key}={expected_scope!r}",
            )
        ]


class SafetyGuardrailEvaluator(Evaluator[InputT, OutputT]):
    """Ensures agent adheres to safety guardrails.

    Generic evaluator for any agent. Checks that the agent never attempts
    forbidden tool calls and never outputs forbidden phrases.

    Configure via eval_config.yaml ``safety`` section or pass sets to
    the constructor. Defaults are empty — you must explicitly declare
    your forbidden actions/phrases.

    .. versionchanged:: 0.3.0
        Default ``forbidden_actions`` and ``forbidden_phrases`` are now
        empty. Earlier versions shipped car-auction bidding defaults
        which silently applied to other domains. Migrate by setting them
        in eval_config.yaml under the ``safety`` section.

    Deterministic evaluator — no LLM calls.
    """

    DEFAULT_FORBIDDEN_ACTIONS: frozenset[str] = frozenset()
    DEFAULT_FORBIDDEN_PHRASES: tuple[str, ...] = ()

    def __init__(
        self,
        forbidden_actions: set[str] | frozenset[str] | None = None,
        forbidden_phrases: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.forbidden_actions = (
            self.DEFAULT_FORBIDDEN_ACTIONS if forbidden_actions is None else forbidden_actions
        )
        self.forbidden_phrases = (
            list(self.DEFAULT_FORBIDDEN_PHRASES)
            if forbidden_phrases is None
            else list(forbidden_phrases)
        )

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        violations: list[str] = []

        # Check trajectory for forbidden tool calls
        tool_names = _extract_tool_names(evaluation_case.actual_trajectory)
        violations.extend(
            f"Attempted forbidden action: {tool_name}"
            for tool_name in tool_names
            if tool_name.lower() in self.forbidden_actions
        )

        # Check output text for forbidden phrases
        output = evaluation_case.actual_output
        output_text = ""
        if isinstance(output, str):
            output_text = output.lower()
        elif isinstance(output, dict):
            output_text = str(output.get("text", output.get("result", ""))).lower()

        violations.extend(
            f"Forbidden phrase: '{phrase}'"
            for phrase in self.forbidden_phrases
            if phrase.lower() in output_text
        )

        if violations:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"Safety violations: {'; '.join(violations)}",
                )
            ]

        return [
            EvaluationOutput(
                score=1.0,
                test_pass=True,
                reason="All safety guardrails respected",
            )
        ]


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
        super().__init__()
        self.p50_threshold_ms = p50_threshold_ms
        self.p99_threshold_ms = p99_threshold_ms

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        latency_ms = _metrics(evaluation_case).get("latency_ms")
        if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool):
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason="Latency measurement is unavailable",
                )
            ]

        if latency_ms <= self.p50_threshold_ms:
            score = 1.0
        elif latency_ms <= self.p99_threshold_ms:
            score = (
                1.0
                - (
                    (latency_ms - self.p50_threshold_ms)
                    / (self.p99_threshold_ms - self.p50_threshold_ms)
                )
                * 0.3
            )
        else:
            score = 0.7 * (self.p99_threshold_ms / latency_ms) if latency_ms > 0 else 0.0

        passed = latency_ms <= self.p99_threshold_ms

        return [
            EvaluationOutput(
                score=score,
                test_pass=passed,
                reason=(
                    f"Latency: {latency_ms:.0f}ms "
                    f"(P50: {self.p50_threshold_ms}ms, P99: {self.p99_threshold_ms}ms)"
                ),
            )
        ]


class CostEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates cost per interaction for economic viability.

    Deterministic evaluator — no LLM calls.
    """

    def __init__(
        self,
        max_cost_per_query: float = 0.50,
        max_tokens_per_query: int = 10000,
    ) -> None:
        super().__init__()
        self.max_cost_per_query = max_cost_per_query
        self.max_tokens_per_query = max_tokens_per_query

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        metrics = _metrics(evaluation_case)
        total_tokens = metrics.get("total_tokens")
        estimated_cost = metrics.get("estimated_cost_usd")
        if (
            not isinstance(total_tokens, (int, float))
            or isinstance(total_tokens, bool)
            or not isinstance(estimated_cost, (int, float))
            or isinstance(estimated_cost, bool)
        ):
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason="Token or cost measurement is unavailable",
                )
            ]

        token_passed = total_tokens <= self.max_tokens_per_query
        cost_passed = estimated_cost <= self.max_cost_per_query

        if token_passed and cost_passed:
            score = 1.0
        elif not token_passed and total_tokens > 0:
            score = min(self.max_tokens_per_query / total_tokens, 1.0)
        elif not cost_passed and estimated_cost > 0:
            score = min(self.max_cost_per_query / estimated_cost, 1.0)
        else:
            score = 1.0

        return [
            EvaluationOutput(
                score=score,
                test_pass=token_passed and cost_passed,
                reason=(
                    f"Cost: ${estimated_cost:.4f} (max: ${self.max_cost_per_query}), "
                    f"Tokens: {total_tokens} (max: {self.max_tokens_per_query})"
                ),
            )
        ]


class ToolSelectionGrader(Evaluator[InputT, OutputT]):
    """Deterministic grader for tool selection correctness.

    Compares the actual trajectory (list of tool names) against expected tools.
    Layer 1 evaluation — no LLM calls.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        super().__init__()
        self.threshold = threshold

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        expected = set(evaluation_case.expected_trajectory or [])
        actual = set(_extract_tool_names(evaluation_case.actual_trajectory))

        if not expected:
            # Safety case: no tools expected, agent shouldn't call any
            if actual:
                return [
                    EvaluationOutput(
                        score=0.0,
                        test_pass=False,
                        reason=f"Expected no tools, but called: {', '.join(sorted(actual))}",
                    )
                ]
            return [
                EvaluationOutput(
                    score=1.0,
                    test_pass=True,
                    reason="Correctly called no tools",
                )
            ]

        correct = actual & expected
        missing = expected - actual
        extra = actual - expected
        score = len(correct) / len(expected)

        parts = []
        if correct:
            parts.append(f"Correct: {', '.join(sorted(correct))}")
        if missing:
            parts.append(f"Missing: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"Extra: {', '.join(sorted(extra))}")

        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= self.threshold,
                reason=" | ".join(parts) if parts else "No tools called",
            )
        ]


class ToolParameterGrader(Evaluator[InputT, OutputT]):
    """Deterministically compare expected argument subsets with actual calls."""

    def __init__(self, threshold: float = 0.95) -> None:
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def _matches(expected: Any, actual: Any) -> bool:
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.casefold() == actual.casefold()
        return expected == actual

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        metadata = evaluation_case.metadata or {}
        expectations = metadata.get("expected_tool_parameters") or {}
        if not expectations:
            return [
                EvaluationOutput(
                    score=1.0,
                    test_pass=True,
                    reason="No tool parameter expectations declared",
                )
            ]
        if not isinstance(expectations, dict):
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason="Tool parameter expectations are malformed",
                )
            ]

        actual_calls = _extract_tool_calls(evaluation_case.actual_trajectory)
        matched = 0
        total = 0
        failures: list[str] = []
        for tool_name, expected_arguments in expectations.items():
            if not isinstance(expected_arguments, dict):
                failures.append(f"{tool_name}: expected parameters are not a mapping")
                total += 1
                continue
            candidates = [arguments for name, arguments in actual_calls if name == tool_name]
            for parameter, expected_value in expected_arguments.items():
                total += 1
                if any(
                    parameter in arguments
                    and self._matches(expected_value, arguments.get(parameter))
                    for arguments in candidates
                ):
                    matched += 1
                else:
                    failures.append(f"{tool_name}.{parameter} expected {expected_value!r}")

        score = matched / total if total else 1.0
        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= self.threshold,
                reason=(
                    f"Matched {matched}/{total} expected tool parameters"
                    if not failures
                    else "; ".join(failures)
                ),
            )
        ]


class TrajectoryOrderGrader(Evaluator[InputT, OutputT]):
    """Deterministic grader for trajectory ordering.

    Checks if tools were called in the expected order (subsequence match).
    Runs in Layer 1 alongside ToolSelectionGrader (deterministic, no LLM calls).
    """

    def __init__(self, threshold: float = 0.85) -> None:
        super().__init__()
        self.threshold = threshold

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        expected = evaluation_case.expected_trajectory or []
        actual = _extract_tool_names(evaluation_case.actual_trajectory)

        if not expected:
            return [EvaluationOutput(score=1.0, test_pass=True, reason="No trajectory expected")]

        # In-order subsequence match
        exp_idx = 0
        for tool in actual:
            if exp_idx < len(expected) and tool == expected[exp_idx]:
                exp_idx += 1

        score = exp_idx / len(expected)

        if score == 1.0:
            reason = "All expected tools called in order"
        else:
            reason = f"Matched {exp_idx}/{len(expected)} tools in order"

        return [
            EvaluationOutput(
                score=score,
                test_pass=score >= self.threshold,
                reason=reason,
            )
        ]
