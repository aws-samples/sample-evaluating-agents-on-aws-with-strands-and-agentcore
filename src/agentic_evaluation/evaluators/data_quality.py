# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Evaluators for the data an agent returns: is it fresh, and is it in scope?

Both are deterministic (no LLM calls) and both change for the same reason — the
shape and lifecycle of the data your agent queries.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT

from agentic_evaluation.evaluators._trajectory import case_metrics
from agentic_evaluation.evaluators._verdict import verdict


class DataFreshnessEvaluator(Evaluator[InputT, OutputT]):
    """Ensures agent is querying fresh data, not stale cached results.

    Generic evaluator for any agent that queries time-sensitive data.
    Default max_age_hours=24 suits daily refresh cycles; override for
    your domain (e.g., 1h for real-time feeds, 72h for weekly reports).

    Reads ``last_refresh_time`` (ISO 8601 string) via the same ``case_metrics()``
    channel as LatencyEvaluator/CostEvaluator: ``actual_environment_state``
    first (the only channel ``strands_evals`` propagates from a live task_fn),
    falling back to case ``metadata`` for static fixtures. Deterministic
    evaluator, no LLM calls.
    """

    def __init__(self, max_age_hours: int = 24) -> None:
        """Initialise the evaluator.

        Args:
            max_age_hours: Age above which the data counts as stale.
        """
        super().__init__()
        self.max_age_hours = max_age_hours

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score how recently the data behind the answer was refreshed.

        Args:
            evaluation_case: The case, carrying ``last_refresh_time`` in its
                metrics channel.

        Returns:
            A pass when the data is within ``max_age_hours``; a zero-score
            failure when it is stale, unmeasured, or unparseable.
        """
        last_refresh_iso = case_metrics(evaluation_case).get("last_refresh_time")
        if not last_refresh_iso:
            return verdict(0.0, passed=False, reason="Data freshness measurement is unavailable")

        # Normalize to aware UTC: ingestion writes tz-aware ISO; naive now() would raise TypeError.
        try:
            last_refresh = datetime.fromisoformat(str(last_refresh_iso))
        except ValueError:
            return verdict(0.0, passed=False, reason="Data freshness measurement is invalid")
        if last_refresh.tzinfo is None:
            last_refresh = last_refresh.replace(tzinfo=UTC)

        hours_since = (datetime.now(UTC) - last_refresh).total_seconds() / 3600
        if hours_since > self.max_age_hours:
            return verdict(
                0.0,
                passed=False,
                reason=(
                    f"Data is {hours_since:.1f}h old "
                    f"(threshold: {self.max_age_hours}h). Stale data."
                ),
            )
        return verdict(1.0, passed=True, reason=f"Data is fresh ({hours_since:.1f}h old)")


@dataclass(frozen=True, slots=True)
class SecondaryScope:
    """A single non-list output object whose scope must also match.

    Groups the three values that only mean anything together, so
    :class:`SchemaScopingEvaluator` cannot be built with a half-declared
    secondary check.

    Attributes:
        field: Key in ``actual_output`` holding the object to check.
        scope_field: Field on that object that must match the expected scope.
        metadata_key: Key in case metadata holding the expected scope value.
    """

    field: str
    scope_field: str
    metadata_key: str


class SchemaScopingEvaluator(Evaluator[InputT, OutputT]):
    """Generic data-scoping evaluator driven entirely by config.

    Verifies that every item in a list-valued output field carries the
    expected scope value sourced from the case metadata. Optionally
    verifies a single secondary object too (e.g. a profile/account).

    Use cases:
        - Multi-tenant data isolation (every record carries tenant_id).
        - Time-window scoping (every record belongs to current_period).
        - Per-user RBAC checks (returned profile matches caller).

    Deterministic evaluator — no LLM calls.
    """

    def __init__(
        self,
        list_field: str,
        scope_field: str,
        metadata_key: str,
        *,
        secondary: SecondaryScope | None = None,
        max_violations_in_reason: int = 5,
    ) -> None:
        """Declare the scope every returned record must carry.

        Args:
            list_field: Key in ``actual_output`` holding a list of items to check.
            scope_field: Field on each item that must match the expected scope.
            metadata_key: Key in case metadata holding the expected scope value.
            secondary: Optional extra check for one non-list output object.
            max_violations_in_reason: Cap on violations listed in the reason text.
        """
        super().__init__()
        self.list_field = list_field
        self.scope_field = scope_field
        self.metadata_key = metadata_key
        self.secondary = secondary
        self.max_violations_in_reason = max_violations_in_reason

    def _check_items(self, items: list[Any], expected_scope: Any) -> tuple[int, list[str]]:
        """Check every dict in the list field against the expected scope.

        Args:
            items: The list held by ``list_field``. Non-dict entries are skipped
                rather than failed: a list of scalars carries no scope to check.
            expected_scope: The value each item's ``scope_field`` must equal.

        Returns:
            The number of items checked and one message per mismatch.
        """
        violations: list[str] = []
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
        return checked, violations

    def _check_secondary(self, output: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
        """Check the optional secondary object against its own expected scope.

        Args:
            output: The case's ``actual_output``.
            metadata: The case metadata holding the expected scope value.

        Returns:
            One message if the secondary object is out of scope, else empty.
            Also empty when no secondary check is declared, the metadata does
            not supply an expected value, or the object is absent.
        """
        if self.secondary is None:
            return []
        expected = metadata.get(self.secondary.metadata_key)
        obj = output.get(self.secondary.field)
        if not expected or not isinstance(obj, dict):
            return []
        actual = obj.get(self.secondary.scope_field)
        if actual == expected:
            return []
        return [
            f"{self.secondary.field}.{self.secondary.scope_field}="
            f"{actual!r} (expected {expected!r})"
        ]

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Verify every returned record is scoped to the case's expected value.

        Args:
            evaluation_case: The case, whose metadata supplies the expected
                scope and whose ``actual_output`` supplies the records.

        Returns:
            A pass when every item matches (or when the check does not apply,
            because the output is not a dict or the metadata declares no
            expected scope); a zero-score failure listing the violations
            otherwise, capped at ``max_violations_in_reason``.
        """
        metadata = evaluation_case.metadata or {}
        expected_scope = metadata.get(self.metadata_key)

        output = evaluation_case.actual_output
        if not isinstance(output, dict):
            return verdict(1.0, passed=True, reason="Non-dict output, no scoping check applicable")
        if expected_scope in (None, ""):
            return verdict(
                1.0,
                passed=True,
                reason=f"metadata['{self.metadata_key}'] not provided, scoping skipped",
            )
        items = output.get(self.list_field, [])
        if not isinstance(items, list):
            return verdict(
                0.0,
                passed=False,
                reason=(f"Field '{self.list_field}' is not a list (got {type(items).__name__})"),
            )

        checked, violations = self._check_items(items, expected_scope)
        violations.extend(self._check_secondary(output, metadata))

        if violations:
            shown = violations[: self.max_violations_in_reason]
            extra = len(violations) - len(shown)
            tail = f" ... and {extra} more" if extra > 0 else ""
            return verdict(
                0.0, passed=False, reason=f"Scoping violations: {'; '.join(shown)}{tail}"
            )
        return verdict(
            1.0,
            passed=True,
            reason=(
                f"All {checked} items correctly scoped to {self.metadata_key}={expected_scope!r}"
            ),
        )
