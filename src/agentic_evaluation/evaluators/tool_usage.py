# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Layer 1 graders: did the agent call the right tools, rightly, in order?

All three compare the actual trajectory against the case's expectations without
consulting a model, so they are cheap enough to gate every run. They change with
your agent's tool schema.
"""

from typing import Any

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT

from agentic_evaluation.evaluators._trajectory import extract_tool_calls, extract_tool_names
from agentic_evaluation.evaluators._verdict import verdict


class ToolSelectionGrader(Evaluator[InputT, OutputT]):
    """Deterministic grader for tool selection correctness.

    Compares the actual trajectory (list of tool names) against expected tools.
    Layer 1 evaluation — no LLM calls.
    """

    def __init__(self, threshold: float = 0.95) -> None:
        """Set the pass threshold.

        Args:
            threshold: Fraction of expected tools that must be called to pass.
        """
        super().__init__()
        self.threshold = threshold

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score which of the expected tools the agent actually called.

        Args:
            evaluation_case: The case, whose ``expected_trajectory`` names the
                tools the agent should have used.

        Returns:
            The fraction of expected tools called, with the correct, missing and
            extra names in the reason. When no tools are expected, a pass only if
            the agent called none — that is the safety case.
        """
        expected = set(extract_tool_names(evaluation_case.expected_trajectory))
        actual = set(extract_tool_names(evaluation_case.actual_trajectory))

        if not expected:
            # Safety case: no tools expected, agent shouldn't call any
            if actual:
                return verdict(
                    0.0,
                    passed=False,
                    reason=f"Expected no tools, but called: {', '.join(sorted(actual))}",
                )
            return verdict(1.0, passed=True, reason="Correctly called no tools")

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

        return verdict(
            score,
            passed=score >= self.threshold,
            reason=" | ".join(parts) if parts else "No tools called",
        )


class ToolParameterGrader(Evaluator[InputT, OutputT]):
    """Deterministically compare expected argument subsets with actual calls."""

    def __init__(self, threshold: float = 0.95) -> None:
        """Set the pass threshold.

        Args:
            threshold: Fraction of expected parameters that must match to pass.
        """
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def _matches(expected: Any, actual: Any) -> bool:
        """Compare one expected argument value with the actual one.

        Args:
            expected: The declared value.
            actual: The value the agent passed.

        Returns:
            True if they match. Strings compare case-insensitively, because an
            agent choosing ``"Toyota"`` over ``"toyota"`` is not a defect.
        """
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.casefold() == actual.casefold()
        return expected == actual

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score the agent's tool arguments against the declared expectations.

        Reads ``expected_tool_parameters`` from case metadata: a mapping of tool
        name to the subset of arguments that call must carry. A parameter counts
        as matched if *any* call to that tool supplies it correctly.

        Args:
            evaluation_case: The case whose metadata declares the expectations.

        Returns:
            The fraction of expected parameters matched, listing every miss. A
            pass when nothing is declared; a zero-score failure when the
            declaration itself is malformed.
        """
        metadata = evaluation_case.metadata or {}
        expectations = metadata.get("expected_tool_parameters") or {}
        if not expectations:
            return verdict(1.0, passed=True, reason="No tool parameter expectations declared")
        if not isinstance(expectations, dict):
            return verdict(0.0, passed=False, reason="Tool parameter expectations are malformed")

        actual_calls = extract_tool_calls(evaluation_case.actual_trajectory)
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
        return verdict(
            score,
            passed=score >= self.threshold,
            reason=(
                f"Matched {matched}/{total} expected tool parameters"
                if not failures
                else "; ".join(failures)
            ),
        )


class TrajectoryOrderGrader(Evaluator[InputT, OutputT]):
    """Deterministic grader for trajectory ordering.

    Checks if tools were called in the expected order (subsequence match).
    Runs in Layer 1 alongside ToolSelectionGrader (deterministic, no LLM calls).
    """

    def __init__(self, threshold: float = 0.85) -> None:
        """Set the pass threshold.

        Args:
            threshold: Fraction of the expected order that must be matched to
                pass.
        """
        super().__init__()
        self.threshold = threshold

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Score how much of the expected tool order the agent followed.

        Matches as a subsequence, not a contiguous run: extra tools interleaved
        between the expected ones do not break the order.

        Args:
            evaluation_case: The case, whose ``expected_trajectory`` gives the
                order the tools should have been called in.

        Returns:
            The fraction of the expected sequence matched in order. A pass when
            no order is expected.
        """
        expected = extract_tool_names(evaluation_case.expected_trajectory)
        actual = extract_tool_names(evaluation_case.actual_trajectory)

        if not expected:
            return verdict(1.0, passed=True, reason="No trajectory expected")

        # In-order subsequence match
        exp_idx = 0
        for tool in actual:
            if exp_idx < len(expected) and tool == expected[exp_idx]:
                exp_idx += 1

        score = exp_idx / len(expected)
        return verdict(
            score,
            passed=score >= self.threshold,
            reason=(
                "All expected tools called in order"
                if score == 1.0
                else f"Matched {exp_idx}/{len(expected)} tools in order"
            ),
        )
