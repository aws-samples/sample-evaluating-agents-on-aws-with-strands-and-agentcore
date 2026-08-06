# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The single-verdict return shape every deterministic evaluator produces.

``Evaluator.evaluate`` returns a *list* of outputs so one evaluator can report
several independent judgements. Every evaluator in this package reports exactly
one, so each would otherwise end in the same five-line
``[EvaluationOutput(score=..., test_pass=..., reason=...)]`` literal — around
twenty-five copies. :func:`verdict` states that shape once and lets each
evaluator's return read as the judgement it is.

Private module: an implementation detail of
:mod:`agentic_evaluation.evaluators`, not part of the SDK's public surface.
"""

from strands_evals.types.evaluation import EvaluationOutput


def verdict(score: float, *, passed: bool, reason: str) -> list[EvaluationOutput]:
    """Build the single-output list an evaluator returns.

    Args:
        score: Score in ``[0.0, 1.0]``.
        passed: Whether the case satisfies the evaluator's threshold.
        reason: Operator-facing explanation, shown verbatim in reports.

    Returns:
        A one-element list holding the corresponding ``EvaluationOutput``.
    """
    return [EvaluationOutput(score=score, test_pass=passed, reason=reason)]
