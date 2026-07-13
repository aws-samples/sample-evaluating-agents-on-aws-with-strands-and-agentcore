# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the offline-eval layer gate (run_experiment._layer_passed).

The gate must catch a single failing case in a deterministic layer (e.g. a
safety violation scoring 0.0) that a mean-only gate would average away, while
NOT vetoing an LLM-judge layer on a single borderline case below the library's
lenient 0.5 per-case bar.
"""

from strands_evals.types.evaluation_report import EvaluationReport

from agentic_evaluation.run_experiment import _DETERMINISTIC_LAYERS, _layer_passed


def _report(scores: list[float], test_passes: list[bool]) -> EvaluationReport:
    return EvaluationReport(
        evaluator_name="Fake",
        overall_score=sum(scores) / len(scores) if scores else 0.0,
        scores=scores,
        cases=[{} for _ in scores],
        test_passes=test_passes,
    )


class TestDeterministicGate:
    """strict_case_pass=True — Layer 1 and domain."""

    def test_single_safety_violation_fails_layer_despite_high_mean(self) -> None:
        # 9 perfect cases + 1 hard-zero safety violation -> mean 0.9 >= threshold,
        # but the violation's test_pass=False must fail the layer.
        scores = [1.0] * 9 + [0.0]
        passes = [True] * 9 + [False]
        report = _report(scores, passes)
        assert report.overall_score >= 0.90  # mean would have passed the old gate
        assert _layer_passed([report], 0.90, strict_case_pass=True) is False

    def test_all_cases_pass_and_mean_clears(self) -> None:
        report = _report([1.0, 0.96, 0.97], [True, True, True])
        assert _layer_passed([report], 0.95, strict_case_pass=True) is True

    def test_all_pass_but_mean_below_threshold_fails(self) -> None:
        # Every case passes its own bar but the layer's mean bar is higher.
        report = _report([0.6, 0.6, 0.6], [True, True, True])
        assert _layer_passed([report], 0.90, strict_case_pass=True) is False


class TestLlmJudgeGate:
    """strict_case_pass=False — Layer 2 and Layer 3."""

    def test_single_borderline_case_does_not_veto_when_mean_clears(self) -> None:
        # One case below the library's 0.5 bar (test_pass=False) but the mean
        # comfortably clears the layer threshold -> layer passes.
        scores = [0.95, 0.95, 0.48]
        passes = [True, True, False]
        report = _report(scores, passes)
        assert report.overall_score >= 0.79
        assert _layer_passed([report], 0.79, strict_case_pass=False) is True

    def test_low_mean_fails_even_if_all_cases_pass_lenient_bar(self) -> None:
        report = _report([0.6, 0.6, 0.6], [True, True, True])
        assert _layer_passed([report], 0.85, strict_case_pass=False) is False


class TestEdgeCases:
    def test_no_reports_fails(self) -> None:
        assert _layer_passed([], 0.90, strict_case_pass=True) is False

    def test_empty_report_fails(self) -> None:
        # An evaluator that produced zero cases must not silently gate green.
        assert _layer_passed([_report([], [])], 0.90, strict_case_pass=False) is False

    def test_layer_classification(self) -> None:
        assert "layer_1" in _DETERMINISTIC_LAYERS
        assert "domain" in _DETERMINISTIC_LAYERS
        assert "layer_2" not in _DETERMINISTIC_LAYERS
        assert "layer_3" not in _DETERMINISTIC_LAYERS
