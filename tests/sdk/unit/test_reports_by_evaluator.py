# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for run_experiment.reports_by_evaluator and evaluator_name.

``Experiment.run_evaluations()`` returns one report whose rows are the cross
product of evaluators and cases. The layer gate needs one report per evaluator,
because it applies a threshold per evaluator and treats each evaluator's
per-case ``test_pass`` as authoritative. These tests pin the regrouping that
bridges the two, above all that each report's ``overall_score`` is its own
evaluator's mean rather than the run-wide mean the flat report carries -- plus
the reader that names the evaluator behind a regrouped report.
"""

import pytest
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationOutput
from strands_evals.types.evaluation_report import EvaluationReport

from agentic_evaluation import evaluator_name, reports_by_evaluator


def _flat_report(rows: list[tuple[str, float, bool]]) -> EvaluationReport:
    """Build a flat report from ``(evaluator, score, test_pass)`` rows."""
    return EvaluationReport(
        overall_score=sum(score for _, score, _ in rows) / len(rows) if rows else 0.0,
        scores=[score for _, score, _ in rows],
        cases=[
            {"evaluator": name, "name": f"case-{index}"} for index, (name, _, _) in enumerate(rows)
        ],
        test_passes=[passed for _, _, passed in rows],
        reasons=[f"{name} says {score}" for name, score, _ in rows],
        detailed_results=[
            [EvaluationOutput(score=score, test_pass=passed)] for _, score, passed in rows
        ],
        diagnoses=[None for _ in rows],
        recommendations=[None for _ in rows],
    )


def test_groups_rows_by_evaluator_in_first_appearance_order():
    flat = _flat_report(
        [
            ("Latency", 1.0, True),
            ("Latency", 0.8, True),
            ("Cost", 0.4, False),
            ("Cost", 0.6, True),
        ]
    )

    grouped = reports_by_evaluator(flat)

    assert [report.cases[0]["evaluator"] for report in grouped] == ["Latency", "Cost"]
    assert [report.scores for report in grouped] == [[1.0, 0.8], [0.4, 0.6]]
    assert [report.test_passes for report in grouped] == [[True, True], [False, True]]
    # Every parallel row array is sliced with the same indices, so a report's
    # reasons still line up with its own scores.
    assert grouped[1].reasons == ["Cost says 0.4", "Cost says 0.6"]
    assert [case["name"] for case in grouped[1].cases] == ["case-2", "case-3"]


def test_overall_score_is_per_evaluator_not_the_run_wide_mean():
    # A perfect evaluator and a failing one. The flat report averages them to
    # 0.5; regrouped, each keeps its own mean so a per-evaluator threshold can
    # fail the offender instead of averaging it away.
    flat = _flat_report(
        [
            ("Safety", 1.0, True),
            ("Safety", 1.0, True),
            ("Cost", 0.0, False),
            ("Cost", 0.0, False),
        ]
    )
    assert flat.overall_score == 0.5

    grouped = reports_by_evaluator(flat)

    assert grouped[0].overall_score == 1.0
    assert grouped[1].overall_score == 0.0


def test_rows_an_evaluator_declined_to_judge_stay_out_of_its_mean():
    # A not-applicable row that passed carries no verdict. The library drops it
    # from its own averages; regrouping must inherit that rather than scoring
    # the row as a real 0.0, which would report the run as worse than it was.
    flat = EvaluationReport(
        overall_score=0.5,
        scores=[1.0, 0.0],
        cases=[{"evaluator": "Freshness"}, {"evaluator": "Freshness"}],
        test_passes=[True, True],
        detailed_results=[
            [EvaluationOutput(score=1.0, test_pass=True)],
            [EvaluationOutput(score=0.0, test_pass=True, label=NOT_APPLICABLE)],
        ],
    )

    grouped = reports_by_evaluator(flat)

    assert len(grouped) == 1
    assert grouped[0].overall_score == 1.0
    # The row is retained in the report details even though it left the mean.
    assert grouped[0].scores == [1.0, 0.0]


def test_report_with_no_rows_yields_no_reports():
    # _layer_passed treats an empty list as a non-pass, which is how an empty
    # layer is kept from gating green.
    assert reports_by_evaluator(_flat_report([])) == []


def test_evaluator_name_reads_the_per_row_tag():
    # The name has no report-level home: it is a tag on each row, and every row
    # of a regrouped report carries the same one, since that is what it grouped by.
    grouped = reports_by_evaluator(_flat_report([("Latency", 1.0, True), ("Cost", 0.4, False)]))

    assert [evaluator_name(report) for report in grouped] == ["Latency", "Cost"]


def test_evaluator_name_will_not_invent_a_name_for_a_report_with_no_rows():
    # Better a loud IndexError than a plausible "unknown" that lands in a
    # results file as though an evaluator of that name had run and scored 0.
    with pytest.raises(IndexError):
        evaluator_name(_flat_report([]))


def test_report_carrying_only_required_rows_does_not_raise():
    # reasons/detailed_results/diagnoses/recommendations default to [] on the
    # model, so a hand-built report has fewer of those rows than scores.
    flat = EvaluationReport(
        overall_score=0.9,
        scores=[0.9],
        cases=[{"evaluator": "Latency"}],
        test_passes=[True],
    )

    grouped = reports_by_evaluator(flat)

    assert len(grouped) == 1
    assert grouped[0].scores == [0.9]
    assert grouped[0].reasons == []
    assert grouped[0].overall_score == 0.9
