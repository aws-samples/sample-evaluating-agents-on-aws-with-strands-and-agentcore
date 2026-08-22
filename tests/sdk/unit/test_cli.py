# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the SDK CLI."""

from types import SimpleNamespace

import pytest
from strands_evals import Case, Experiment

from agentic_evaluation import reports_by_evaluator
from agentic_evaluation.cli import _build_output_payload, _import_task_fn
from agentic_evaluation.evaluators import LatencyEvaluator, ToolSelectionGrader
from agentic_evaluation.exceptions import ConfigError


@pytest.mark.sdk
def test_import_task_fn_rejects_missing_colon():
    with pytest.raises(ConfigError, match="must be of the form"):
        _import_task_fn("no_colon_here")


@pytest.mark.sdk
def test_import_task_fn_unknown_module():
    with pytest.raises(ConfigError, match="not found"):
        _import_task_fn("totally.fake.module:fn")


@pytest.mark.sdk
def test_import_task_fn_unknown_attr():
    with pytest.raises(ConfigError, match="has no attribute"):
        _import_task_fn("agentic_evaluation.cli:does_not_exist")


@pytest.mark.sdk
def test_import_task_fn_not_callable():
    # Use a non-callable attribute on the cli module
    with pytest.raises(ConfigError, match="not callable"):
        _import_task_fn("agentic_evaluation.cli:logger")


@pytest.mark.sdk
def test_validate_subcommand_on_quickstart_config(tmp_path, monkeypatch):
    from agentic_evaluation.cli import main

    cfg = tmp_path / "eval_config.yaml"
    cfg.write_text("""
project: {name: t, region: us-east-1}
judge_backend: noop
tools:
  search: {description: "x"}
test_cases:
  - {id: a, query: q, category: happy_path, expected_tools: [search],
     expected_behavior: "x", evaluation_layers: [layer_1_tool_usage], tags: []}
""")
    rc = main(["validate", "--config", str(cfg)])
    assert rc == 0


@pytest.mark.sdk
def test_full_json_output_names_the_evaluator_behind_each_score():
    """`--detail full` is only useful if each score says which evaluator gave it.

    The evaluator is a per-row tag on the report's cases, not an attribute of
    the report, so this runs a real two-evaluator Experiment and asserts on the
    names that come out the far end -- reading a report-level `evaluator_name`
    instead labelled every evaluator "unknown" and still produced valid JSON.
    """
    reports = reports_by_evaluator(
        Experiment(
            cases=[Case(name="case-a", input="query", expected_trajectory=[])],
            evaluators=[ToolSelectionGrader(threshold=0.95), LatencyEvaluator()],
        ).run_evaluations(lambda case: {"output": "ok", "trajectory": []})
    )
    results = {
        "all_passed": False,
        "layer_1": {"passed": False, "pass_rate": 0.0, "reports": reports},
    }

    payload = _build_output_payload(results, SimpleNamespace(project_name="demo"), detail="full")

    evaluators = payload["layers"]["layer_1"]["evaluators"]
    assert [entry["name"] for entry in evaluators] == [
        "ToolSelectionGrader",
        "LatencyEvaluator",
    ]
    # No tools were expected and none were called, so selection scores 1.0;
    # latency was never measured, so it scores 0.0 and fails its own gate.
    assert [entry["overall"] for entry in evaluators] == [1.0, 0.0]
    assert [case["id"] for entry in evaluators for case in entry["cases"]] == [
        "case-a",
        "case-a",
    ]
    assert [case["passed"] for entry in evaluators for case in entry["cases"]] == [
        True,
        False,
    ]


@pytest.mark.sdk
def test_init_scaffold_disables_unmeasured_metrics_and_uses_environment_state(tmp_path):
    from agentic_evaluation.cli import main

    rc = main(["init", "--name", "demo", "--directory", str(tmp_path)])

    assert rc == 0
    config = (tmp_path / "eval_config.yaml").read_text()
    task_fn = (tmp_path / "task_fn.py").read_text()
    assert "latency: {enabled: false}" in config
    assert "cost: {enabled: false}" in config
    assert 'EnvironmentState(name="metrics", state=metrics)' in task_fn
