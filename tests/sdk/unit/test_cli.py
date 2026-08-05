# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the SDK CLI."""

import pytest

from agentic_evaluation.cli import _import_task_fn
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
def test_init_scaffold_disables_unmeasured_metrics_and_uses_environment_state(tmp_path):
    from agentic_evaluation.cli import main

    rc = main(["init", "--name", "demo", "--directory", str(tmp_path)])

    assert rc == 0
    config = (tmp_path / "eval_config.yaml").read_text()
    task_fn = (tmp_path / "task_fn.py").read_text()
    assert "latency: {enabled: false}" in config
    assert "cost: {enabled: false}" in config
    assert 'EnvironmentState(name="metrics", state=metrics)' in task_fn
