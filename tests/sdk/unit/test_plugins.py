# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for plugin loader."""

import pytest

from agentic_evaluation.exceptions import PluginNotFoundError
from agentic_evaluation.plugins import build_evaluators_from_config, load_evaluator_plugin


@pytest.mark.sdk
def test_unknown_plugin_raises():
    with pytest.raises(PluginNotFoundError) as info:
        load_evaluator_plugin("nonexistent_plugin")
    assert "nonexistent_plugin" in str(info.value)


@pytest.mark.sdk
def test_build_from_empty_config_returns_empty_list():
    assert build_evaluators_from_config([]) == []


@pytest.mark.sdk
def test_build_rejects_spec_without_name():
    from agentic_evaluation.exceptions import PluginLoadError

    with pytest.raises(PluginLoadError):
        build_evaluators_from_config([{"params": {}}])
