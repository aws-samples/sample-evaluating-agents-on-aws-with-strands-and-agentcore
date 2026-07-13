# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for JudgeBackend factory + NoOp behavior."""

import pytest

from agentic_evaluation import NoOpJudgeBackend, build_judge
from agentic_evaluation.exceptions import PluginNotFoundError


@pytest.mark.sdk
def test_build_noop_returns_noop():
    judge = build_judge("noop")
    assert isinstance(judge, NoOpJudgeBackend)


@pytest.mark.sdk
def test_build_unknown_judge_raises_plugin_error():
    with pytest.raises(PluginNotFoundError):
        build_judge("definitely-not-installed")


@pytest.mark.sdk
def test_noop_returns_passing_layer2():
    judge = NoOpJudgeBackend()
    evaluators = judge.layer2_evaluators(model="x", rubric="y", tool_descriptions={})
    assert len(evaluators) == 1
    ev = evaluators[0]
    out = ev.evaluate(None)[0]
    assert out.test_pass and "noop" in out.reason
    # Must be runnable through the strands_evals async path.
    assert ev.get_type_name() == "NoOpJudge_layer_2"


@pytest.mark.sdk
def test_noop_returns_passing_layer3():
    judge = NoOpJudgeBackend()
    evaluators = judge.layer3_evaluators(model="x", rubric="y")
    assert len(evaluators) == 1
    ev = evaluators[0]
    out = ev.evaluate(None)[0]
    assert out.test_pass
    assert ev.get_type_name() == "NoOpJudge_layer_3"


@pytest.mark.sdk
def test_build_strands_succeeds_when_extra_installed():
    # strands_evals is in the base deps for this repo, so this should always work
    from agentic_evaluation import StrandsJudgeBackend

    judge = build_judge("strands")
    assert isinstance(judge, StrandsJudgeBackend)
