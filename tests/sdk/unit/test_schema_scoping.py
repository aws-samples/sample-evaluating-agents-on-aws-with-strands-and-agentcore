# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the generic SchemaScopingEvaluator."""

import pytest
from strands_evals.types.evaluation import EvaluationData

from agentic_evaluation import SchemaScopingEvaluator


def _case(output, metadata):
    return EvaluationData(
        input="q",
        expected_output="",
        actual_output=output,
        expected_trajectory=[],
        actual_trajectory=[],
        metadata=metadata,
    )


@pytest.mark.sdk
def test_passes_when_every_item_in_scope():
    ev = SchemaScopingEvaluator(
        list_field="items", scope_field="tenant_id", metadata_key="tenant_id"
    )
    result = ev.evaluate(
        _case(
            {"items": [{"tenant_id": "t1"}, {"tenant_id": "t1"}]},
            {"tenant_id": "t1"},
        )
    )[0]
    assert result.test_pass
    assert "2 items" in result.reason


@pytest.mark.sdk
def test_fails_on_cross_tenant_leak():
    ev = SchemaScopingEvaluator(
        list_field="items", scope_field="tenant_id", metadata_key="tenant_id"
    )
    result = ev.evaluate(
        _case(
            {"items": [{"tenant_id": "t1"}, {"tenant_id": "t2", "id": "x"}]},
            {"tenant_id": "t1"},
        )
    )[0]
    assert not result.test_pass
    assert "t2" in result.reason


@pytest.mark.sdk
def test_caps_violation_count_in_reason():
    ev = SchemaScopingEvaluator(
        list_field="items",
        scope_field="t",
        metadata_key="t",
        max_violations_in_reason=2,
    )
    items = [{"t": "wrong", "id": i} for i in range(7)]
    result = ev.evaluate(_case({"items": items}, {"t": "right"}))[0]
    assert "and 5 more" in result.reason


@pytest.mark.sdk
def test_secondary_object_scoping():
    ev = SchemaScopingEvaluator(
        list_field="items",
        scope_field="t",
        metadata_key="t",
        secondary_field="profile",
        secondary_scope="user_id",
        secondary_metadata_key="user_id",
    )
    result = ev.evaluate(
        _case(
            {"items": [], "profile": {"user_id": "u2"}},
            {"t": "ignored", "user_id": "u1"},
        )
    )[0]
    assert not result.test_pass
    assert "profile.user_id" in result.reason


@pytest.mark.sdk
def test_skips_when_metadata_missing():
    ev = SchemaScopingEvaluator(list_field="items", scope_field="t", metadata_key="t")
    result = ev.evaluate(_case({"items": [{"t": "x"}]}, {}))[0]
    assert result.test_pass
    assert "not provided" in result.reason


@pytest.mark.sdk
def test_non_dict_output_passes_safely():
    ev = SchemaScopingEvaluator(list_field="items", scope_field="t", metadata_key="t")
    result = ev.evaluate(_case("plain string output", {"t": "x"}))[0]
    assert result.test_pass
