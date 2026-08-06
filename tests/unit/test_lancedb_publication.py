# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Failure-path tests for immutable LanceDB snapshot publication."""

import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def ingestion_handler():
    handler_path = (
        Path(__file__).resolve().parents[2]
        / "examples/vehicle-auction-agent/lambda/functions/data_ingestion/handler.py"
    )
    spec = importlib.util.spec_from_file_location("test_data_ingestion_handler", handler_path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(os.environ, {"DATA_BUCKET": "test-bucket"}),
        patch("boto3.client", return_value=MagicMock()),
    ):
        spec.loader.exec_module(module)
    return module


def _vehicle(vehicle_id: str, embedding: list[float]) -> dict:
    return {
        "id": vehicle_id,
        "make": "BMW",
        "model": "X3",
        "embedding": embedding,
    }


def test_snapshot_validation_rejects_empty_duplicate_and_bad_vectors(ingestion_handler) -> None:
    with pytest.raises(ValueError, match="empty"):
        ingestion_handler.validate_lancedb_snapshot([], expected_dimension=4)

    with pytest.raises(ValueError, match="unique"):
        ingestion_handler.validate_lancedb_snapshot(
            [_vehicle("V1", [1.0] * 4), _vehicle("V1", [0.0] * 4)],
            expected_dimension=4,
        )

    with pytest.raises(ValueError, match="expected"):
        ingestion_handler.validate_lancedb_snapshot(
            [_vehicle("V1", [1.0] * 3)],
            expected_dimension=4,
        )

    with pytest.raises(ValueError, match="non-finite"):
        ingestion_handler.validate_lancedb_snapshot(
            [_vehicle("V1", [1.0, 0.0, float("nan"), 0.0])],
            expected_dimension=4,
        )

    with pytest.raises(ValueError, match="non-empty"):
        ingestion_handler.validate_lancedb_snapshot(
            [_vehicle(None, [1.0] * 4)],
            expected_dimension=4,
        )


@pytest.mark.parametrize("missing_id", [None, "", "   "])
def test_normalization_rejects_missing_vehicle_id(ingestion_handler, missing_id) -> None:
    config = {
        "field_mappings": {
            "id": ["id"],
            "make": ["make"],
            "model": ["model"],
        },
        "required_fields": ["id", "make", "model"],
    }

    with pytest.raises(ValueError, match="Missing required fields"):
        ingestion_handler.normalize_vehicle(
            {"id": missing_id, "make": "BMW", "model": "X3"},
            config,
        )


def test_manifest_is_promoted_after_immutable_snapshot(ingestion_handler) -> None:
    s3 = MagicMock()
    s3.put_object.side_effect = [{"ETag": '"snapshot-etag"'}, {}]
    vehicles = [_vehicle("V1", [1.0, 0.0, 0.0, 0.0])]

    with patch.object(ingestion_handler, "s3_client", s3):
        publication = ingestion_handler.write_lancedb_source_to_s3(
            vehicles,
            expected_dimension=4,
        )

    written_keys = [call.kwargs["Key"] for call in s3.put_object.call_args_list]
    assert written_keys[0].startswith("lancedb/snapshots/")
    assert written_keys[1] == "lancedb/manifest.json"
    manifest = json.loads(s3.put_object.call_args_list[-1].kwargs["Body"])
    assert manifest["data_key"] == publication["data_key"]
    assert manifest["version"] == publication["version"]


def test_partial_embedding_failure_does_not_promote_manifest(ingestion_handler) -> None:
    raw = [{"id": "V1"}, {"id": "V2"}]
    normalized = [
        _vehicle("V1", [1.0] * 4),
        _vehicle("V2", [0.0] * 4),
    ]
    partially_embedded = normalized[:1]
    s3 = MagicMock()

    with (
        patch.object(ingestion_handler, "fetch_sample_data_from_s3", return_value=raw),
        patch.object(ingestion_handler, "contextualize_vehicles", return_value=normalized),
        patch.object(
            ingestion_handler,
            "generate_embeddings",
            return_value=partially_embedded,
        ),
        patch.object(ingestion_handler, "s3_client", s3),
        patch.object(ingestion_handler, "MIN_EMBEDDING_SUCCESS_RATIO", 0.95),
        pytest.raises(ValueError, match="success ratio"),
    ):
        ingestion_handler.lambda_handler({}, None)

    s3.put_object.assert_not_called()
