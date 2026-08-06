# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for warm-runtime LanceDB refresh and bounded local caching."""

import hashlib
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pandas as pd


def _publication(version_seed: int) -> tuple[dict, bytes]:
    generated_at = f"2026-07-{version_seed:02d}T01:00:00+00:00"
    snapshot = {
        "schema_version": 1,
        "generated_at": generated_at,
        "vehicle_count": 1,
        "vector_dimension": 4,
        "vehicles": [
            {
                "id": f"V{version_seed}",
                "make": "BMW",
                "model": "X3",
                "embedding": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    }
    body = json.dumps(snapshot, separators=(",", ":"), sort_keys=True).encode()
    version = hashlib.sha256(body).hexdigest()
    manifest = {
        "schema_version": 1,
        "version": version,
        "data_key": f"lancedb/snapshots/vehicles_v{version_seed}_{version[:12]}.json",
        "data_etag": "etag",
        "generated_at": generated_at,
        "vehicle_count": 1,
        "vector_dimension": 4,
    }
    return manifest, body


class _FakeS3:
    def __init__(self, manifest: dict, snapshot_body: bytes):
        self.manifest = manifest
        self.snapshot_body = snapshot_body

    # N803: these are the S3 API's own parameter names. A fake client has to
    # spell them exactly as boto3 does or the code under test cannot call it.
    def get_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803
        del Bucket
        if Key == "lancedb/manifest.json":
            body = json.dumps(self.manifest).encode()
        elif Key == self.manifest["data_key"]:
            body = self.snapshot_body
        else:
            raise AssertionError(f"Unexpected key: {Key}")
        return {"Body": BytesIO(body)}


def _reset_runtime_state(app) -> None:
    app._lancedb = None
    app._lancedb_frame = None
    app._lancedb_frame_version = None
    app._lancedb_version = None
    app._lancedb_generated_at = None
    app._lancedb_last_refresh_check = 0.0


def test_warm_runtime_activates_new_manifest_version(monkeypatch, tmp_path) -> None:
    from agent import app

    manifest_v1, snapshot_v1 = _publication(1)
    manifest_v2, snapshot_v2 = _publication(2)
    s3 = _FakeS3(manifest_v1, snapshot_v1)
    table_v1 = MagicMock()
    table_v2 = MagicMock()
    table_v1.count_rows.return_value = 1
    table_v2.count_rows.return_value = 1
    _reset_runtime_state(app)

    monkeypatch.setattr(app, "EXPECTED_EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(app, "LANCEDB_REFRESH_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(app, "LANCEDB_CACHE_ROOT", tmp_path)
    with (
        patch("agent.app.boto3.client", return_value=s3),
        patch("agent.app._materialize_lancedb", side_effect=[table_v1, table_v2]),
    ):
        assert app._get_lancedb() is table_v1
        s3.manifest = manifest_v2
        s3.snapshot_body = snapshot_v2
        assert app._get_lancedb() is table_v2

    assert app._lancedb_version == manifest_v2["version"]
    assert app._lancedb_generated_at == manifest_v2["generated_at"]


def test_refresh_failure_retains_last_known_good_table(monkeypatch, tmp_path) -> None:
    from agent import app

    manifest_v1, snapshot_v1 = _publication(1)
    manifest_v2, _ = _publication(2)
    s3 = _FakeS3(manifest_v1, snapshot_v1)
    live_table = MagicMock()
    live_table.count_rows.return_value = 1
    _reset_runtime_state(app)

    monkeypatch.setattr(app, "EXPECTED_EMBEDDING_DIMENSION", 4)
    monkeypatch.setattr(app, "LANCEDB_REFRESH_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(app, "LANCEDB_CACHE_ROOT", tmp_path)
    with (
        patch("agent.app.boto3.client", return_value=s3),
        patch("agent.app._materialize_lancedb", return_value=live_table),
    ):
        assert app._get_lancedb() is live_table
        s3.manifest = manifest_v2
        s3.snapshot_body = b'{"corrupt":true}'
        assert app._get_lancedb() is live_table

    assert app._lancedb_version == manifest_v1["version"]


def test_cache_pruning_keeps_current_and_bounds_generations(monkeypatch, tmp_path) -> None:
    from agent import app

    versions = [f"{index:064x}" for index in range(1, 6)]
    for version in versions:
        (tmp_path / version).mkdir()

    monkeypatch.setattr(app, "LANCEDB_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(app, "LANCEDB_CACHE_GENERATIONS", 3)
    app._prune_lancedb_cache(versions[0])

    remaining = {path.name for path in tmp_path.iterdir()}
    assert versions[0] in remaining
    assert len(remaining) == 3


def test_hybrid_search_prefilters_and_never_requests_all_rows() -> None:
    from agent import app

    table = MagicMock()
    query = MagicMock()
    table.search.return_value = query
    query.distance_type.return_value = query
    query.where.return_value = query
    query.select.return_value = query
    query.limit.return_value = query
    query.to_pandas.return_value = pd.DataFrame([{"id": "V1", "_distance": 0.1, "make": "BMW"}])

    with (
        patch("agent.app._get_lancedb", return_value=table),
        patch("agent.app._generate_embedding", return_value=[1.0, 0.0, 0.0, 0.0]),
    ):
        result = app.hybrid_search(
            query_text="BMW",
            where_clause="make == 'BMW'",
            limit=5,
        )

    query.where.assert_called_once_with("(make = 'BMW')", prefilter=True)
    query.distance_type.assert_called_once_with("cosine")
    query.limit.assert_called_once_with(5)
    table.count_rows.assert_not_called()
    assert result["vehicles"][0]["similarity"] == 0.9
