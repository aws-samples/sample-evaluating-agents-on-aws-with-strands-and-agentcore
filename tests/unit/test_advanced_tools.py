# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for agent tool implementations in agent/app.py.

Tests the Strands @tool decorated functions: search_vehicles, run_sql,
filter_by_distance, hybrid_search, get_bids, get_dealer_profile, get_schema.
"""

import json
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_VEHICLES = pd.DataFrame(
    [
        {
            "id": "V001",
            "make": "BMW",
            "model": "X3",
            "year": 2023,
            "price": 35000,
            "fuel_type": "diesel",
            "transmission": "automatic",
            "body_type": "suv",
            "mileage": 12000,
            "seller_latitude": 51.5074,
            "seller_longitude": -0.1278,
            "seller_city": "London",
            "seller_region": "South East",
            "contextualized_description": "BMW X3 diesel automatic SUV",
        },
        {
            "id": "V002",
            "make": "Tesla",
            "model": "Model 3",
            "year": 2023,
            "price": 38000,
            "fuel_type": "electric",
            "transmission": "automatic",
            "body_type": "sedan",
            "mileage": 5000,
            "seller_latitude": 53.4808,
            "seller_longitude": -2.2426,
            "seller_city": "Manchester",
            "seller_region": "North West",
            "contextualized_description": "Tesla Model 3 electric sedan",
        },
        {
            "id": "V003",
            "make": "BMW",
            "model": "320i",
            "year": 2022,
            "price": 28000,
            "fuel_type": "petrol",
            "transmission": "manual",
            "body_type": "sedan",
            "mileage": 18000,
            "seller_latitude": 52.5086,
            "seller_longitude": -1.8853,
            "seller_city": "Birmingham",
            "seller_region": "West Midlands",
            "contextualized_description": "BMW 320i petrol manual sedan",
        },
        {
            "id": "V004",
            "make": "Audi",
            "model": "Q5",
            "year": 2021,
            "price": 22000,
            "fuel_type": "diesel",
            "transmission": "automatic",
            "body_type": "suv",
            "mileage": 45000,
            "seller_latitude": None,
            "seller_longitude": None,
            "seller_city": "Unknown",
            "seller_region": "Unknown",
            "contextualized_description": "Audi Q5 diesel automatic SUV",
        },
    ]
)


@pytest.fixture(autouse=True)
def _patch_lancedb():
    """Patch _get_lancedb to return sample DataFrame for all tests."""
    with patch("agent.app._get_lancedb", return_value=SAMPLE_VEHICLES.copy()):
        yield


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    """Set environment variables for agent module."""
    monkeypatch.setenv("DATA_BUCKET", "test-bucket")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")


# ---------------------------------------------------------------------------
# Import tools after patching to avoid module-level side effects
# ---------------------------------------------------------------------------


def _get_tool_fn(tool_name: str):
    """Import and return a tool function from agent.app."""
    import agent.app as app

    tool_map = {
        "get_schema": app.get_schema,
        "search_vehicles": app.search_vehicles,
        "run_sql": app.run_sql,
        "hybrid_search": app.hybrid_search,
        "get_embedding": app.get_embedding,
        "filter_by_distance": app.filter_by_distance,
        "get_bids": app.get_bids,
    }
    return tool_map[tool_name]


# ---------------------------------------------------------------------------
# search_vehicles tests
# ---------------------------------------------------------------------------


class TestSearchVehicles:
    """Test search_vehicles structured filter tool."""

    def test_filter_by_make(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(make="BMW")
        assert result["count"] == 2
        assert all(v["make"] == "BMW" for v in result["vehicles"])

    def test_filter_by_fuel_type(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(fuel_type="electric")
        assert result["count"] == 1
        assert result["vehicles"][0]["make"] == "Tesla"

    def test_filter_by_price_range(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(min_price=30000, max_price=40000)
        assert result["count"] == 2
        assert all(30000 <= v["price"] <= 40000 for v in result["vehicles"])

    def test_filter_by_year(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(min_year=2023)
        assert result["count"] == 2
        assert all(v["year"] >= 2023 for v in result["vehicles"])

    def test_filter_by_transmission(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(transmission="manual")
        assert result["count"] == 1
        assert result["vehicles"][0]["id"] == "V003"

    def test_no_results(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(make="Porsche")
        assert result["count"] == 0
        assert result["vehicles"] == []

    def test_limit(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(limit=2)
        assert len(result["vehicles"]) == 2

    def test_combined_filters(self):
        fn = _get_tool_fn("search_vehicles")
        result = fn(make="BMW", fuel_type="diesel")
        assert result["count"] == 1
        assert result["vehicles"][0]["id"] == "V001"


# ---------------------------------------------------------------------------
# run_sql tests
# ---------------------------------------------------------------------------


class TestExecuteSql:
    """Test run_sql pandas query tool."""

    def test_valid_query(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="make == 'BMW' and price < 30000")
        assert result["count"] == 1
        assert result["vehicles"][0]["id"] == "V003"

    def test_injection_blocked(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="__import__('os').system('rm -rf /')")
        assert "error" in result

    def test_method_call_blocked(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="make.str.contains('BMW')")
        assert "error" in result

    def test_scope_access_blocked(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="@globals()['os']")
        assert "error" in result

    def test_empty_query_rejected(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="")
        assert "error" in result

    def test_limit_applied(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="price > 0", limit=2)
        assert len(result["vehicles"]) == 2

    def test_invalid_column_returns_error(self):
        fn = _get_tool_fn("run_sql")
        result = fn(where_clause="nonexistent_col == 'x'")
        assert "error" in result


# ---------------------------------------------------------------------------
# filter_by_distance tests
# ---------------------------------------------------------------------------


class TestFilterByDistance:
    """Test filter_by_distance geo tool."""

    def test_london_200_mile_radius(self):
        fn = _get_tool_fn("filter_by_distance")
        result = fn(latitude=51.5074, longitude=-0.1278, max_distance_miles=200.0)
        assert result["count"] > 0
        assert any(v["id"] == "V001" for v in result["vehicles"])
        assert all("distance_miles" in v for v in result["vehicles"])

    def test_sorted_by_proximity(self):
        fn = _get_tool_fn("filter_by_distance")
        result = fn(latitude=51.5074, longitude=-0.1278, max_distance_miles=200.0)
        distances = [v["distance_miles"] for v in result["vehicles"]]
        assert distances == sorted(distances)

    def test_small_radius_london_only(self):
        fn = _get_tool_fn("filter_by_distance")
        result = fn(latitude=51.5074, longitude=-0.1278, max_distance_miles=1.0)
        assert result["count"] <= 1
        if result["count"] == 1:
            assert result["vehicles"][0]["id"] == "V001"

    def test_null_coordinates_skipped(self):
        """V004 has None coordinates and should be excluded."""
        fn = _get_tool_fn("filter_by_distance")
        result = fn(latitude=52.0, longitude=-1.5, max_distance_miles=500.0)
        assert all(v["id"] != "V004" for v in result["vehicles"])

    def test_haversine_accuracy(self):
        """London to Birmingham is approx 100-105 miles."""
        fn = _get_tool_fn("filter_by_distance")
        result = fn(latitude=51.5074, longitude=-0.1278, max_distance_miles=120.0)
        bham = [v for v in result["vehicles"] if v["id"] == "V003"]
        assert len(bham) == 1
        assert 95 <= bham[0]["distance_miles"] <= 110


# ---------------------------------------------------------------------------
# get_schema tests
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Test get_schema tool."""

    def test_returns_schema_string(self):
        fn = _get_tool_fn("get_schema")
        result = fn()
        assert "4 vehicles" in result
        assert "make" in result
        assert "price" in result


# ---------------------------------------------------------------------------
# get_bids tests
# ---------------------------------------------------------------------------


class TestGetBids:
    """Test get_bids tool."""

    def test_returns_bids(self):
        fn = _get_tool_fn("get_bids")
        mock_bids = {
            "bids": [
                {"vehicle_id": "V001", "bid_count": 3},
                {"vehicle_id": "V002", "bid_count": 0},
            ]
        }
        with patch("agent.app.boto3") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.client.return_value = mock_s3
            mock_s3.get_object.return_value = {
                "Body": MagicMock(read=MagicMock(return_value=json.dumps(mock_bids).encode()))
            }
            result = fn(min_bid_count=1)
            mock_boto.client.assert_called_with("s3", region_name=ANY)
            mock_s3.get_object.assert_called_once()
            assert result["count"] == 1
            assert result["bids"][0]["vehicle_id"] == "V001"


# ---------------------------------------------------------------------------
# Dealer profiles are now served through the AgentCore Gateway (MCP), not a
# local tool, so there is no in-process get_dealer_profile to unit-test here.
# Gateway wiring is exercised by the deployed smoke test (scripts/post_deploy_eval.py).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Query safety tests (cross-cutting)
# ---------------------------------------------------------------------------


class TestQuerySafety:
    """Test _is_safe_query validation."""

    def test_safe_queries(self):
        from agent.app import _is_safe_query

        safe = [
            "make == 'BMW'",
            "price > 20000 and price < 40000",
            "year >= 2022",
            "fuel_type == 'diesel' or fuel_type == 'electric'",
        ]
        for q in safe:
            assert _is_safe_query(q) is True, f"Should be safe: {q}"

    def test_unsafe_queries(self):
        from agent.app import _is_safe_query

        unsafe = [
            "__import__('os')",
            "exec('print(1)')",
            "eval('1+1')",
            "@globals()",
            "make.apply(lambda x: x)",
            "make.str.contains('BMW')",
            "",
            "   ",
        ]
        for q in unsafe:
            assert _is_safe_query(q) is False, f"Should be unsafe: {q}"

    def test_unknown_column_rejected(self):
        """Identifiers outside _VALID_COLUMNS are rejected (allowlist defense)."""
        from agent.app import _is_safe_query

        for q in [
            "embedding == 1",  # internal column, never queryable
            "_original == 'x'",  # raw payload column
            "nonexistent_col == 'x'",
            "os == 'linux'",  # not a column
        ]:
            assert _is_safe_query(q) is False, f"Should be unsafe: {q}"

    def test_string_literals_not_treated_as_columns(self):
        """Quoted values may be arbitrary; only bare identifiers are allowlisted."""
        from agent.app import _is_safe_query

        # 'os' here is a string literal value, not a column reference.
        assert _is_safe_query("make == 'os'") is True

    def test_overlong_query_rejected(self):
        """Queries beyond _MAX_QUERY_LEN are rejected before parsing."""
        from agent.app import _MAX_QUERY_LEN, _is_safe_query

        long_clause = "price > 0 or " * (_MAX_QUERY_LEN // 12 + 5)
        assert len(long_clause) > _MAX_QUERY_LEN
        assert _is_safe_query(long_clause) is False


class TestCapPrompt:
    """Test the caller-supplied prompt length cap (_cap_prompt)."""

    def test_short_prompt_unchanged(self):
        from agent.app import _cap_prompt

        text, truncated = _cap_prompt("hello", limit=100)
        assert text == "hello"
        assert truncated is False

    def test_long_prompt_truncated(self):
        from agent.app import _cap_prompt

        text, truncated = _cap_prompt("x" * 200, limit=50)
        assert len(text) == 50
        assert truncated is True
