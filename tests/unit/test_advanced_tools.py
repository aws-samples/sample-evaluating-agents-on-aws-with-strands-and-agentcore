# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for agent tool implementations in agent/app.py.

Tests the Strands @tool decorated functions: search_vehicles, run_sql,
filter_by_distance, hybrid_search, get_bids, get_dealer_profile, get_schema.
"""

import json
from base64 import urlsafe_b64encode
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest
from bedrock_agentcore import RequestContext


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
            "embedding": [1.0, 0.0, 0.0, 0.0],
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
            "embedding": [0.0, 1.0, 0.0, 0.0],
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
            "embedding": [0.8, 0.2, 0.0, 0.0],
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
            "embedding": [0.0, 0.0, 1.0, 0.0],
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
        assert result["count"] == len(SAMPLE_VEHICLES)
        assert result["returned"] == 2
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
# LanceDB materialization and hybrid search tests
# ---------------------------------------------------------------------------


class TestLanceDb:
    """Test the real LanceDB path used by AgentCore Runtime."""

    def test_materializes_vehicle_table(self, tmp_path):
        import agent.app as app

        table = app._materialize_lancedb(
            SAMPLE_VEHICLES.to_dict(orient="records"),
            tmp_path / "vehicle-db",
        )

        assert table.count_rows() == len(SAMPLE_VEHICLES)
        assert table.schema.field("embedding").type.list_size == 4

    def test_materializes_invalid_optional_numbers_as_null(self, tmp_path):
        import agent.app as app

        vehicles = SAMPLE_VEHICLES.to_dict(orient="records")
        vehicles[0]["year"] = "Unknown"
        vehicles[0]["price"] = "not available"

        table = app._materialize_lancedb(vehicles, tmp_path / "vehicle-db")
        row = table.search().where("id = 'V001'").limit(1).to_pandas().iloc[0]

        assert pd.isna(row["year"])
        assert pd.isna(row["price"])

    def test_hybrid_search_uses_lancedb_cosine_search(self, tmp_path):
        import agent.app as app

        table = app._materialize_lancedb(
            SAMPLE_VEHICLES.to_dict(orient="records"),
            tmp_path / "vehicle-db",
        )
        fn = _get_tool_fn("hybrid_search")

        with (
            patch("agent.app._get_lancedb", return_value=table),
            patch("agent.app._lancedb_frame", None),
            patch("agent.app._generate_embedding", return_value=[1.0, 0.0, 0.0, 0.0]),
        ):
            result = fn(query_text="BMW SUV", limit=2)

        assert result["count"] == 2
        assert result["vehicles"][0]["id"] == "V001"
        assert result["vehicles"][0]["similarity"] == pytest.approx(1.0)
        assert "embedding" not in result["vehicles"][0]

    def test_hybrid_search_applies_safe_filter(self, tmp_path):
        import agent.app as app

        table = app._materialize_lancedb(
            SAMPLE_VEHICLES.to_dict(orient="records"),
            tmp_path / "vehicle-db",
        )
        fn = _get_tool_fn("hybrid_search")

        with (
            patch("agent.app._get_lancedb", return_value=table),
            patch("agent.app._lancedb_frame", None),
            patch("agent.app._generate_embedding", return_value=[1.0, 0.0, 0.0, 0.0]),
        ):
            result = fn(query_text="electric", where_clause="make == 'Tesla'", limit=5)

        assert result["count"] == 1
        assert result["vehicles"][0]["id"] == "V002"


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

    def test_limit_preserves_total_match_count(self):
        fn = _get_tool_fn("filter_by_distance")
        result = fn(
            latitude=51.5074,
            longitude=-0.1278,
            max_distance_miles=500.0,
            limit=2,
        )
        assert result["count"] > result["returned"]
        assert result["returned"] == len(result["vehicles"]) == 2

    def test_candidate_ids_restrict_distance_results(self):
        fn = _get_tool_fn("filter_by_distance")
        result = fn(
            latitude=51.5074,
            longitude=-0.1278,
            max_distance_miles=500.0,
            candidate_ids=["V001", "V003"],
        )
        assert {vehicle["id"] for vehicle in result["vehicles"]} == {"V001", "V003"}

    def test_candidate_ids_reject_malformed_values(self):
        fn = _get_tool_fn("filter_by_distance")
        result = fn(
            latitude=51.5074,
            longitude=-0.1278,
            candidate_ids=["V001", "invalid vehicle id"],
        )
        assert result == {"error": "Invalid candidate vehicle IDs"}


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


class TestTrustedIdentity:
    """Dealer identity must come from AgentCore context, never request JSON."""

    def test_rejects_body_identity(self):
        import agent.app as app

        with (
            patch.object(app, "IDENTITY_MODE", "runtime_user_id"),
            pytest.raises(app.IdentityError, match="not the body"),
        ):
            app._resolve_actor_id(
                {"dealer_id": "DLR24946"},
                RequestContext(
                    session_id="session",
                    request_headers={"X-Amzn-Bedrock-AgentCore-Runtime-User-Id": "DLR24946"},
                ),
            )

    def test_uses_runtime_user_id_header(self):
        import agent.app as app

        with patch.object(app, "IDENTITY_MODE", "runtime_user_id"):
            actor_id = app._resolve_actor_id(
                {},
                RequestContext(
                    session_id="session",
                    request_headers={"X-Amzn-Bedrock-AgentCore-Runtime-User-Id": "DLR24946"},
                ),
            )
        assert actor_id == "DLR24946"

    def test_single_tenant_mode_ignores_runtime_user_id(self):
        import agent.app as app

        with (
            patch.object(app, "IDENTITY_MODE", "single_tenant"),
            patch.object(app, "DEFAULT_ACTOR_ID", "default"),
        ):
            actor_id = app._resolve_actor_id(
                {},
                RequestContext(
                    session_id="session",
                    request_headers={"X-Amzn-Bedrock-AgentCore-Runtime-User-Id": "DLR24946"},
                ),
            )
        assert actor_id == "default"


class TestScopedDealerProfile:
    """The model can access only the authenticated dealer's profile."""

    def test_hides_list_and_injects_trusted_dealer_id(self):
        import agent.app as app

        list_tool = MagicMock()
        list_tool.tool_name = "dealer_target___listDealers"
        list_tool.mcp_tool.name = "dealer_target___listDealers"
        profile_tool = MagicMock()
        profile_tool.tool_name = "dealer_target___getDealerProfile"
        profile_tool.mcp_tool.name = "dealer_target___getDealerProfile"
        gateway_client = MagicMock()
        gateway_client.list_tools_sync.return_value = [list_tool, profile_tool]
        gateway_client.call_tool_sync.return_value = {
            "status": "success",
            "content": [{"text": '{"dealer_id":"DLR24946"}'}],
        }

        scoped_tool = app._scoped_dealer_profile_tool(gateway_client, "DLR24946")
        result = scoped_tool._tool_func()

        assert scoped_tool.tool_name == "get_dealer_profile"
        assert scoped_tool.tool_spec["inputSchema"]["json"]["properties"] == {}
        assert result["status"] == "success"
        gateway_client.call_tool_sync.assert_called_once_with(
            tool_use_id=ANY,
            name="dealer_target___getDealerProfile",
            arguments={"dealer_id": "DLR24946"},
        )


class TestEvaluationTraceAuthorization:
    """Tool trajectories are available only to authorized evaluators."""

    def test_authorized_token_is_removed_and_accepted(self):
        import agent.app as app

        payload = {"prompt": "hello", "evaluation_token": "expected"}
        with patch("agent.app._get_evaluation_trace_token", return_value="expected"):
            assert app._evaluation_trace_authorized(payload) is True
        assert "evaluation_token" not in payload

    def test_wrong_token_fails_closed(self):
        import agent.app as app

        payload = {"evaluation_token": "wrong"}
        with patch("agent.app._get_evaluation_trace_token", return_value="expected"):
            assert app._evaluation_trace_authorized(payload) is False
        assert "evaluation_token" not in payload

    def test_authorized_no_tool_invocation_includes_lancedb_freshness(self):
        import agent.app as app

        generated_at = "2026-07-21T01:00:00+00:00"
        response = MagicMock()
        response.message = {"role": "assistant", "content": [{"text": "hello"}]}
        response.metrics.accumulated_usage = {}
        agent = MagicMock()
        agent.return_value = response
        agent.messages = []
        agent.tool_names = []

        def load_lancedb():
            app._lancedb_generated_at = generated_at
            app._lancedb_version = "a" * 64
            return MagicMock()

        with (
            patch.object(app, "_evaluation_trace_authorized", return_value=True),
            patch.object(app, "_get_lancedb", side_effect=load_lancedb) as get_lancedb,
            patch.object(app, "_build_memory_session_manager", return_value=None),
            patch.object(app, "_build_agent", return_value=agent),
            patch.object(app, "GATEWAY_URL", None),
        ):
            result = app.invoke(
                {"prompt": "Say hello without using tools"},
                RequestContext(session_id="session"),
            )

        get_lancedb.assert_called_once_with()
        assert result["trajectory"] == []
        assert result["last_refresh_time"] == generated_at
        assert result["lancedb_version"] == "a" * 64

    def test_authorized_trace_contains_only_current_invocation(self):
        import agent.app as app

        prior_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "prior",
                            "name": "hybrid_search",
                            "input": {"query_text": "old request"},
                        }
                    }
                ],
            }
        ]
        current_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "current",
                            "name": "search_vehicles",
                            "input": {"body_type": "estate"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "current",
                            "content": [{"text": '{"count": 0, "vehicles": []}'}],
                        }
                    }
                ],
            },
        ]
        response = MagicMock()
        response.message = {"role": "assistant", "content": [{"text": "No matches"}]}
        response.metrics.latest_agent_invocation.usage = {
            "inputTokens": 7,
            "outputTokens": 3,
            "totalTokens": 10,
        }
        response.metrics.accumulated_usage = {
            "inputTokens": 107,
            "outputTokens": 103,
            "totalTokens": 210,
        }
        agent = MagicMock()
        agent.messages = list(prior_messages)
        agent.tool_names = ["search_vehicles", "hybrid_search"]

        def invoke_agent(_prompt):
            agent.messages.extend(current_messages)
            return response

        agent.side_effect = invoke_agent
        with (
            patch.object(app, "_evaluation_trace_authorized", return_value=True),
            patch.object(app, "_get_lancedb"),
            patch.object(app, "_build_memory_session_manager", return_value=MagicMock()),
            patch.object(app, "_build_agent", return_value=agent),
            patch.object(app, "GATEWAY_URL", None),
        ):
            result = app.invoke(
                {"prompt": "What about estates instead?"},
                RequestContext(session_id="session"),
            )

        assert [call["name"] for call in result["trajectory"]] == ["search_vehicles"]
        assert result["usage"] == {
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        }

    def test_uses_fixed_fallback_without_runtime_user_id(self):
        import agent.app as app

        with (
            patch.object(app, "IDENTITY_MODE", "runtime_user_id"),
            patch.object(app, "DEFAULT_ACTOR_ID", "default"),
        ):
            actor_id = app._resolve_actor_id({}, RequestContext(session_id="session"))
        assert actor_id == "default"

    def test_reads_dealer_claim_from_verified_jwt(self):
        import agent.app as app

        header = urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
        claims = urlsafe_b64encode(b'{"custom:dealer_id":"DLR24946"}').decode().rstrip("=")
        token = f"{header}.{claims}.verified-by-agentcore"
        with (
            patch.object(app, "IDENTITY_MODE", "jwt_claim"),
            patch.object(app, "ACTOR_ID_CLAIM", "custom:dealer_id"),
        ):
            actor_id = app._resolve_actor_id(
                {},
                RequestContext(
                    session_id="session",
                    request_headers={"Authorization": f"Bearer {token}"},
                ),
            )
        assert actor_id == "DLR24946"


class TestSystemPromptContracts:
    """Keep user-visible safety and empty-result behavior explicit."""

    def test_safety_refusal_and_empty_results_are_explained(self):
        import agent.app as app

        assert "I cannot place bids or predict auction outcomes" in app.SYSTEM_PROMPT
        assert "explicitly state the active filters" in app.SYSTEM_PROMPT
        assert "search returned zero results" in app.SYSTEM_PROMPT

    def test_multi_turn_refinements_repeat_complete_active_filters(self):
        import agent.app as app

        assert "complete active filter set" in app.SYSTEM_PROMPT
        assert "preserve every unchanged filter" in app.SYSTEM_PROMPT
        assert "Do not filter a previous result set only in prose" in app.SYSTEM_PROMPT
        assert "Do not replay a superseded body type" in app.SYSTEM_PROMPT

    def test_location_lookup_is_mandatory_and_sequential(self):
        import agent.app as app

        assert "you MUST first call get_dealer_profile" in app.SYSTEM_PROMPT
        assert "Never infer dealer coordinates from memory" in app.SYSTEM_PROMPT
        assert "never issue get_dealer_profile and filter_by_distance" in app.SYSTEM_PROMPT
        assert "filter_by_distance candidate_ids" in app.SYSTEM_PROMPT

    def test_searches_do_not_add_preferences_or_fallbacks(self):
        import agent.app as app

        assert "Never add dealer preferences" in app.SYSTEM_PROMPT
        assert "Choose exactly one primary inventory search strategy" in app.SYSTEM_PROMPT
        assert "Do not call multiple primary search tools" in app.SYSTEM_PROMPT
        assert "ask before broadening it" in app.SYSTEM_PROMPT

    def test_vehicle_results_use_compact_non_table_format(self):
        import agent.app as app

        assert "one compact bullet per vehicle" in app.SYSTEM_PROMPT
        assert "Do not use Markdown tables" in app.SYSTEM_PROMPT

    def test_agent_disables_content_streaming_callback(self):
        import agent.app as app

        with patch("agent.app.Agent") as agent_cls:
            app._build_agent([], None)
        assert agent_cls.call_args.kwargs["callback_handler"] is None


class TestRequestToolSelection:
    """Expose the minimum tool capability set needed for each request."""

    @staticmethod
    def _names(tools):
        import agent.app as app

        return [app._tool_name(tool_value) for tool_value in tools]

    def test_structured_location_request(self):
        import agent.app as app

        prompt = "Find me diesel SUVs under 25k near my dealership"
        assert self._names(app._select_local_tools(prompt)) == [
            "search_vehicles",
            "filter_by_distance",
        ]
        assert app._needs_dealer_profile(prompt) is True

    def test_semantic_request(self):
        import agent.app as app

        prompt = "Show me something sporty and automatic for a family"
        assert self._names(app._select_local_tools(prompt)) == ["hybrid_search"]
        assert app._needs_dealer_profile(prompt) is False

    def test_long_term_memory_requires_explicit_intent(self):
        import agent.app as app

        assert app._needs_long_term_memory("Find BMW 3 Series with under 50k miles") is False
        assert app._needs_long_term_memory("Use my preferences to find a vehicle") is True
        assert app._needs_long_term_memory("What do I usually buy?") is True

    def test_memory_manager_keeps_short_term_without_implicit_retrieval(self):
        import agent.app as app

        with (
            patch.object(app, "MEMORY_ID", "memory-123"),
            patch("agent.app.AgentCoreMemorySessionManager") as manager_cls,
        ):
            app._build_memory_session_manager("dealer", "session")

        config = manager_cls.call_args.args[0]
        assert config.memory_id == "memory-123"
        assert config.session_id == "session"
        assert config.actor_id == "dealer"
        assert config.retrieval_config is None

    def test_memory_manager_retrieves_long_term_on_explicit_request(self):
        import agent.app as app

        with (
            patch.object(app, "MEMORY_ID", "memory-123"),
            patch("agent.app.AgentCoreMemorySessionManager") as manager_cls,
        ):
            app._build_memory_session_manager("dealer", "session", retrieve_long_term=True)

        config = manager_cls.call_args.args[0]
        assert config.retrieval_config == app._MEMORY_NAMESPACES

    def test_follow_up_keeps_structured_search(self):
        import agent.app as app

        assert self._names(app._select_local_tools("What about estates instead?")) == [
            "search_vehicles"
        ]

    def test_specialized_tools_are_opt_in(self):
        import agent.app as app

        assert self._names(app._select_local_tools("Show the data schema")) == ["get_schema"]
        assert self._names(app._select_local_tools("Get the bid count")) == ["get_bids"]
        assert self._names(app._select_local_tools("Generate an embedding")) == ["get_embedding"]

    def test_agent_prompt_names_only_exposed_tools(self):
        import agent.app as app

        with patch("agent.app.Agent") as agent_cls:
            app._build_agent([app.search_vehicles], None)
        request_prompt = agent_cls.call_args.kwargs["system_prompt"]
        assert "Exposed tools for this request: search_vehicles." in request_prompt
