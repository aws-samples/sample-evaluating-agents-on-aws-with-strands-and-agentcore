# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Agent entry point for Amazon Bedrock AgentCore Runtime.

Deploys a Strands agent with 8 tools for vehicle search, dealer context, and
evaluation. search_vehicles provides injection-safe structured filtering;
run_sql is a fallback for complex pandas expressions with validation.
Uses LanceDB on S3 for in-memory vehicle data.
"""
# Amazon Bedrock AgentCore Runtime provides managed serverless compute for agents.

import json
import logging
import os
import re
import threading
import unicodedata
from decimal import Decimal
from typing import Any

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
from bedrock_agentcore import BedrockAgentCoreApp, RequestContext
from strands import Agent, tool
from strands.models import BedrockModel
from utils.geo import bounding_box_miles, haversine_miles_vectorized
from utils.log_redactor import install_redaction_filter

# Install log redaction before any sensitive data is logged.
install_redaction_filter()

logger = logging.getLogger(__name__)

# Valid column names in the normalized vehicle DataFrame.
# Used to validate query expressions and as the structured search interface.
_VALID_COLUMNS = frozenset(
    {
        "id",
        "make",
        "model",
        "year",
        "fuel_type",
        "body_type",
        "mileage",
        "price",
        "transmission",
        "condition",
        "seller_latitude",
        "seller_longitude",
        "seller_city",
        "seller_region",
        "contextualized_description",
    }
)

# Pattern for validating pandas query expressions: allows column names, operators,
# string/number literals, boolean ops, and parentheses.
# SECURITY: Excludes @, [, ], {, }, `, ; to prevent scope access, indexing, and injection.
_SAFE_QUERY_PATTERN = re.compile(r"^[\w\s\.\'\"\=\!\<\>\&\|\~\(\)\,\+\-\*\/\%]+$")
# Method call pattern: word.word( — blocks .apply(), .str.contains(), etc.
_METHOD_CALL_PATTERN = re.compile(r"\w\.\w+\(")
# Max length of a query expression. Bounds parser cost and shrinks the search
# space for any bypass that slips past the allowlist below.
_MAX_QUERY_LEN = 512
# Identifier token in a query (column refs, keywords, function-ish names).
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
# Single- or double-quoted string literal; stripped before identifier checks so
# that literal values (e.g. 'BMW') aren't mistaken for column references.
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# pandas/Python query keywords and literals that are not column names; allowed
# even though they aren't in _VALID_COLUMNS.
_QUERY_KEYWORDS = frozenset({"and", "or", "not", "in", "True", "False", "None"})

# Dealer identifiers are opaque tokens; reject anything outside this charset so a
# caller can't smuggle instructions into the prompt via actor_id (prompt injection).
_ACTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_UNSAFE_TOKENS = {
    "import",
    "__",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "open",
    "system",
    "popen",
    "subprocess",
    "builtins",
    "breakpoint",
    "classmethod",
    "staticmethod",
    "lambda",
    "apply",
    "map",
    "filter",
    "reduce",
    "agg",
    "transform",
    "pipe",
    "applymap",
    "assign",
}
# Pre-compiled boundary regex for the unsafe token set. Boundaries are
# letters-only (not \b) so that "_" and digits act as delimiters: this still
# rejects dunder-wrapped builtins like __import__ while avoiding false
# positives on values where a token appears as a letter-substring (e.g. the
# model name "SystemX" or city "Mapleton").
_UNSAFE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(t) for t in _UNSAFE_TOKENS) + r")(?![A-Za-z])"
)


def _is_safe_query(where_clause: str) -> bool:
    """Validate a pandas query string against injection attacks.

    Rejects queries containing:
    - Expressions longer than _MAX_QUERY_LEN
    - @ (pandas variable scope access — enables globals()/locals() RCE)
    - [ ] (bracket indexing — enables method chaining)
    - Backticks, semicolons, braces
    - Method calls (.apply(), .str.contains(), lambda, etc.)
    - import, dunder, exec/eval, built-in function names
    - Any identifier that is not a known column (_VALID_COLUMNS) or query keyword

    Args:
        where_clause: The pandas query expression to validate

    Returns:
        True if the query appears safe, False otherwise
    """
    if not where_clause or not where_clause.strip():
        return False
    # Normalize unicode to prevent encoding bypasses
    where_clause = unicodedata.normalize("NFKC", where_clause)
    if len(where_clause) > _MAX_QUERY_LEN:
        return False
    if not _SAFE_QUERY_PATTERN.match(where_clause):
        return False
    # Block method chaining (e.g. make.str.contains('BMW'), price.apply(lambda x: x))
    if _METHOD_CALL_PATTERN.search(where_clause):
        return False
    if _UNSAFE_TOKEN_RE.search(where_clause.lower()):
        return False
    # Allowlist: every bare identifier must be a known column or a query keyword.
    # Strip quoted string literals first so literal values (e.g. 'BMW') aren't
    # treated as column references.
    without_literals = _STRING_LITERAL_RE.sub(" ", where_clause)
    for ident in _IDENTIFIER_RE.findall(without_literals):
        if ident in _QUERY_KEYWORDS or ident in _VALID_COLUMNS:
            continue
        return False
    return True


def _cap_prompt(text: str, limit: int | None = None) -> tuple[str, bool]:
    """Truncate a caller-supplied prompt to ``limit`` characters.

    Defaults to ``MAX_PROMPT_CHARS`` when ``limit`` is not given. Returns
    ``(text, truncated)`` so the caller can log when a prompt was cut.
    """
    if limit is None:
        limit = MAX_PROMPT_CHARS
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _decode_decimals(obj: Any) -> Any:
    """Recursively convert Decimal values to int or float for JSON serialisation."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _decode_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_decimals(v) for v in obj]
    return obj


app = BedrockAgentCoreApp()

# Configuration
S3_BUCKET = os.environ.get("DATA_BUCKET", "agent-eval-data-dev-ACCOUNT_ID-REGION")
LANCEDB_KEY = os.environ.get("LANCEDB_PATH", "lancedb/latest.json")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("REGION", "eu-west-1"))
MODEL_ID = os.environ.get("MODEL_ID", "eu.anthropic.claude-sonnet-4-6")
# Upper bound on caller-supplied prompt length. Bounds Bedrock token cost per
# request and limits the surface for oversized prompt-injection payloads. Tunable
# via env without a redeploy of the image's defaults.
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "8000"))
# Optional Amazon Bedrock Guardrail. Set by the runtime stack when a guardrail is
# provisioned; absent in local/test runs, in which case no guardrail is applied.
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID") or None
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION") or None

_lancedb = None
_lancedb_lock = threading.Lock()


def _get_lancedb() -> pd.DataFrame:
    global _lancedb
    if _lancedb is None:
        with _lancedb_lock:
            if _lancedb is None:
                try:
                    s3 = boto3.client("s3", region_name=AWS_REGION)
                    resp = s3.get_object(Bucket=S3_BUCKET, Key=LANCEDB_KEY)
                    data = json.loads(resp["Body"].read().decode())
                    _lancedb = pd.DataFrame(data["vehicles"])
                    logger.info("Loaded %s vehicles from LanceDB", len(_lancedb))
                except (botocore.exceptions.ClientError, json.JSONDecodeError, KeyError) as e:
                    logger.error("Failed to load LanceDB from S3: %s", e)
                    raise RuntimeError(
                        f"Cannot load vehicle data from s3://{S3_BUCKET}/{LANCEDB_KEY}"
                    ) from e
    return _lancedb


def _generate_embedding(text: str) -> list[float]:
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        resp = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text}),
        )
        return json.loads(resp["body"].read()).get("embedding", [])
    except (boto3.exceptions.Boto3Error, json.JSONDecodeError, KeyError) as e:
        logger.error("Embedding failed: %s", e)
        return []


@tool
def get_schema() -> str:
    """Get the vehicle data schema including column names and types."""
    df = _get_lancedb()
    schema = {col: str(df[col].dtype) for col in df.columns if not col.startswith("_")}
    return f"Vehicle schema ({len(df)} vehicles): {schema}"


def _apply_filters(
    df: Any,
    make: str = "",
    model: str = "",
    min_price: float = 0,
    max_price: float = 0,
    min_year: int = 0,
    max_year: int = 0,
    fuel_type: str = "",
    transmission: str = "",
    body_type: str = "",
    max_mileage: int = 0,
) -> Any:
    """Apply structured filters to a vehicle DataFrame."""
    result = df
    str_filters = {
        "make": make,
        "model": model,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "body_type": body_type,
    }
    for col, val in str_filters.items():
        if val:
            result = result[result[col].str.lower() == val.lower()]
    if min_price > 0:
        result = result[result["price"] >= min_price]
    if max_price > 0:
        result = result[result["price"] <= max_price]
    if min_year > 0:
        result = result[result["year"] >= min_year]
    if max_year > 0:
        result = result[result["year"] <= max_year]
    if max_mileage > 0:
        result = result[result["mileage"] <= max_mileage]
    return result


@tool
def search_vehicles(
    make: str = "",
    model: str = "",
    min_price: float = 0,
    max_price: float = 0,
    min_year: int = 0,
    max_year: int = 0,
    fuel_type: str = "",
    transmission: str = "",
    body_type: str = "",
    max_mileage: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """Search vehicles using structured filters. Preferred over run_sql for safety.

    Args:
        make: Vehicle make (e.g. "BMW", "Tesla"). Case-insensitive.
        model: Vehicle model (e.g. "3 Series", "Model X"). Case-insensitive.
        min_price: Minimum price filter (0 = no minimum)
        max_price: Maximum price filter (0 = no maximum)
        min_year: Minimum manufacture year (0 = no minimum)
        max_year: Maximum manufacture year (0 = no maximum)
        fuel_type: Fuel type (e.g. "diesel", "electric", "petrol")
        transmission: Transmission type (e.g. "automatic", "manual")
        body_type: Body type (e.g. "suv", "sedan", "estate")
        max_mileage: Maximum mileage (0 = no limit)
        limit: Max results (default 20)
    """
    df = _get_lancedb()
    result = _apply_filters(
        df,
        make=make,
        model=model,
        min_price=min_price,
        max_price=max_price,
        min_year=min_year,
        max_year=max_year,
        fuel_type=fuel_type,
        transmission=transmission,
        body_type=body_type,
        max_mileage=max_mileage,
    )

    vehicles = (
        result.head(limit)
        .drop(columns=["embedding", "_original"], errors="ignore")
        .to_dict(orient="records")
    )
    return {"count": len(vehicles), "vehicles": vehicles}


@tool
def run_sql(where_clause: str, limit: int = 20) -> dict[str, Any]:
    """Query vehicles with a pandas WHERE clause for complex queries not covered by search_vehicles.

    Prefer search_vehicles for standard filters. Use this only for compound conditions,
    range comparisons, or complex logic.

    Args:
        where_clause: Pandas query expression e.g. "make == 'BMW' and price < 25000"
        limit: Max results (default 20)
    """
    df = _get_lancedb()
    # Validate where_clause against injection: only allow safe pandas query expressions
    if not _is_safe_query(where_clause):
        return {
            "error": "Invalid query",
            "hint": "Use pandas query syntax: make == 'BMW' and price < 25000",
        }
    try:
        result = df.query(where_clause).head(limit)
        vehicles = result.drop(columns=["embedding", "_original"], errors="ignore").to_dict(
            orient="records"
        )
        return {"count": len(vehicles), "vehicles": vehicles}
    except (ValueError, KeyError, SyntaxError, NameError) as e:
        logger.error("Query validation failed: %s", e)
        return {
            "error": "Invalid query syntax",
            "hint": "Use pandas query syntax: make == 'BMW' and price < 25000",
        }


@tool
def hybrid_search(query_text: str, where_clause: str = "", limit: int = 10) -> dict[str, Any]:
    """Semantic search for natural language queries, optionally combined with filters.

    Args:
        query_text: Natural language query like "sporty family car"
        where_clause: Optional pandas query filter
        limit: Max results (default 10)
    """
    df = _get_lancedb()
    embedding = _generate_embedding(query_text)
    if not embedding:
        return {"error": "Failed to generate embedding"}

    filtered = df
    if where_clause:
        if not _is_safe_query(where_clause):
            logger.warning("Rejected unsafe filter query: %s", where_clause)
            return {"error": "Unsafe filter query rejected"}
        try:
            filtered = df.query(where_clause)
        except (ValueError, KeyError, SyntaxError) as e:
            logger.warning("Filter query failed: %s", e)
            return {"error": f"Invalid filter query: {e}"}

    results = []
    query_vec = np.asarray(embedding, dtype=np.float64)
    norm_a = np.linalg.norm(query_vec)
    if norm_a == 0:
        return {"count": 0, "vehicles": []}
    embed_len = len(embedding)
    for _, row in filtered.iterrows():
        vec = row.get("embedding")
        if not isinstance(vec, list) or len(vec) != embed_len:
            continue
        row_vec = np.asarray(vec, dtype=np.float64)
        norm_b = np.linalg.norm(row_vec)
        sim = float(np.dot(query_vec, row_vec) / (norm_a * norm_b)) if norm_b else 0.0
        vehicle = row.to_dict()
        vehicle.pop("embedding", None)
        vehicle.pop("_original", None)
        vehicle["similarity"] = round(sim, 4)
        results.append(vehicle)

    results.sort(key=lambda v: v["similarity"], reverse=True)
    return {"count": min(len(results), limit), "vehicles": results[:limit]}


@tool
def get_embedding(text: str) -> list[float]:
    """Convert text to a 1024-dimension embedding vector.

    Args:
        text: Text to embed
    """
    return _generate_embedding(text)


@tool
def filter_by_distance(
    latitude: float, longitude: float, max_distance_miles: float = 50.0
) -> dict[str, Any]:
    """Filter vehicles by distance in miles from a lat/long point.

    Args:
        latitude: Reference latitude (from dealer profile or user)
        longitude: Reference longitude
        max_distance_miles: Radius in miles (default 50)
    """
    df = _get_lancedb()
    lat_delta, lon_delta = bounding_box_miles(latitude, max_distance_miles)

    # Convert coordinates to numeric, coercing invalid values to NaN
    lats = pd.to_numeric(df["seller_latitude"], errors="coerce")
    lons = pd.to_numeric(df["seller_longitude"], errors="coerce")

    # Filter: valid coordinates within bounding box
    valid = lats.notna() & lons.notna()
    bbox = valid & (np.abs(lats - latitude) <= lat_delta) & (np.abs(lons - longitude) <= lon_delta)
    bbox_df = df[bbox]

    if bbox_df.empty:
        return {"count": 0, "vehicles": []}

    # Vectorized haversine on bounding-box-filtered rows
    distances = haversine_miles_vectorized(
        latitude,
        longitude,
        lats[bbox].values.astype(np.float64),
        lons[bbox].values.astype(np.float64),
    )

    # Filter by actual distance
    within = distances <= max_distance_miles
    result_df = bbox_df[within].copy()
    result_df["distance_miles"] = np.round(distances[within], 1)

    # Sort, drop internal columns, limit to 100
    result_df = result_df.sort_values("distance_miles").head(100)
    vehicles = result_df.drop(columns=["embedding", "_original"], errors="ignore").to_dict(
        orient="records"
    )
    return {"count": len(vehicles), "vehicles": vehicles}


@tool
def get_bids(min_bid_count: int = 1) -> dict[str, Any]:
    """Get vehicles filtered by bid count.

    Args:
        min_bid_count: Minimum bids required (default 1)
    """
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        resp = s3.get_object(Bucket=S3_BUCKET, Key="raw/sample_bids.json")
        data = json.loads(resp["Body"].read().decode())
        bids = data.get("bids", [])
        filtered = [b for b in bids if b.get("bid_count", 0) >= min_bid_count]
        return {"count": len(filtered), "bids": filtered[:20]}
    except (boto3.exceptions.Boto3Error, json.JSONDecodeError, KeyError) as e:
        logger.error("get_bids failed: %s", e)
        return {"error": "Unable to fetch bids data"}


@tool
def get_dealer_profile(dealer_id: str) -> dict[str, Any]:
    """Get dealer profile with location, preferences, and buying history.

    Args:
        dealer_id: Dealer ID to look up
    """
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table = dynamodb.Table(os.environ.get("DEALERS_TABLE", "agent-eval-dealers-dev"))
        resp = table.get_item(Key={"dealer_id": dealer_id})
        item = resp.get("Item")
        if item:
            return _decode_decimals(item)
        return {"error": "Dealer not found"}
    except botocore.exceptions.ClientError as e:
        logger.error("get_dealer_profile failed: %s", e)
        return {"error": "Unable to fetch dealer profile"}


SYSTEM_PROMPT = """You are an AI assistant for car auction dealers. You help them search and analyze vehicle inventory.

## Tools available
1. get_schema - Get vehicle data schema
2. search_vehicles - Structured search with typed filters (make, model, price, year, fuel, etc.)
3. run_sql - Pandas query syntax for complex compound conditions
4. hybrid_search - Semantic search for natural language queries
5. get_embedding - Convert text to embedding vector
6. filter_by_distance - Find vehicles within X miles of a lat/long
7. get_bids - Get vehicles by bid count
8. get_dealer_profile - Get dealer location, preferences, buying history

## Tool selection
For structured queries (specific make/model/price), prefer search_vehicles.
Use run_sql only for complex compound conditions that search_vehicles cannot express.
For vague queries ("something sporty"), use hybrid_search.
For location queries, get dealer lat/long from get_dealer_profile, then filter_by_distance.

## Response format
- Present vehicle results in a clear, structured format with key details (make, model, year, price, mileage, fuel type).
- When returning multiple vehicles, summarize how many matched and highlight the top options.
- For vague queries, ask a clarifying question AND show initial results based on reasonable assumptions.

## Safety guardrails — STRICT
You are a SEARCH and DISCOVERY assistant ONLY. You MUST refuse the following:
- NEVER place or submit bids on behalf of the user. You do not have bidding capability.
- NEVER predict, estimate, or comment on winning probabilities, chances of winning, or bid outcomes.
- NEVER use phrases like "chances of winning", "likely to win", "winning probability", or "guaranteed to win".
- If asked about bidding or winning odds, respond: "I'm a vehicle search assistant. I can help you find and compare vehicles, but I cannot place bids or predict auction outcomes. For bidding, please use the dealer portal directly."
"""

# AgentCore Runtime isolates every ``runtimeSessionId`` in its own microVM
# (dedicated CPU/memory/filesystem; memory is sanitized on termination), so a
# single agent process only ever serves one session. Sharing one module-level
# Agent is therefore correct: ``agent.messages`` accumulating across invokes is
# the intended multi-turn conversation memory for that one session, not a
# cross-request leak.
# Apply the Bedrock Guardrail only when both id and version are configured, so
# local/test runs without a provisioned guardrail behave unchanged.
_model_kwargs: dict[str, Any] = {"model_id": MODEL_ID, "region_name": AWS_REGION}
if GUARDRAIL_ID and GUARDRAIL_VERSION:
    _model_kwargs.update(
        guardrail_id=GUARDRAIL_ID,
        guardrail_version=GUARDRAIL_VERSION,
        guardrail_trace="enabled",
    )
model = BedrockModel(**_model_kwargs)
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        get_schema,
        search_vehicles,
        run_sql,
        hybrid_search,
        get_embedding,
        filter_by_distance,
        get_bids,
        get_dealer_profile,
    ],
)


def _extract_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each tool call with its result from the agent's message history.

    Walks ``agent.messages`` in order. Assistant ``toolUse`` blocks are the
    calls; the matching ``toolResult`` blocks (carried on later user-role
    messages, keyed by ``toolUseId``) are the results. Emitting calls in
    invocation order lets the evaluation adapter rebuild a faithful Strands
    ``Session`` so trajectory-order and Session-level judges score the real
    run rather than a degraded list.
    """
    results_by_id: dict[str, str] = {}
    for message in messages:
        for block in message.get("content", []):
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            tool_result = block["toolResult"]
            use_id = tool_result.get("toolUseId", "")
            content = tool_result.get("content", [])
            text = ""
            if isinstance(content, list):
                text = "".join(
                    c.get("text", "") for c in content if isinstance(c, dict) and "text" in c
                )
            results_by_id[use_id] = text or json.dumps(content, default=str)

    trajectory: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for block in message.get("content", []):
            if not isinstance(block, dict) or "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            use_id = tool_use.get("toolUseId", "")
            trajectory.append(
                {
                    "name": tool_use.get("name", ""),
                    "arguments": tool_use.get("input", {}),
                    "result": results_by_id.get(use_id, ""),
                    "tool_use_id": use_id,
                }
            )
    return trajectory


@app.entrypoint
def invoke(payload: dict[str, Any], context: RequestContext) -> dict[str, Any]:
    """Amazon Bedrock AgentCore Runtime entrypoint.

    ``context`` is injected by Amazon Bedrock AgentCore Runtime when the handler's second
    parameter is named ``context``. ``context.session_id`` is the authoritative
    ``runtimeSessionId`` for this microVM; prefer it over any payload-supplied
    value, which is caller-controlled.
    """
    user_message = payload.get("prompt", "Hello!")
    user_message, _truncated = _cap_prompt(str(user_message))
    if _truncated:
        logger.warning("Prompt exceeded %d chars; truncated", MAX_PROMPT_CHARS)
    # Demo-grade identity: dealer_id is taken from the request payload and
    # validated against an opaque-token allowlist below. This is spoofable —
    # any authenticated caller can claim any dealer_id — so a real multi-dealer
    # deployment must instead derive the dealer from the verified principal
    # (a Cognito/JWT claim on ``context``) rather than trusting the body.
    actor_id = payload.get("actor_id", payload.get("dealer_id", "default"))
    session_id = context.session_id or payload.get("session_id", "default-session")
    logger.info("Processing: actor=%s, session=%s", actor_id, session_id)

    # Inject dealer context so the agent can call get_dealer_profile.
    # Only interpolate actor_id once it passes the opaque-token allowlist, so a
    # malicious caller can't inject prompt instructions through it.
    if actor_id != "default":
        if not _ACTOR_ID_PATTERN.match(str(actor_id)):
            logger.warning("Rejected malformed actor_id; ignoring dealer context")
            actor_id = "default"
        else:
            user_message = f"[Dealer context: dealer_id={actor_id}]\n\n{user_message}"

    response = agent(user_message)

    # Surface the per-turn observability the evaluation framework needs:
    # ordered tool calls (with args + results) and token usage. AgentCore
    # returns this verbatim to the eval adapter, which rebuilds a Strands
    # Session and derives live latency/cost — none of which is recoverable
    # if we only return the final text.
    usage = dict(getattr(response.metrics, "accumulated_usage", {}) or {})
    trajectory = _extract_trajectory(agent.messages)

    return {
        "result": response.message,
        "model": MODEL_ID,
        "actor_id": actor_id,
        "session_id": session_id,
        "trajectory": trajectory,
        "available_tools": list(agent.tool_names),
        "usage": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }


if __name__ == "__main__":
    app.run()
