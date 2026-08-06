# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Agent entry point for Amazon Bedrock AgentCore Runtime.

Deploys a Strands agent with 8 tools for vehicle search, dealer context, and
evaluation. search_vehicles provides injection-safe structured filtering;
run_sql is a fallback for complex pandas expressions with validation.
Materializes versioned vector-enriched records from Amazon S3 into a local
LanceDB table and refreshes warm runtime processes from an atomic manifest.
"""
# Amazon Bedrock AgentCore Runtime provides managed serverless compute for agents.

import ast
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import unicodedata
import uuid
from base64 import urlsafe_b64decode
from binascii import Error as BinasciiError
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from bedrock_agentcore import BedrockAgentCoreApp, RequestContext
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from botocore.config import Config as BotocoreConfig
from strands import Agent, tool
from strands.models import BedrockModel
from utils.gateway import build_gateway_mcp_client
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
_PUBLIC_VEHICLE_COLUMNS = tuple(sorted(_VALID_COLUMNS))

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
# caller can't smuggle instructions into the prompt via an identity claim.
_ACTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_VEHICLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RUNTIME_USER_ID_HEADER = "x-amzn-bedrock-agentcore-runtime-user-id"
# header.payload.signature — a compact JWS always has exactly three segments.
_JWT_SEGMENT_COUNT = 3
_SNAPSHOT_VERSION_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_SNAPSHOT_KEY_PATTERN = re.compile(
    r"^lancedb/snapshots/vehicles_[A-Za-z0-9_.-]+_[a-f0-9]{12}\.json$"
)

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


def _is_safe_query(where_clause: str) -> bool:  # noqa: PLR0911 - one guard per rejected construct
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


app = BedrockAgentCoreApp()

# Configuration
S3_BUCKET = os.environ.get("DATA_BUCKET", "agent-eval-data-dev-ACCOUNT_ID-REGION")
LANCEDB_KEY = os.environ.get("LANCEDB_PATH", "lancedb/manifest.json")
LANCEDB_CACHE_ROOT = Path(
    os.environ.get("LANCEDB_CACHE_ROOT", str(Path(tempfile.gettempdir()) / "agent-eval-lancedb"))
)
LANCEDB_REFRESH_INTERVAL_SECONDS = max(
    5, int(os.environ.get("LANCEDB_REFRESH_INTERVAL_SECONDS", "60"))
)
LANCEDB_CACHE_GENERATIONS = max(2, int(os.environ.get("LANCEDB_CACHE_GENERATIONS", "3")))
EXPECTED_EMBEDDING_DIMENSION = int(os.environ.get("EXPECTED_EMBEDDING_DIMENSION", "1024"))
MAX_TOOL_RESULTS = max(1, int(os.environ.get("MAX_TOOL_RESULTS", "20")))
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
# Amazon Bedrock AgentCore Memory ID. Set by the runtime stack; when absent
# (local/test runs) the agent falls back to in-process conversation only.
MEMORY_ID = os.environ.get("MEMORY_ID") or None
# Amazon Bedrock AgentCore Gateway MCP endpoint for the dealer-profile tool.
# Set by the runtime stack; when absent the agent has no dealer-profile tool.
GATEWAY_URL = os.environ.get("GATEWAY_URL") or None
# Identity is never accepted from the JSON body. IAM deployments receive an
# upstream-resolved runtimeUserId; JWT deployments read a claim from the token
# that AgentCore Runtime has already authenticated.
IDENTITY_MODE = os.environ.get("IDENTITY_MODE", "single_tenant").strip().lower()
DEFAULT_ACTOR_ID = os.environ.get("DEFAULT_ACTOR_ID", "default")
ACTOR_ID_CLAIM = os.environ.get("ACTOR_ID_CLAIM", "custom:dealer_id")
# Secret identifier only; the token itself is fetched from Secrets Manager and
# never stored in source or runtime environment variables.
EVALUATION_TRACE_SECRET_ID = os.environ.get("EVALUATION_TRACE_SECRET_ID") or None
# Namespaces the built-in memory strategies write to. {actorId} is substituted
# by AgentCore Memory per dealer, giving each dealer isolated long-term memory.
_MEMORY_NAMESPACES = {
    "/preferences/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.7),
    "/facts/{actorId}/": RetrievalConfig(top_k=10, relevance_score=0.3),
}

_lancedb = None
_lancedb_frame: pd.DataFrame | None = None
_lancedb_frame_version: str | None = None
_lancedb_version: str | None = None
_lancedb_generated_at: str | None = None
_lancedb_last_refresh_check = 0.0
_lancedb_lock = threading.Lock()
_evaluation_trace_token: str | None = None
_evaluation_trace_token_lock = threading.Lock()


class IdentityError(ValueError):
    """Raised when a request has no trustworthy dealer identity."""


def _validated_actor_id(value: Any, *, source: str) -> str:
    actor_id = str(value or "")
    if not _ACTOR_ID_PATTERN.fullmatch(actor_id):
        raise IdentityError(f"Invalid dealer identity from {source}")
    return actor_id


def _request_headers(context: RequestContext) -> dict[str, str]:
    """Return request headers with case-insensitive keys."""
    return {str(key).lower(): str(value) for key, value in (context.request_headers or {}).items()}


def _jwt_claim(context: RequestContext, claim_name: str) -> Any:
    """Read a claim from the JWT already verified by AgentCore Runtime.

    Signature, issuer, expiry, client, audience, and scope validation belongs to
    the Runtime custom JWT authorizer. This function only decodes that verified
    token so application authorization can use its dealer claim.
    """
    authorization = _request_headers(context).get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise IdentityError("Authenticated bearer token is required")

    parts = token.split(".")
    if len(parts) != _JWT_SEGMENT_COUNT:
        raise IdentityError("Malformed authenticated bearer token")
    try:
        payload_segment = parts[1] + ("=" * (-len(parts[1]) % 4))
        claims = json.loads(urlsafe_b64decode(payload_segment).decode("utf-8"))
    except (BinasciiError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityError("Malformed authenticated bearer token claims") from exc
    if not isinstance(claims, dict) or claim_name not in claims:
        raise IdentityError(f"Authenticated token is missing {claim_name!r}")
    return claims[claim_name]


def _resolve_actor_id(payload: dict[str, Any], context: RequestContext) -> str:
    """Resolve dealer identity exclusively from the configured trust boundary."""
    if "actor_id" in payload or "dealer_id" in payload:
        raise IdentityError(
            "Dealer identity must be supplied by the authenticated runtime context, not the body"
        )

    if IDENTITY_MODE == "jwt_claim":
        return _validated_actor_id(_jwt_claim(context, ACTOR_ID_CLAIM), source=ACTOR_ID_CLAIM)
    if IDENTITY_MODE == "runtime_user_id":
        runtime_user_id = _request_headers(context).get(_RUNTIME_USER_ID_HEADER)
        if runtime_user_id:
            return _validated_actor_id(runtime_user_id, source="runtimeUserId")
        return _validated_actor_id(DEFAULT_ACTOR_ID, source="DEFAULT_ACTOR_ID")
    if IDENTITY_MODE == "single_tenant":
        return _validated_actor_id(DEFAULT_ACTOR_ID, source="DEFAULT_ACTOR_ID")
    raise IdentityError(f"Unsupported IDENTITY_MODE: {IDENTITY_MODE!r}")


def _read_evaluation_trace_secret() -> str:
    """Fetch the evaluation trace token from Secrets Manager.

    Returns:
        The secret string.

    Raises:
        ValueError: If the secret holds no string value.
    """
    response = boto3.client("secretsmanager", region_name=AWS_REGION).get_secret_value(
        SecretId=EVALUATION_TRACE_SECRET_ID
    )
    token = response.get("SecretString")
    if not isinstance(token, str) or not token:
        raise ValueError("Evaluation trace secret has no SecretString")
    return token


def _get_evaluation_trace_token() -> str | None:
    """Load and cache the privileged evaluation token from Secrets Manager."""
    global _evaluation_trace_token  # noqa: PLW0603 - process-wide warm-start cache
    if not EVALUATION_TRACE_SECRET_ID:
        return None
    if _evaluation_trace_token is None:
        with _evaluation_trace_token_lock:
            if _evaluation_trace_token is None:
                try:
                    _evaluation_trace_token = _read_evaluation_trace_secret()
                except (botocore.exceptions.ClientError, ValueError):
                    logger.exception("Evaluation trace authorization unavailable")
                    return None
    return _evaluation_trace_token


def _evaluation_trace_authorized(payload: dict[str, Any]) -> bool:
    """Authorize and remove the privileged trace token from the request body."""
    presented = payload.pop("evaluation_token", None)
    if presented is None:
        return False
    expected = _get_evaluation_trace_token()
    if not expected or not isinstance(presented, str):
        return False
    return secrets.compare_digest(presented, expected)


def _scoped_dealer_profile_tool(gateway_client: Any, actor_id: str) -> Any:
    """Expose only the authenticated dealer's profile to the model.

    The raw Gateway schema includes a caller-selectable ``dealer_id`` path
    parameter and a list operation. Passing those tools directly to the model
    would let prompt injection cross dealer boundaries. This wrapper removes
    both capabilities and injects the trusted actor ID server-side.
    """
    gateway_tools = list(gateway_client.list_tools_sync())
    profile_tools = [
        candidate
        for candidate in gateway_tools
        if "getdealerprofile" in re.sub(r"[^a-z0-9]", "", str(candidate.tool_name).lower())
    ]
    if len(profile_tools) != 1:
        raise RuntimeError(
            f"Expected one getDealerProfile Gateway tool, found {len(profile_tools)}"
        )
    profile_tool = profile_tools[0]
    upstream_name = profile_tool.mcp_tool.name

    @tool(
        name="get_dealer_profile",
        description=(
            "Get the authenticated dealer's location, preferences, and buying history. "
            "The dealer identity is supplied by the runtime and cannot be overridden."
        ),
    )
    def get_dealer_profile() -> dict[str, Any]:
        return gateway_client.call_tool_sync(
            tool_use_id=f"dealer-profile-{uuid.uuid4().hex}",
            name=upstream_name,
            arguments={"dealer_id": actor_id},
        )

    return get_dealer_profile


def _lancedb_schema(vector_size: int) -> pa.Schema:
    """Return the stable schema used by the local LanceDB vehicle table."""
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("make", pa.string()),
            pa.field("model", pa.string()),
            pa.field("year", pa.int64()),
            pa.field("fuel_type", pa.string()),
            pa.field("body_type", pa.string()),
            pa.field("mileage", pa.float64()),
            pa.field("price", pa.float64()),
            pa.field("transmission", pa.string()),
            pa.field("condition", pa.string()),
            pa.field("seller_latitude", pa.float64()),
            pa.field("seller_longitude", pa.float64()),
            pa.field("seller_city", pa.string()),
            pa.field("seller_region", pa.string()),
            pa.field("contextualized_description", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), vector_size), nullable=False),
        ]
    )


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _validate_snapshot_records(
    vehicles: list[dict[str, Any]],
    *,
    expected_dimension: int | None = None,
) -> int:
    """Validate IDs and vectors before creating or replacing a LanceDB table."""
    if not vehicles:
        raise ValueError("Cannot materialize LanceDB without vehicle records")

    ids: set[str] = set()
    vector_size: int | None = None
    for vehicle in vehicles:
        raw_vehicle_id = vehicle.get("id")
        vehicle_id = str(raw_vehicle_id).strip() if raw_vehicle_id is not None else ""
        if not vehicle_id or vehicle_id in ids:
            raise ValueError(f"Vehicle IDs must be non-empty and unique: {vehicle_id!r}")
        ids.add(vehicle_id)

        embedding = vehicle.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Vehicle {vehicle_id} has no embedding")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in embedding
        ):
            raise ValueError(f"Vehicle {vehicle_id} has invalid embedding values")
        vector_size = vector_size or len(embedding)
        if len(embedding) != vector_size:
            raise ValueError(f"Vehicle {vehicle_id} has an invalid embedding dimension")

    if expected_dimension and vector_size != expected_dimension:
        raise ValueError(
            f"Embedding dimension {vector_size} does not match expected {expected_dimension}"
        )
    return vector_size or 0


def _materialize_lancedb(
    vehicles: list[dict[str, Any]],
    db_path: Path,
    *,
    expected_dimension: int | None = None,
) -> Any:
    """Create an actual local LanceDB table from S3-persisted vehicle records."""
    vector_size = _validate_snapshot_records(
        vehicles,
        expected_dimension=expected_dimension,
    )

    rows = []
    for vehicle in vehicles:
        embedding = vehicle.get("embedding")
        rows.append(
            {
                "id": str(vehicle["id"]).strip(),
                "make": str(vehicle["make"]) if vehicle.get("make") is not None else None,
                "model": str(vehicle["model"]) if vehicle.get("model") is not None else None,
                "year": _as_int(vehicle.get("year")),
                "fuel_type": (
                    str(vehicle["fuel_type"]) if vehicle.get("fuel_type") is not None else None
                ),
                "body_type": (
                    str(vehicle["body_type"]) if vehicle.get("body_type") is not None else None
                ),
                "mileage": _as_float(vehicle.get("mileage")),
                "price": _as_float(vehicle.get("price")),
                "transmission": (
                    str(vehicle["transmission"])
                    if vehicle.get("transmission") is not None
                    else None
                ),
                "condition": (
                    str(vehicle["condition"]) if vehicle.get("condition") is not None else None
                ),
                "seller_latitude": _as_float(vehicle.get("seller_latitude")),
                "seller_longitude": _as_float(vehicle.get("seller_longitude")),
                "seller_city": (
                    str(vehicle["seller_city"]) if vehicle.get("seller_city") is not None else None
                ),
                "seller_region": (
                    str(vehicle["seller_region"])
                    if vehicle.get("seller_region") is not None
                    else None
                ),
                "contextualized_description": (
                    str(vehicle["contextualized_description"])
                    if vehicle.get("contextualized_description") is not None
                    else None
                ),
                "embedding": [float(value) for value in embedding],
            }
        )

    db_path.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pylist(rows, schema=_lancedb_schema(vector_size))
    database = lancedb.connect(str(db_path))
    return database.create_table("vehicles", data=arrow_table, mode="overwrite")


def _validated_manifest(value: Any) -> dict[str, Any]:
    """Validate the untrusted S3 manifest before using its object key."""
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported LanceDB manifest schema")

    version = value.get("version")
    data_key = value.get("data_key")
    generated_at = value.get("generated_at")
    vehicle_count = value.get("vehicle_count")
    vector_dimension = value.get("vector_dimension")
    if not isinstance(version, str) or not _SNAPSHOT_VERSION_PATTERN.fullmatch(version):
        raise ValueError("Invalid LanceDB manifest version")
    if not isinstance(data_key, str) or not _SNAPSHOT_KEY_PATTERN.fullmatch(data_key):
        raise ValueError("Invalid LanceDB snapshot key")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise ValueError("Invalid LanceDB generation time")
    generated = datetime.fromisoformat(generated_at)
    if generated.tzinfo is None:
        raise ValueError("LanceDB generation time must include a timezone")
    if not isinstance(vehicle_count, int) or isinstance(vehicle_count, bool) or vehicle_count < 1:
        raise ValueError("Invalid LanceDB vehicle count")
    if (
        not isinstance(vector_dimension, int)
        or isinstance(vector_dimension, bool)
        or vector_dimension < 1
    ):
        raise ValueError("Invalid LanceDB vector dimension")
    if EXPECTED_EMBEDDING_DIMENSION > 0 and vector_dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"LanceDB vector dimension {vector_dimension} does not match "
            f"expected {EXPECTED_EMBEDDING_DIMENSION}"
        )
    return value


def _read_s3_json(s3: Any, key: str) -> tuple[Any, bytes]:
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    body = response["Body"].read()
    return json.loads(body.decode("utf-8")), body


def _load_manifest_snapshot(s3: Any, manifest: dict[str, Any]) -> Any:
    snapshot, body = _read_s3_json(s3, manifest["data_key"])
    if not secrets.compare_digest(hashlib.sha256(body).hexdigest(), manifest["version"]):
        raise ValueError("LanceDB snapshot hash does not match its manifest")
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported LanceDB snapshot schema")
    if (
        snapshot.get("generated_at") != manifest["generated_at"]
        or snapshot.get("vehicle_count") != manifest["vehicle_count"]
        or snapshot.get("vector_dimension") != manifest["vector_dimension"]
    ):
        raise ValueError("LanceDB snapshot metadata does not match its manifest")
    vehicles = snapshot.get("vehicles")
    if not isinstance(vehicles, list) or len(vehicles) != manifest["vehicle_count"]:
        raise ValueError("LanceDB snapshot vehicle count is inconsistent")

    db_path = LANCEDB_CACHE_ROOT / manifest["version"]
    return _materialize_lancedb(
        vehicles,
        db_path,
        expected_dimension=manifest["vector_dimension"],
    )


def _prune_lancedb_cache(current_version: str) -> None:
    """Bound ephemeral LanceDB generations while preserving recent readers."""
    if not LANCEDB_CACHE_ROOT.exists():
        return
    generations = [
        path
        for path in LANCEDB_CACHE_ROOT.iterdir()
        if path.is_dir() and _SNAPSHOT_VERSION_PATTERN.fullmatch(path.name)
    ]
    generations.sort(
        key=lambda path: (path.name == current_version, path.stat().st_mtime),
        reverse=True,
    )
    for stale_path in generations[LANCEDB_CACHE_GENERATIONS:]:
        shutil.rmtree(stale_path, ignore_errors=True)


def _refresh_lancedb_locked() -> None:
    # The snapshot cache is process-wide by design: AgentCore reuses a warm
    # container across invocations, and every write here holds _lancedb_lock.
    global _lancedb, _lancedb_frame, _lancedb_frame_version  # noqa: PLW0603
    global _lancedb_generated_at, _lancedb_version  # noqa: PLW0603

    s3 = boto3.client("s3", region_name=AWS_REGION)
    manifest_value, _ = _read_s3_json(s3, LANCEDB_KEY)
    manifest = _validated_manifest(manifest_value)
    if _lancedb is not None and manifest["version"] == _lancedb_version:
        _lancedb_generated_at = manifest["generated_at"]
        return

    candidate = _load_manifest_snapshot(s3, manifest)
    _lancedb = candidate
    _lancedb_version = manifest["version"]
    _lancedb_generated_at = manifest["generated_at"]
    _lancedb_frame = None
    _lancedb_frame_version = None
    _prune_lancedb_cache(manifest["version"])
    logger.info(
        "Activated LanceDB version %s with %s vehicles",
        manifest["version"][:12],
        candidate.count_rows(),
    )


def _get_lancedb() -> Any:
    global _lancedb_last_refresh_check  # noqa: PLW0603 - warm-start cache, see above
    now = time.monotonic()
    refresh_due = (
        _lancedb is None or now - _lancedb_last_refresh_check >= LANCEDB_REFRESH_INTERVAL_SECONDS
    )
    if not refresh_due:
        return _lancedb

    with _lancedb_lock:
        now = time.monotonic()
        refresh_due = (
            _lancedb is None
            or now - _lancedb_last_refresh_check >= LANCEDB_REFRESH_INTERVAL_SECONDS
        )
        if refresh_due:
            _lancedb_last_refresh_check = now
            try:
                _refresh_lancedb_locked()
            except (
                botocore.exceptions.BotoCoreError,
                json.JSONDecodeError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                if _lancedb is None:
                    raise RuntimeError(
                        f"Cannot load vehicle data from s3://{S3_BUCKET}/{LANCEDB_KEY}"
                    ) from exc
                logger.warning(
                    "LanceDB refresh failed; retaining version %s: %s",
                    (_lancedb_version or "unknown")[:12],
                    type(exc).__name__,
                )
    return _lancedb


def _get_vehicle_frame() -> pd.DataFrame:
    """Return a cached DataFrame view for deterministic structured tools."""
    global _lancedb_frame, _lancedb_frame_version  # noqa: PLW0603 - warm-start cache
    table = _get_lancedb()
    # Tests and local callers may inject a DataFrame directly.
    if isinstance(table, pd.DataFrame):
        return table
    if _lancedb_frame is not None and _lancedb_frame_version == _lancedb_version:
        return _lancedb_frame
    frame = table.search().limit(table.count_rows()).to_pandas()
    with _lancedb_lock:
        if table is _lancedb:
            _lancedb_frame = frame
            _lancedb_frame_version = _lancedb_version
    return frame


def _bounded_limit(value: Any, *, default: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = default
    return min(MAX_TOOL_RESULTS, max(1, limit))


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return repr(value)
    raise ValueError("Unsupported query literal")


def _compile_lance_node(node: ast.AST) -> str:  # noqa: PLR0911, PLR0912 - one arm per AST node kind
    if isinstance(node, ast.Name) and node.id in _VALID_COLUMNS:
        return node.id
    if isinstance(node, ast.Constant):
        return _sql_literal(node.value)
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        return "(" + ", ".join(_compile_lance_node(item) for item in node.elts) + ")"
    if isinstance(node, ast.BoolOp):
        # ast.BoolOp.op is And or Or by grammar, so no third case can occur.
        operator = "AND" if isinstance(node.op, ast.And) else "OR"
        return "(" + f" {operator} ".join(_compile_lance_node(item) for item in node.values) + ")"
    if isinstance(node, ast.BinOp):
        operators = {
            ast.BitAnd: "AND",
            ast.BitOr: "OR",
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        operator = next(
            (text for kind, text in operators.items() if isinstance(node.op, kind)), None
        )
        if operator is None:
            raise ValueError("Unsupported binary operator")
        return f"({_compile_lance_node(node.left)} {operator} {_compile_lance_node(node.right)})"
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return f"(NOT {_compile_lance_node(node.operand)})"
        if isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
            sign = "+" if isinstance(node.op, ast.UAdd) else "-"
            return sign + _compile_lance_node(node.operand)
        raise ValueError("Unsupported unary operator")
    if isinstance(node, ast.Compare):
        operator_map = {
            ast.Eq: "=",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.In: "IN",
            ast.NotIn: "NOT IN",
        }
        comparisons = []
        left = node.left
        for operator_node, right in zip(node.ops, node.comparators, strict=True):
            operator = next(
                (text for kind, text in operator_map.items() if isinstance(operator_node, kind)),
                None,
            )
            if operator is None:
                raise ValueError("Unsupported comparison operator")
            left_sql = _compile_lance_node(left)
            right_sql = _compile_lance_node(right)
            if isinstance(right, ast.Constant) and right.value is None:
                if operator == "=":
                    comparisons.append(f"{left_sql} IS NULL")
                elif operator == "!=":
                    comparisons.append(f"{left_sql} IS NOT NULL")
                else:
                    raise ValueError("NULL only supports equality comparisons")
            else:
                comparisons.append(f"{left_sql} {operator} {right_sql}")
            left = right
        return "(" + " AND ".join(comparisons) + ")"
    raise ValueError("Unsupported query expression")


def _compile_lance_filter(where_clause: str) -> str:
    """Parse the safe pandas subset and emit an equivalent Lance SQL filter."""
    normalized = unicodedata.normalize("NFKC", where_clause)
    if not _is_safe_query(normalized):
        raise ValueError("Unsafe query")
    try:
        expression = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid query syntax") from exc
    return _compile_lance_node(expression.body)


def _structured_lance_filter(**filters: Any) -> str | None:
    clauses: list[str] = []
    for column in ("make", "model", "fuel_type", "transmission", "body_type"):
        value = filters.get(column)
        if value:
            clauses.append(f"lower({column}) = {_sql_literal(str(value).lower())}")
    comparisons = (
        ("min_price", "price", ">="),
        ("max_price", "price", "<="),
        ("min_year", "year", ">="),
        ("max_year", "year", "<="),
        ("max_mileage", "mileage", "<="),
    )
    for argument, column, operator in comparisons:
        value = filters.get(argument)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            clauses.append(f"{column} {operator} {_sql_literal(value)}")
    return " AND ".join(clauses) or None


def _generate_embedding(text: str) -> list[float]:
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        resp = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": text}),
        )
        return json.loads(resp["body"].read()).get("embedding", [])
    except (boto3.exceptions.Boto3Error, json.JSONDecodeError, KeyError):
        logger.exception("Embedding failed")
        return []


@tool
def get_schema() -> str:
    """Get the vehicle data schema including column names and types."""
    table = _get_lancedb()
    if isinstance(table, pd.DataFrame):
        schema = {col: str(table[col].dtype) for col in table.columns if not col.startswith("_")}
        count = len(table)
    else:
        schema = {
            field.name: str(field.type)
            for field in table.schema
            if not field.name.startswith("_") and field.name != "embedding"
        }
        count = table.count_rows()
    return f"Vehicle schema ({count} vehicles): {schema}"


def _apply_filters(  # noqa: PLR0913 - mirrors the search_vehicles tool schema
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
def search_vehicles(  # noqa: PLR0913 - the flat signature is the tool schema the model sees
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
    limit: int = 10,
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
        limit: Max returned results (default 10, hard-capped by the runtime)
    """
    limit = _bounded_limit(limit, default=10)
    table = _get_lancedb()
    filters = {
        "make": make,
        "model": model,
        "min_price": min_price,
        "max_price": max_price,
        "min_year": min_year,
        "max_year": max_year,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "body_type": body_type,
        "max_mileage": max_mileage,
    }
    if isinstance(table, pd.DataFrame):
        result = _apply_filters(table, **filters)
        total_count = len(result)
        vehicles = (
            result.head(limit)
            .drop(columns=["embedding", "_original"], errors="ignore")
            .to_dict(orient="records")
        )
    else:
        query = table.search()
        lance_filter = _structured_lance_filter(**filters)
        if lance_filter:
            query = query.where(lance_filter)
        total_count = table.count_rows(lance_filter)
        vehicles = query.select(list(_PUBLIC_VEHICLE_COLUMNS)).limit(limit).to_list()
    return {"count": total_count, "returned": len(vehicles), "vehicles": vehicles}


@tool
def run_sql(where_clause: str, limit: int = 10) -> dict[str, Any]:
    """Query vehicles with a pandas WHERE clause for complex queries not covered by search_vehicles.

    Prefer search_vehicles for standard filters. Use this only for compound conditions,
    range comparisons, or complex logic.

    Args:
        where_clause: Pandas query expression e.g. "make == 'BMW' and price < 25000"
        limit: Max returned results (default 10, hard-capped by the runtime)
    """
    limit = _bounded_limit(limit, default=10)
    # Validate where_clause against injection: only allow safe pandas query expressions
    if not _is_safe_query(where_clause):
        return {
            "error": "Invalid query",
            "hint": "Use pandas query syntax: make == 'BMW' and price < 25000",
        }
    try:
        table = _get_lancedb()
        if isinstance(table, pd.DataFrame):
            result = table.query(where_clause)
            total_count = len(result)
            vehicles = (
                result.head(limit)
                .drop(columns=["embedding", "_original"], errors="ignore")
                .to_dict(orient="records")
            )
        else:
            lance_filter = _compile_lance_filter(where_clause)
            total_count = table.count_rows(lance_filter)
            vehicles = (
                table.search()
                .where(lance_filter)
                .select(list(_PUBLIC_VEHICLE_COLUMNS))
                .limit(limit)
                .to_list()
            )
        return {"count": total_count, "returned": len(vehicles), "vehicles": vehicles}
    except (ValueError, KeyError, SyntaxError, NameError) as exc:
        logger.warning("Query validation failed: %s", type(exc).__name__)
        return {
            "error": "Invalid query syntax",
            "hint": "Use pandas query syntax: make == 'BMW' and price < 25000",
        }


@tool
def hybrid_search(  # noqa: PLR0911 - one early return per rejected input shape
    query_text: str, where_clause: str = "", limit: int = 10
) -> dict[str, Any]:
    """Semantic search for natural language queries, optionally combined with filters.

    Args:
        query_text: Natural language query like "sporty family car"
        where_clause: Optional pandas query filter
        limit: Max results (default 10)
    """
    limit = _bounded_limit(limit, default=10)
    table = _get_lancedb()
    embedding = _generate_embedding(query_text)
    if not embedding:
        return {"error": "Failed to generate embedding"}

    if isinstance(table, pd.DataFrame):
        # Lightweight fallback for tests that inject a DataFrame.
        filtered = table
        if where_clause:
            if not _is_safe_query(where_clause):
                logger.warning("Rejected unsafe filter query")
                return {"error": "Unsafe filter query rejected"}
            try:
                filtered = table.query(where_clause)
            except (ValueError, KeyError, SyntaxError) as exc:
                logger.warning("Filter query failed: %s", type(exc).__name__)
                return {"error": "Invalid filter query"}
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
        results.sort(key=lambda vehicle: vehicle["similarity"], reverse=True)
        return {"count": min(len(results), limit), "vehicles": results[:limit]}

    query = table.search(embedding, vector_column_name="embedding").distance_type("cosine")
    if where_clause:
        try:
            query = query.where(_compile_lance_filter(where_clause), prefilter=True)
        except ValueError:
            logger.warning("Rejected invalid hybrid-search filter")
            return {"error": "Invalid filter query"}
    hits = query.select([*_PUBLIC_VEHICLE_COLUMNS, "_distance"]).limit(limit).to_pandas()
    hits["similarity"] = (1.0 - hits["_distance"]).round(4)
    vehicles = hits.drop(columns=["embedding", "_distance", "_original"], errors="ignore").to_dict(
        orient="records"
    )
    return {"count": len(vehicles), "vehicles": vehicles}


@tool
def get_embedding(text: str) -> list[float]:
    """Convert text to a 1024-dimension embedding vector.

    Args:
        text: Text to embed
    """
    return _generate_embedding(text)


@tool
def filter_by_distance(
    latitude: float,
    longitude: float,
    max_distance_miles: float = 50.0,
    limit: int = 10,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Filter vehicles by distance in miles from a lat/long point.

    Args:
        latitude: Reference latitude (from dealer profile or user)
        longitude: Reference longitude
        max_distance_miles: Radius in miles (default 50)
        limit: Max returned results (default 10, hard-capped by the runtime)
        candidate_ids: Optional vehicle IDs from a preceding inventory search.
            When supplied, distance filtering is restricted to those candidates.
    """
    limit = _bounded_limit(limit, default=10)
    df = _get_vehicle_frame()
    if candidate_ids is not None:
        if len(candidate_ids) > MAX_TOOL_RESULTS or any(
            not isinstance(vehicle_id, str) or not _VEHICLE_ID_PATTERN.fullmatch(vehicle_id)
            for vehicle_id in candidate_ids
        ):
            return {"error": "Invalid candidate vehicle IDs"}
        df = df[df["id"].astype(str).isin(set(candidate_ids))]

    lat_delta, lon_delta = bounding_box_miles(latitude, max_distance_miles)

    # Convert coordinates to numeric, coercing invalid values to NaN
    lats = pd.to_numeric(df["seller_latitude"], errors="coerce")
    lons = pd.to_numeric(df["seller_longitude"], errors="coerce")

    # Filter: valid coordinates within bounding box
    valid = lats.notna() & lons.notna()
    bbox = valid & (np.abs(lats - latitude) <= lat_delta) & (np.abs(lons - longitude) <= lon_delta)
    bbox_df = df[bbox]

    if bbox_df.empty:
        return {"count": 0, "returned": 0, "vehicles": []}

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

    total_count = len(result_df)
    result_df = result_df.sort_values("distance_miles").head(limit)
    vehicles = result_df.drop(columns=["embedding", "_original"], errors="ignore").to_dict(
        orient="records"
    )
    return {"count": total_count, "returned": len(vehicles), "vehicles": vehicles}


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
    except (boto3.exceptions.Boto3Error, json.JSONDecodeError, KeyError):
        logger.exception("get_bids failed")
        return {"error": "Unable to fetch bids data"}


# NOTE: the dealer-profile lookup is no longer a local @tool. Dealer profiles are
# served through the Amazon Bedrock AgentCore Gateway (which fronts the Dealer
# API) and reach the agent as an MCP tool. See utils/gateway.py and the
# per-invoke wiring in ``invoke`` below.

SYSTEM_PROMPT = """You are an AI assistant for car auction dealers. You help them search and analyze vehicle inventory.

The runtime exposes only the tools relevant to the current request. Use only tools listed as
exposed for this request.

## Tool execution policy
Use only filters stated in the current request or preserved from the active conversation.
Never add dealer preferences, remembered constraints, location, budget, year, mileage, fuel,
transmission, or body type unless the user explicitly asks to apply preferences or those filters.
The current request overrides any conflicting remembered context.

Choose exactly one primary inventory search strategy per request:
- Structured criteria (make, model, price, year, mileage, fuel, transmission, body): search_vehicles.
- Vague or semantic criteria ("something sporty for a family"): hybrid_search, with only explicit
  structured constraints in where_clause.
- Complex compound logic that search_vehicles cannot express: run_sql.
Do not call get_embedding before hybrid_search. Do not call multiple primary search tools, repeat a
search with broader filters, or search for unsolicited alternatives. If a search returns zero,
report zero with the active filters and ask before broadening it.

For every request that refers to "my dealership", "near me", or a distance from the dealer and does
not provide coordinates, you MUST first call get_dealer_profile. Wait for that result, then call
filter_by_distance with the returned coordinates. Never infer dealer coordinates from memory and
never issue get_dealer_profile and filter_by_distance in the same parallel tool batch. When the
request also has structured vehicle criteria, the required order is get_dealer_profile,
search_vehicles, then filter_by_distance. Pass the IDs returned by search_vehicles as
filter_by_distance candidate_ids so the distance tool checks only matching vehicles.

For a follow-up that narrows or changes a previous search, call search_vehicles again with the
complete active filter set: preserve every unchanged filter, apply the new filter, and replace only
filters the user explicitly changes. Do not filter a previous result set only in prose, and do not
say that a new search is unnecessary. Do not replay a superseded body type or any other replaced
filter.

## Response format
- Present vehicle results in a clear, structured format with key details (make, model, year, price, mileage, fuel type).
- State the total match count, but show at most 5 vehicles as one compact bullet per vehicle unless
  the user asks for more. Do not use Markdown tables, decorative separators, or emoji. Keep the
  answer under 250 words and do not repeat the raw tool payload.
- When returning multiple vehicles, summarize how many matched and highlight the top options.
- For vague queries, ask a clarifying question AND show initial results based on reasonable assumptions.
- If no vehicles match, explicitly state the active filters and that the search returned zero results before suggesting alternatives.

## Safety guardrails — STRICT
You are a SEARCH and DISCOVERY assistant ONLY. You MUST refuse the following:
- NEVER place or submit bids on behalf of the user. You do not have bidding capability.
- NEVER predict, estimate, or comment on winning probabilities, chances of winning, or bid outcomes.
- NEVER use phrases like "chances of winning", "likely to win", "winning probability", or "guaranteed to win".
- If asked about bidding or winning odds, respond: "I'm a vehicle search assistant. I can help you find and compare vehicles, but I cannot place bids or predict auction outcomes. For bidding, please use the dealer portal directly."
"""

# Apply the Bedrock Guardrail only when both id and version are configured, so
# local/test runs without a provisioned guardrail behave unchanged. The model is
# shared across invocations (stateless); the Agent itself is built per-invoke so
# each request binds its own AgentCore Memory session and Gateway tool set.
_model_kwargs: dict[str, Any] = {"model_id": MODEL_ID, "region_name": AWS_REGION}
if GUARDRAIL_ID and GUARDRAIL_VERSION:
    _model_kwargs.update(
        guardrail_id=GUARDRAIL_ID,
        guardrail_version=GUARDRAIL_VERSION,
        guardrail_trace="enabled",
    )
model = BedrockModel(
    **_model_kwargs,
    boto_client_config=BotocoreConfig(
        connect_timeout=5,
        read_timeout=120,
        retries={"max_attempts": 3, "mode": "standard"},
    ),
)

# Local (in-container) tools. The dealer-profile tool is not here — it is served
# through the AgentCore Gateway and attached per-invoke as an MCP tool.
_LOCAL_TOOLS = [
    get_schema,
    search_vehicles,
    run_sql,
    hybrid_search,
    get_embedding,
    filter_by_distance,
    get_bids,
]

_LOCATION_MARKERS = ("dealership", "near me", "nearby", "distance", "within ")
_SEMANTIC_MARKERS = (
    "something ",
    "sporty",
    "family",
    "similar to",
    "like a ",
    "good for",
)
_COMPLEX_QUERY_MARKERS = (" either ", " or ", " except ", " between ")
_LONG_TERM_MEMORY_MARKERS = (
    "my preference",
    "what i like",
    "what do i like",
    "remember",
    "based on my",
    "buying history",
    "usually buy",
)


def _tool_name(tool_value: Any) -> str:
    return str(getattr(tool_value, "tool_name", getattr(tool_value, "__name__", "")))


def _needs_dealer_profile(message: str) -> bool:
    normalized = message.casefold()
    return any(marker in normalized for marker in (*_LOCATION_MARKERS, "my preferences", "profile"))


def _needs_long_term_memory(message: str) -> bool:
    """Retrieve cross-session memory only when the user explicitly requests it."""
    normalized = message.casefold()
    return any(marker in normalized for marker in _LONG_TERM_MEMORY_MARKERS)


def _select_local_tools(message: str) -> list[Any]:
    """Narrow tool schemas to the capabilities relevant to this request."""
    normalized = f" {message.casefold()} "
    if any(marker in normalized for marker in ("schema", "columns", "data fields")):
        return [get_schema]
    if any(marker in normalized for marker in ("embedding", "vector representation")):
        return [get_embedding]
    if any(marker in normalized for marker in ("bid count", "number of bids")):
        return [get_bids]

    tools: list[Any]
    if any(marker in normalized for marker in _SEMANTIC_MARKERS):
        tools = [hybrid_search]
    else:
        tools = [search_vehicles]
        if any(marker in normalized for marker in _COMPLEX_QUERY_MARKERS):
            tools.append(run_sql)
    if any(marker in normalized for marker in _LOCATION_MARKERS):
        tools.append(filter_by_distance)
    return tools


def _build_memory_session_manager(
    actor_id: str,
    session_id: str,
    *,
    retrieve_long_term: bool = False,
) -> AgentCoreMemorySessionManager | None:
    """Build an AgentCore Memory session manager for this dealer + session.

    Returns ``None`` when no memory is provisioned (``MEMORY_ID`` unset), in
    which case the agent runs with in-process conversation only. Short-term
    session history is always restored. Cross-session preference and fact
    retrieval is enabled only for explicit memory requests so prior search
    filters cannot silently affect an unrelated request.
    """
    if not MEMORY_ID:
        return None
    config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config=_MEMORY_NAMESPACES if retrieve_long_term else None,
    )
    return AgentCoreMemorySessionManager(config, region_name=AWS_REGION)


def _build_agent(
    tools: list[Any],
    session_manager: AgentCoreMemorySessionManager | None,
) -> Agent:
    """Construct the Strands agent for a single invocation.

    ``tools`` combines the local in-container tools with any tools discovered
    on the AgentCore Gateway. ``session_manager`` binds AgentCore Memory when
    provisioned; when ``None`` the agent keeps only in-process conversation.
    """
    exposed_tool_names = ", ".join(_tool_name(tool_value) for tool_value in tools)
    request_prompt = (
        f"{SYSTEM_PROMPT}\n\nExposed tools for this request: {exposed_tool_names or 'none'}."
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "system_prompt": request_prompt,
        "tools": tools,
        # AgentCore captures stdout/stderr. Suppress Strands' default streaming
        # callback so prompts, tool payloads, and model responses are not logged.
        "callback_handler": None,
    }
    if session_manager is not None:
        kwargs["session_manager"] = session_manager
    return Agent(**kwargs)


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


def _current_invocation_usage(metrics: Any) -> dict[str, Any]:
    """Return token usage for this agent invocation, not prior calls."""
    latest = getattr(metrics, "latest_agent_invocation", None)
    usage = getattr(latest, "usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    return dict(getattr(metrics, "accumulated_usage", {}) or {})


@app.entrypoint
def invoke(payload: dict[str, Any], context: RequestContext) -> dict[str, Any]:
    """Amazon Bedrock AgentCore Runtime entrypoint.

    ``context`` is injected by Amazon Bedrock AgentCore Runtime when the handler's second
    parameter is named ``context``. ``context.session_id`` is the authoritative
    ``runtimeSessionId`` for this microVM; prefer it over any payload-supplied
    value, which is caller-controlled.
    """
    include_evaluation_trace = _evaluation_trace_authorized(payload)
    if include_evaluation_trace:
        # Freshness is part of the privileged evaluation contract even when the
        # prompt does not call a vehicle-data tool.
        _get_lancedb()
    user_message = payload.get("prompt", "Hello!")
    user_message, _truncated = _cap_prompt(str(user_message))
    if _truncated:
        logger.warning("Prompt exceeded %d chars; truncated", MAX_PROMPT_CHARS)
    request_tools = _select_local_tools(user_message)
    needs_dealer_profile = _needs_dealer_profile(user_message)
    needs_long_term_memory = _needs_long_term_memory(user_message)
    actor_id = _resolve_actor_id(payload, context)
    session_id = context.session_id or payload.get("session_id", "default-session")
    logger.info("Processing authenticated agent request")

    # Inject dealer context so the agent can call the dealer-profile tool.
    # _resolve_actor_id has validated the trusted identity as an opaque token, so
    # it cannot inject prompt instructions through this context line.
    if actor_id != "default":
        user_message = f"[Dealer context: dealer_id={actor_id}]\n\n{user_message}"

    # Always preserve same-session continuity. Retrieve cross-session facts and
    # preferences only when the current user explicitly asks for remembered context.
    session_manager = _build_memory_session_manager(
        actor_id,
        session_id,
        retrieve_long_term=needs_long_term_memory,
    )

    # Attach the dealer-profile tool from the AgentCore Gateway. The MCP client
    # is scoped to this request so its connection lifecycle matches the invoke.
    if GATEWAY_URL and needs_dealer_profile:
        gateway_client = build_gateway_mcp_client(GATEWAY_URL, AWS_REGION)
        with gateway_client:
            tools = [*request_tools, _scoped_dealer_profile_tool(gateway_client, actor_id)]
            agent = _build_agent(tools, session_manager)
            history_start = len(agent.messages)
            response = agent(user_message)
            if include_evaluation_trace:
                trajectory = _extract_trajectory(agent.messages[history_start:])
                tool_names = list(agent.tool_names)
    else:
        agent = _build_agent(request_tools, session_manager)
        history_start = len(agent.messages)
        response = agent(user_message)
        if include_evaluation_trace:
            trajectory = _extract_trajectory(agent.messages[history_start:])
            tool_names = list(agent.tool_names)

    result = {
        "result": response.message,
        "model": MODEL_ID,
        "session_id": session_id,
    }
    if include_evaluation_trace:
        # Tool arguments/results are privileged evaluation data. Never expose
        # them on the normal application response path.
        usage = _current_invocation_usage(response.metrics)
        result["trajectory"] = trajectory
        result["available_tools"] = tool_names
        result["usage"] = {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        }
        if _lancedb_generated_at:
            result["last_refresh_time"] = _lancedb_generated_at
            result["lancedb_version"] = _lancedb_version
    return result


if __name__ == "__main__":
    app.run()
