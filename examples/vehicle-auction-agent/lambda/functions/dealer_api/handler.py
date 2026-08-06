# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Dealer API AWS Lambda Handler.

Provides REST API for dealer data stored in DynamoDB.
"""

import json
import logging
import os
from decimal import Decimal
from typing import Any

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Custom JSON encoder for DynamoDB Decimal types
class DecimalEncoder(json.JSONEncoder):
    """Convert Decimal to int or float for JSON serialization."""

    def default(self, o: Any) -> Any:
        """Serialize a Decimal as an int when whole, otherwise as a float.

        Args:
            o: The object json cannot serialize on its own.

        Returns:
            A JSON-serializable equivalent.
        """
        if isinstance(o, Decimal):
            # Convert to int if no decimal places, otherwise float
            return int(o) if o % 1 == 0 else float(o)
        return super().default(o)


# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb")
table_name = os.environ.get("DEALERS_TABLE")
if not table_name:
    raise RuntimeError("DEALERS_TABLE environment variable required")
dealers_table = dynamodb.Table(table_name)

# CORS origin from environment (CDK sets per-environment value)
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "")

# Fields returned by the list endpoint — keeps the summary view free of internal/PII attributes.
# Single-item GET /dealers/{id} still returns the full item intentionally.
LIST_PROJECTION_FIELDS = ("dealer_id", "name", "location", "preferences")

# Bound the list scan so a caller can never force an unbounded full-table walk
# (DynamoDB scan exhaustion → Lambda timeout). SCAN_PAGE_SIZE caps items per
# page; SCAN_MAX_ITEMS caps the total returned across pages.
SCAN_PAGE_SIZE = int(os.environ.get("DEALERS_SCAN_PAGE_SIZE", "100"))
SCAN_MAX_ITEMS = int(os.environ.get("DEALERS_SCAN_MAX_ITEMS", "500"))


class DealerNotFoundError(Exception):
    """Raised when a requested dealer_id has no item in the table."""


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001 - Lambda passes context positionally
    """Handle dealer API requests.

    Endpoints:
    - GET /dealers - List all dealers
    - GET /dealers/{dealer_id} - Get specific dealer

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        API Gateway response
    """
    logger.info("Event: method=%s, path=%s", event.get("httpMethod"), event.get("path"))

    http_method = event.get("httpMethod", "")
    path_parameters = event.get("pathParameters") or {}

    try:
        if http_method == "GET":
            if path_parameters.get("dealer_id"):
                # Single dealer lookup by path parameter.
                result = get_dealer(path_parameters["dealer_id"])
            else:
                # Bounded summary listing.
                result = list_dealers()

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
                },
                "body": json.dumps(result, cls=DecimalEncoder),
            }

        return {
            "statusCode": 405,
            "headers": {"Access-Control-Allow-Origin": ALLOWED_ORIGIN},
            "body": json.dumps({"error": "Method not allowed"}, cls=DecimalEncoder),
        }

    except DealerNotFoundError as e:
        return {
            "statusCode": 404,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            },
            "body": json.dumps({"error": str(e)}, cls=DecimalEncoder),
        }

    except Exception as e:
        logger.exception("Unhandled error: %s", type(e).__name__)

        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": ALLOWED_ORIGIN},
            "body": json.dumps({"error": "Internal server error"}, cls=DecimalEncoder),
        }


def get_dealer(dealer_id: str) -> dict[str, Any]:
    """Get a specific dealer by ID.

    Args:
        dealer_id: Dealer ID

    Returns:
        Dealer data or error
    """
    logger.info("Getting dealer: %s", dealer_id)

    response = dealers_table.get_item(Key={"dealer_id": dealer_id})

    if "Item" not in response:
        logger.info("Dealer not found: %s", dealer_id)
        raise DealerNotFoundError("Dealer not found")

    return response["Item"]


def list_dealers() -> dict[str, Any]:
    """List dealers using a bounded DynamoDB scan.

    The scan is capped at SCAN_MAX_ITEMS across pages (SCAN_PAGE_SIZE per page)
    so a caller cannot force an unbounded full-table walk that exhausts the
    Lambda timeout. ``truncated`` indicates the cap was hit.

    Returns:
        List of dealers
    """
    logger.info("Listing dealers")

    items: list[dict[str, Any]] = []
    scan_kwargs: dict[str, Any] = {
        # Project only the safe summary fields on the DynamoDB side to avoid
        # reading unneeded attributes over the wire.  "name" and "location" are
        # reserved words in DynamoDB expression syntax, so they must be aliased.
        "ProjectionExpression": "dealer_id, #nm, #loc, preferences",
        "ExpressionAttributeNames": {"#nm": "name", "#loc": "location"},
        "Limit": SCAN_PAGE_SIZE,
    }
    truncated = False
    while True:
        response = dealers_table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        if len(items) >= SCAN_MAX_ITEMS:
            items = items[:SCAN_MAX_ITEMS]
            truncated = True
            break
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    if truncated:
        logger.warning("Dealer list scan hit cap of %d items; response truncated", SCAN_MAX_ITEMS)

    summary = [{k: item[k] for k in LIST_PROJECTION_FIELDS if k in item} for item in items]
    return {
        "dealers": summary,
        "count": len(summary),
        "truncated": truncated,
    }
