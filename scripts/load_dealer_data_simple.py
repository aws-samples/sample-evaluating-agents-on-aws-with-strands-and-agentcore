# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Simple loader for dealer data - converts floats to Decimals for Amazon DynamoDB."""

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

try:
    from scripts.aws_safety import confirm_mutation, reverify_identity, verified_session
except ModuleNotFoundError:
    from aws_safety import confirm_mutation, reverify_identity, verified_session

_SAMPLE_DEALERS = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "vehicle-auction-agent"
    / "lambda"
    / "functions"
    / "data_ingestion"
    / "sample_dealerships.json"
)


def main() -> None:
    """Load dealer records from a JSON file into the DynamoDB dealers table."""
    parser = argparse.ArgumentParser(description="Load dealer data into DynamoDB")
    parser.add_argument(
        "--file",
        default=str(_SAMPLE_DEALERS),
        help="Path to the dealerships JSON file",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Explicit AWS region",
    )
    parser.add_argument("--profile", required=True, help="Explicit AWS CLI profile")
    parser.add_argument("--expected-account", required=True)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--table",
        default=os.environ.get("DEALERS_TABLE", "agent-eval-dealers-dev"),
        help="DynamoDB table name (default: agent-eval-dealers-dev or DEALERS_TABLE env var)",
    )
    args = parser.parse_args()

    try:
        print(f"Loading dealer data from {args.file}...")
        # parse_float=Decimal converts every JSON float on the way in, which is
        # what DynamoDB requires; no second conversion pass is needed.
        with Path(args.file).open() as f:
            data = json.load(f, parse_float=Decimal)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    dealers = data.get("dealerships", [])
    print(f"Found {len(dealers)} dealers")

    try:
        session, identity = verified_session(
            profile=args.profile,
            region=args.region,
            expected_account=args.expected_account,
        )
        confirm_mutation(
            action="load-dealer-data",
            account=identity["Account"],
            region=args.region,
            cost="DynamoDB on-demand write request charges only; no new fixed resource cost",
            approved=args.yes,
        )
        reverify_identity(
            session,
            profile=args.profile,
            region=args.region,
            expected_account=args.expected_account,
        )
        dynamodb = session.resource("dynamodb", region_name=args.region)
        table = dynamodb.Table(args.table)

        print(f"Loading into DynamoDB table '{args.table}' in {args.region}...")
        for dealer in dealers:
            if "dealer_id" in dealer:
                dealer["dealer_id"] = str(dealer["dealer_id"])
            table.put_item(Item=dealer)
            print(f"  Loaded: {dealer['dealer_id']}")
    except Exception as e:
        print(f"Error loading data into DynamoDB: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded {len(dealers)} dealers successfully")


if __name__ == "__main__":
    main()
