# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Simple loader for dealer data - converts floats to Decimals for Amazon DynamoDB."""

import argparse
import json
import os
import sys
from decimal import Decimal

import boto3


def decimal_converter(obj):
    """Convert floats to Decimals for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: decimal_converter(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_converter(item) for item in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Load dealer data into DynamoDB")
    parser.add_argument(
        "--file",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "examples",
            "vehicle-auction-agent",
            "lambda",
            "functions",
            "data_ingestion",
            "sample_dealerships.json",
        ),
        help="Path to the dealerships JSON file",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "eu-west-1"),
        help="AWS region (default: eu-west-1 or AWS_REGION env var)",
    )
    parser.add_argument(
        "--table",
        default=os.environ.get("DEALERS_TABLE", "agent-eval-dealers-dev"),
        help="DynamoDB table name (default: agent-eval-dealers-dev or DEALERS_TABLE env var)",
    )
    args = parser.parse_args()

    try:
        print(f"Loading dealer data from {args.file}...")
        with open(args.file, "r") as f:
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
        dynamodb = boto3.resource("dynamodb", region_name=args.region)
        table = dynamodb.Table(args.table)

        print(f"Loading into DynamoDB table '{args.table}' in {args.region}...")
        for dealer in dealers:
            dealer = decimal_converter(dealer)
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
