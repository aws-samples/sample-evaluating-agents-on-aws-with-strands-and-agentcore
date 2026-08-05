#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Seed DynamoDB dealers table with mock data for evaluation testing.

Usage:
    python scripts/seed_dealer_data.py [--table agent-eval-dealers-dev] [--region eu-west-1]
"""

import argparse
from decimal import Decimal

import boto3

try:
    from scripts.aws_safety import confirm_mutation, reverify_identity, verified_session
except ModuleNotFoundError:
    from aws_safety import confirm_mutation, reverify_identity, verified_session


DEALERS = [
    {
        "dealer_id": "DLR24946",
        "name": "Example Motors London",
        "location": {
            "city": "London",
            "region": "Greater London",
            "latitude": Decimal("51.5074"),
            "longitude": Decimal("-0.1278"),
        },
        "preferences": {
            "fuel_types": ["diesel", "petrol", "hybrid"],
            "body_types": ["SUV", "saloon", "estate"],
            "price_range": {"min": Decimal("5000"), "max": Decimal("45000")},
            "max_mileage": 80000,
            "min_year": 2018,
        },
        "buying_history": {
            "total_purchases": 287,
            "avg_price": Decimal("22450.75"),
            "favorite_makes": ["BMW", "Audi", "Mercedes-Benz"],
        },
        "created_at": "2024-06-15T10:30:00",
        "updated_at": "2026-03-20T14:00:00",
    },
    {
        "dealer_id": "DLR10010",
        "name": "AnyCompany Cars",
        "location": {
            "city": "Leicester",
            "region": "East Midlands",
            "latitude": Decimal("52.6369"),
            "longitude": Decimal("-1.1398"),
        },
        "preferences": {
            "fuel_types": ["petrol", "diesel"],
            "body_types": ["coupe", "saloon", "pickup"],
            "price_range": {"min": Decimal("24512"), "max": Decimal("58935")},
            "max_mileage": 150000,
            "min_year": 2022,
        },
        "buying_history": {
            "total_purchases": 443,
            "avg_price": Decimal("32497.43"),
            "favorite_makes": ["Nissan"],
        },
        "created_at": "2024-12-11T01:08:35",
        "updated_at": "2025-12-06T01:08:35",
    },
    {
        "dealer_id": "DLR55021",
        "name": "Example Auto Group",
        "location": {
            "city": "Manchester",
            "region": "North West",
            "latitude": Decimal("53.4808"),
            "longitude": Decimal("-2.2426"),
        },
        "preferences": {
            "fuel_types": ["diesel", "electric"],
            "body_types": ["SUV", "hatchback"],
            "price_range": {"min": Decimal("8000"), "max": Decimal("35000")},
            "max_mileage": 60000,
            "min_year": 2020,
        },
        "buying_history": {
            "total_purchases": 156,
            "avg_price": Decimal("18900.50"),
            "favorite_makes": ["Ford", "Volkswagen", "Toyota"],
        },
        "created_at": "2025-01-10T09:00:00",
        "updated_at": "2026-03-18T11:30:00",
    },
    {
        "dealer_id": "default",
        "name": "Demo Dealer (Default)",
        "location": {
            "city": "Birmingham",
            "region": "West Midlands",
            "latitude": Decimal("52.4862"),
            "longitude": Decimal("-1.8904"),
        },
        "preferences": {
            "fuel_types": ["petrol", "diesel", "hybrid", "electric"],
            "body_types": ["SUV", "saloon", "hatchback", "estate"],
            "price_range": {"min": Decimal("5000"), "max": Decimal("50000")},
            "max_mileage": 100000,
            "min_year": 2017,
        },
        "buying_history": {
            "total_purchases": 0,
            "avg_price": Decimal("0"),
            "favorite_makes": [],
        },
        "created_at": "2026-03-23T00:00:00",
        "updated_at": "2026-03-23T00:00:00",
    },
]


def seed(table_name: str, session: boto3.Session, region: str) -> None:
    dynamodb = session.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    for dealer in DEALERS:
        table.put_item(Item=dealer)
        print(f"  Seeded {dealer['dealer_id']} ({dealer['name']})")

    print(f"\nDone: {len(DEALERS)} dealers written to {table_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed dealer data")
    parser.add_argument("--table", default="agent-eval-dealers-dev")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--expected-account", required=True)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    session, identity = verified_session(
        profile=args.profile,
        region=args.region,
        expected_account=args.expected_account,
    )
    confirm_mutation(
        action="seed-dealer-data",
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
    seed(args.table, session, args.region)
