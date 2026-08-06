# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#!/usr/bin/env python3
"""Post-deployment validation script."""

import argparse
import json
import re
import sys
from typing import Any

import boto3

try:
    from scripts.aws_safety import verified_session
except ModuleNotFoundError:
    from aws_safety import verified_session


def validate_s3_bucket(session: boto3.Session, bucket_name: str, region: str) -> bool:
    """Validate S3 bucket exists and has correct configuration."""
    try:
        s3 = session.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket_name)
        print(f"[OK] S3 bucket '{bucket_name}' exists")

        # Check encryption
        encryption = s3.get_bucket_encryption(Bucket=bucket_name)
        rules = encryption["ServerSideEncryptionConfiguration"]["Rules"]
        print(
            "[OK] S3 bucket encryption: "
            f"{rules[0]['ApplyServerSideEncryptionByDefault']['SSEAlgorithm']}"
        )

        # Check versioning
        versioning = s3.get_bucket_versioning(Bucket=bucket_name)
        print(f"[OK] S3 bucket versioning: {versioning.get('Status', 'Disabled')}")
    except Exception as e:
        print(f"[FAIL] S3 bucket validation failed: {e}")
        return False
    else:
        return True


def validate_lambda_function(session: boto3.Session, function_name: str, region: str) -> bool:
    """Validate Lambda function exists and has correct configuration."""
    try:
        lambda_client = session.client("lambda", region_name=region)
        response = lambda_client.get_function(FunctionName=function_name)

        config = response["Configuration"]
        print(f"[OK] Lambda function '{function_name}' exists")
        print(f"   Runtime: {config['Runtime']}")
        print(f"   Memory: {config['MemorySize']} MB")
        print(f"   Timeout: {config['Timeout']} seconds")
    except Exception as e:
        print(f"[FAIL] Lambda function validation failed: {e}")
        return False
    else:
        return True


def _source_arns(statement: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for operator_values in statement.get("Condition", {}).values():
        if not isinstance(operator_values, dict):
            continue
        for key, value in operator_values.items():
            if key.lower() != "aws:sourcearn":
                continue
            values = value if isinstance(value, list) else [value]
            sources.extend(item for item in values if isinstance(item, str))
    return sources


def _invoke_grant_violation(
    statement: dict[str, Any],
    allowed_service_principal: str,
    allowed_source_arn_patterns: tuple[str, ...],
) -> str | None:
    """Describe why an Allow statement is not a scoped service invoke grant."""
    actions = statement.get("Action", [])
    action_list = actions if isinstance(actions, list) else [actions]
    if action_list != ["lambda:InvokeFunction"]:
        return f"has a non-standard resource-policy action: {action_list}"

    principal = statement.get("Principal")
    if principal != {"Service": allowed_service_principal}:
        return f"has an unexpected resource-policy principal: {principal}"

    condition_keys = {
        key.lower()
        for values in statement.get("Condition", {}).values()
        if isinstance(values, dict)
        for key in values
    }
    if {"lambda:functionurlauthtype", "lambda:invokedviafunctionurl"} & condition_keys:
        return "has Function URL permissions"

    sources = _source_arns(statement)
    if not sources or any(
        not any(re.fullmatch(pattern, source) for pattern in allowed_source_arn_patterns)
        for source in sources
    ):
        return f"has an unexpected or unscoped SourceArn: {sources}"
    return None


def validate_lambda_exposure(
    session: boto3.Session,
    function_name: str,
    region: str,
    *,
    allowed_service_principal: str,
    allowed_source_arn_patterns: tuple[str, ...],
) -> bool:
    """Fail if a function URL or an unscoped/non-service invoke grant exists."""
    try:
        lambda_client = session.client("lambda", region_name=region)

        function_urls = [
            config
            for page in lambda_client.get_paginator("list_function_url_configs").paginate(
                FunctionName=function_name
            )
            for config in page.get("FunctionUrlConfigs", [])
        ]
        if function_urls:
            print(f"[FAIL] Lambda '{function_name}' has a Function URL")
            return False

        policy = json.loads(lambda_client.get_policy(FunctionName=function_name)["Policy"])
        invoke_statement_count = 0
        for statement in policy.get("Statement", []):
            if statement.get("Effect") != "Allow":
                continue
            violation = _invoke_grant_violation(
                statement, allowed_service_principal, allowed_source_arn_patterns
            )
            if violation:
                print(f"[FAIL] Lambda '{function_name}' {violation}")
                return False
            invoke_statement_count += 1

        if invoke_statement_count == 0:
            print(f"[FAIL] Lambda '{function_name}' has no approved invocation policy")
            return False
    except Exception as e:
        print(f"[FAIL] Lambda '{function_name}' exposure validation failed: {e}")
        return False
    else:
        print(
            f"[OK] Lambda '{function_name}' has no Function URL and only "
            f"scoped {allowed_service_principal} invocation"
        )
        return True


def find_rest_api_id(session: boto3.Session, api_name: str, region: str) -> str:
    """Resolve one exact REST API name without relying on CLI defaults."""
    apigateway = session.client("apigateway", region_name=region)
    matches = [
        item["id"]
        for page in apigateway.get_paginator("get_rest_apis").paginate()
        for item in page.get("items", [])
        if item.get("name") == api_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one REST API named {api_name!r}, found {len(matches)}")
    return matches[0]


def validate_cloudwatch_dashboard(session: boto3.Session, dashboard_name: str, region: str) -> bool:
    """Validate CloudWatch dashboard exists."""
    try:
        cw = session.client("cloudwatch", region_name=region)
        cw.get_dashboard(DashboardName=dashboard_name)
        print(f"[OK] CloudWatch dashboard '{dashboard_name}' exists")
    except Exception as e:
        print(f"[FAIL] CloudWatch dashboard validation failed: {e}")
        return False
    else:
        return True


def validate_eventbridge_rule(session: boto3.Session, rule_name: str, region: str) -> bool:
    """Validate EventBridge rule exists and is enabled."""
    try:
        events = session.client("events", region_name=region)
        response = events.describe_rule(Name=rule_name)
        print(f"[OK] EventBridge rule '{rule_name}' exists")
        print(f"   State: {response['State']}")
        print(f"   Schedule: {response.get('ScheduleExpression', 'N/A')}")
    except Exception as e:
        print(f"[FAIL] EventBridge rule validation failed: {e}")
        return False
    else:
        return True


def validate_sns_topic(
    session: boto3.Session, topic_name: str, region: str, account_id: str
) -> bool:
    """Validate SNS topic exists."""
    try:
        sns = session.client("sns", region_name=region)

        topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"
        sns.get_topic_attributes(TopicArn=topic_arn)
        print(f"[OK] SNS topic '{topic_name}' exists")

        # Check subscriptions. Paginated: a single call caps at 100 and would
        # under-report the count this check reports on.
        sub_count = sum(
            len(page["Subscriptions"])
            for page in sns.get_paginator("list_subscriptions_by_topic").paginate(
                TopicArn=topic_arn
            )
        )
        print(f"   Subscriptions: {sub_count}")
    except Exception as e:
        print(f"[FAIL] SNS topic validation failed: {e}")
        return False
    else:
        return True


def main() -> int:
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate CDK deployment")
    parser.add_argument("--env", default="dev", help="Environment (dev, staging, prod)")
    parser.add_argument("--profile", required=True, help="Explicit AWS CLI profile")
    parser.add_argument("--region", required=True, help="Explicit AWS region")
    parser.add_argument("--expected-account", required=True)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Post-Deployment Validation - {args.env.upper()} ({args.region})")
    print(f"{'=' * 60}\n")

    session, identity = verified_session(
        profile=args.profile,
        region=args.region,
        expected_account=args.expected_account,
    )
    account_id = identity["Account"]

    # Resource names based on environment
    bucket_name = f"agent-eval-data-{args.env}-{account_id}-{args.region}"  # mirrors the deployed bucket name from data_pipeline_stack.py; must match exactly to validate the real bucket
    ingestion_function_name = f"agent-eval-data-ingestion-{args.env}"
    dealer_function_name = f"agent-eval-dealer-api-{args.env}"
    dealer_api_name = f"agent-eval-dealer-api-{args.env}"
    dashboard_name = f"agent-eval-{args.env}"
    rule_name = f"agent-eval-daily-refresh-{args.env}"
    topic_name = f"agent-eval-alerts-{args.env}"

    # Run validations
    results = []

    print("\nValidating Data Pipeline Stack...")
    results.append(validate_s3_bucket(session, bucket_name, args.region))
    results.append(validate_lambda_function(session, ingestion_function_name, args.region))
    results.append(
        validate_lambda_exposure(
            session,
            ingestion_function_name,
            args.region,
            allowed_service_principal="events.amazonaws.com",
            allowed_source_arn_patterns=(
                re.escape(
                    f"arn:aws:events:{args.region}:{account_id}:"
                    f"rule/agent-eval-daily-refresh-{args.env}"
                ),
            ),
        )
    )
    results.append(validate_eventbridge_rule(session, rule_name, args.region))

    print("\nValidating Dealer API Stack...")
    results.append(validate_lambda_function(session, dealer_function_name, args.region))
    try:
        dealer_api_id = find_rest_api_id(session, dealer_api_name, args.region)
        dealer_source_prefix = re.escape(
            f"arn:aws:execute-api:{args.region}:{account_id}:{dealer_api_id}/"
        )
        dealer_source_pattern = (
            dealer_source_prefix
            + rf"(?:{re.escape(args.env)}|test-invoke-stage)/GET/dealers(?:/\*)?"
        )
        results.append(
            validate_lambda_exposure(
                session,
                dealer_function_name,
                args.region,
                allowed_service_principal="apigateway.amazonaws.com",
                allowed_source_arn_patterns=(dealer_source_pattern,),
            )
        )
    except Exception as e:
        print(f"[FAIL] Dealer REST API validation failed: {e}")
        results.append(False)

    print("\nValidating Monitoring Stack...")
    results.append(validate_cloudwatch_dashboard(session, dashboard_name, args.region))

    print("\nValidating Evaluation Stack...")
    results.append(validate_sns_topic(session, topic_name, args.region, account_id))

    # Summary
    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"[OK] All {total} validation checks passed!")
        print(f"{'=' * 60}\n")
        return 0
    print(f"[FAIL] {total - passed}/{total} validation checks failed!")
    print(f"{'=' * 60}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
