# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#!/usr/bin/env python3
"""Post-deployment validation script."""

import argparse
import sys

import boto3


def validate_s3_bucket(bucket_name: str, region: str) -> bool:
    """Validate S3 bucket exists and has correct configuration."""
    try:
        s3 = boto3.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket_name)
        print(f"[OK] S3 bucket '{bucket_name}' exists")

        # Check encryption
        encryption = s3.get_bucket_encryption(Bucket=bucket_name)
        print(
            f"[OK] S3 bucket encryption: {encryption['Rules'][0]['ApplyServerSideEncryptionByDefault']['SSEAlgorithm']}"
        )

        # Check versioning
        versioning = s3.get_bucket_versioning(Bucket=bucket_name)
        print(f"[OK] S3 bucket versioning: {versioning.get('Status', 'Disabled')}")

        return True
    except Exception as e:
        print(f"[FAIL] S3 bucket validation failed: {e}")
        return False


def validate_lambda_function(function_name: str, region: str) -> bool:
    """Validate Lambda function exists and has correct configuration."""
    try:
        lambda_client = boto3.client("lambda", region_name=region)
        response = lambda_client.get_function(FunctionName=function_name)

        config = response["Configuration"]
        print(f"[OK] Lambda function '{function_name}' exists")
        print(f"   Runtime: {config['Runtime']}")
        print(f"   Memory: {config['MemorySize']} MB")
        print(f"   Timeout: {config['Timeout']} seconds")

        return True
    except Exception as e:
        print(f"[FAIL] Lambda function validation failed: {e}")
        return False


def validate_cloudwatch_dashboard(dashboard_name: str, region: str) -> bool:
    """Validate CloudWatch dashboard exists."""
    try:
        cw = boto3.client("cloudwatch", region_name=region)
        cw.get_dashboard(DashboardName=dashboard_name)
        print(f"[OK] CloudWatch dashboard '{dashboard_name}' exists")
        return True
    except Exception as e:
        print(f"[FAIL] CloudWatch dashboard validation failed: {e}")
        return False


def validate_eventbridge_rule(rule_name: str, region: str) -> bool:
    """Validate EventBridge rule exists and is enabled."""
    try:
        events = boto3.client("events", region_name=region)
        response = events.describe_rule(Name=rule_name)
        print(f"[OK] EventBridge rule '{rule_name}' exists")
        print(f"   State: {response['State']}")
        print(f"   Schedule: {response.get('ScheduleExpression', 'N/A')}")
        return True
    except Exception as e:
        print(f"[FAIL] EventBridge rule validation failed: {e}")
        return False


def validate_sns_topic(topic_name: str, region: str) -> bool:
    """Validate SNS topic exists."""
    try:
        sns = boto3.client("sns", region_name=region)

        # Get account ID
        sts = boto3.client("sts")
        account_id = sts.get_caller_identity()["Account"]

        topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"
        sns.get_topic_attributes(TopicArn=topic_arn)
        print(f"[OK] SNS topic '{topic_name}' exists")

        # Check subscriptions
        subscriptions = sns.list_subscriptions_by_topic(TopicArn=topic_arn)
        sub_count = len(subscriptions["Subscriptions"])
        print(f"   Subscriptions: {sub_count}")

        return True
    except Exception as e:
        print(f"[FAIL] SNS topic validation failed: {e}")
        return False


def main() -> int:
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate CDK deployment")
    parser.add_argument("--env", default="dev", help="Environment (dev, staging, prod)")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Post-Deployment Validation - {args.env.upper()} ({args.region})")
    print(f"{'=' * 60}\n")

    # Get account ID
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]

    # Resource names based on environment
    bucket_name = f"agent-eval-data-{args.env}-{account_id}-{args.region}"  # mirrors the deployed bucket name from data_pipeline_stack.py; must match exactly to validate the real bucket
    function_name = f"agent-eval-data-ingestion-{args.env}"
    dashboard_name = f"agent-eval-{args.env}"
    rule_name = f"agent-eval-daily-refresh-{args.env}"
    topic_name = f"agent-eval-alerts-{args.env}"

    # Run validations
    results = []

    print("\nValidating Data Pipeline Stack...")
    results.append(validate_s3_bucket(bucket_name, args.region))
    results.append(validate_lambda_function(function_name, args.region))
    results.append(validate_eventbridge_rule(rule_name, args.region))

    print("\nValidating Monitoring Stack...")
    results.append(validate_cloudwatch_dashboard(dashboard_name, args.region))

    print("\nValidating Evaluation Stack...")
    results.append(validate_sns_topic(topic_name, args.region))

    # Summary
    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"[OK] All {total} validation checks passed!")
        print(f"{'=' * 60}\n")
        return 0
    else:
        print(f"[FAIL] {total - passed}/{total} validation checks failed!")
        print(f"{'=' * 60}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
