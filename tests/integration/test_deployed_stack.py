# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Integration tests for deployed CDK stacks.

These tests verify that CDK resources exist in your AWS account.
They require deployed infrastructure and valid AWS credentials.

Run with:  pytest tests/integration/test_deployed_stack.py -m deployed
Skip with: pytest -m "not deployed"
"""

import os

import boto3
import pytest

pytestmark = pytest.mark.deployed


@pytest.fixture
def aws_region() -> str:
    """Get AWS region from environment."""
    return os.environ.get("AWS_REGION", "eu-west-1")


@pytest.fixture
def environment() -> str:
    """Get environment from environment variables."""
    return os.environ.get("ENVIRONMENT", "dev")


@pytest.fixture
def account_id() -> str:
    """Get AWS account ID."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


class TestDataPipelineStack:
    """Tests for deployed Data Pipeline Stack."""

    def test_data_bucket_exists(self, aws_region: str, environment: str, account_id: str) -> None:
        """Verify S3 data bucket is created with correct configuration."""
        bucket_name = f"agent-eval-data-{environment}-{account_id}-{aws_region}"

        s3 = boto3.client("s3", region_name=aws_region)
        response = s3.head_bucket(Bucket=bucket_name)
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

        # Check encryption
        encryption = s3.get_bucket_encryption(Bucket=bucket_name)
        rules = encryption["ServerSideEncryptionConfiguration"]["Rules"]
        assert rules[0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] in [
            "AES256",
            "aws:kms",
        ]

        # Check versioning
        versioning = s3.get_bucket_versioning(Bucket=bucket_name)
        assert versioning.get("Status") == "Enabled"

    def test_lambda_function_exists(self, aws_region: str, environment: str) -> None:
        """Verify Lambda function is deployed with correct configuration."""
        function_name = f"agent-eval-data-ingestion-{environment}"

        lambda_client = boto3.client("lambda", region_name=aws_region)
        response = lambda_client.get_function(FunctionName=function_name)

        config = response["Configuration"]
        assert config["Runtime"] == "python3.14"
        assert config["MemorySize"] == 2048
        assert config["Timeout"] == 600

        # Check environment variables
        env_vars = config["Environment"]["Variables"]
        assert "DATA_BUCKET" in env_vars
        assert "MOCK_BIGQUERY" in env_vars
        assert env_vars["MOCK_BIGQUERY"] == "true"

    def test_eventbridge_rule_exists(self, aws_region: str, environment: str) -> None:
        """Verify EventBridge rule is created and enabled."""
        rule_name = f"agent-eval-daily-refresh-{environment}"

        events = boto3.client("events", region_name=aws_region)
        response = events.describe_rule(Name=rule_name)

        assert response["State"] == "ENABLED"
        assert response["ScheduleExpression"] == "cron(0 1 ? * * *)"


class TestEvaluationStack:
    """Tests for deployed Evaluation Stack."""

    def test_sns_topic_exists(self, aws_region: str, environment: str, account_id: str) -> None:
        """Verify SNS topic is created."""
        topic_name = f"agent-eval-alerts-{environment}"
        topic_arn = f"arn:aws:sns:{aws_region}:{account_id}:{topic_name}"

        sns = boto3.client("sns", region_name=aws_region)
        response = sns.get_topic_attributes(TopicArn=topic_arn)

        assert response["Attributes"]["TopicArn"] == topic_arn

    def test_log_group_exists(self, aws_region: str, environment: str) -> None:
        """Verify CloudWatch Log Group is created."""
        log_group_name = f"/aws/agent-evaluation/{environment}"

        logs = boto3.client("logs", region_name=aws_region)
        response = logs.describe_log_groups(logGroupNamePrefix=log_group_name)

        log_groups = [lg["logGroupName"] for lg in response["logGroups"]]
        assert log_group_name in log_groups

    def test_evaluation_role_exists(self, aws_region: str, environment: str) -> None:
        """Verify evaluation IAM role is created with correct permissions."""
        role_name = f"agent-eval-agentcore-{environment}"

        iam = boto3.client("iam")
        response = iam.get_role(RoleName=role_name)

        assert response["Role"]["RoleName"] == role_name

        # Check policies (CDK creates inline policies, not attached)
        inline_policies = iam.list_role_policies(RoleName=role_name)
        attached_policies = iam.list_attached_role_policies(RoleName=role_name)
        total_policies = len(inline_policies["PolicyNames"]) + len(
            attached_policies["AttachedPolicies"]
        )

        # Should have at least one policy (inline or attached)
        assert total_policies > 0


class TestMonitoringStack:
    """Tests for deployed Monitoring Stack."""

    def test_cloudwatch_dashboard_exists(self, aws_region: str, environment: str) -> None:
        """Verify CloudWatch dashboard is created."""
        dashboard_name = f"agent-eval-{environment}"

        cw = boto3.client("cloudwatch", region_name=aws_region)
        response = cw.get_dashboard(DashboardName=dashboard_name)

        assert response["DashboardName"] == dashboard_name
        assert "DashboardBody" in response

    def test_cloudwatch_alarms_exist(self, aws_region: str, environment: str) -> None:
        """Verify CloudWatch alarms are created."""
        alarm_prefix = "agent-eval-"

        cw = boto3.client("cloudwatch", region_name=aws_region)
        response = cw.describe_alarms(AlarmNamePrefix=alarm_prefix)

        alarms = response["MetricAlarms"]
        alarm_names = [alarm["AlarmName"] for alarm in alarms]

        # Should have at least 5 alarms (as per design)
        assert len(alarms) >= 5

        # Check for key evaluation alarms
        expected_keywords = [
            "task-completion",
            "tool-selection",
            "helpfulness",
            "latency",
            "hallucination",
        ]
        for expected in expected_keywords:
            assert any(expected in alarm for alarm in alarm_names), (
                f"Missing alarm containing: {expected}"
            )


class TestAgentRuntimeStack:
    """Tests for deployed AgentCore Runtime Stack."""

    @pytest.fixture
    def runtime_id(self, aws_region: str, environment: str) -> str:
        """Find the runtime ID by listing runtimes and matching by name."""
        runtime_name = f"agent_eval_runtime_{environment}"
        client = boto3.client("bedrock-agentcore-control", region_name=aws_region)
        response = client.list_agent_runtimes()
        for rt in response.get("agentRuntimes", []):
            if rt.get("agentRuntimeName") == runtime_name:
                return rt["agentRuntimeId"]
        pytest.fail(f"Runtime '{runtime_name}' not found")

    def test_agentcore_runtime_exists(self, aws_region: str, runtime_id: str) -> None:
        """Verify AgentCore Runtime is deployed and ready."""
        client = boto3.client("bedrock-agentcore-control", region_name=aws_region)
        response = client.get_agent_runtime(agentRuntimeId=runtime_id)

        assert response["status"] in ("READY", "ACTIVE", "CREATING", "UPDATING")

    def test_agentcore_runtime_has_endpoint(self, aws_region: str, runtime_id: str) -> None:
        """Verify AgentCore Runtime has an HTTP endpoint."""
        client = boto3.client("bedrock-agentcore-control", region_name=aws_region)
        endpoints = client.list_agent_runtime_endpoints(agentRuntimeId=runtime_id)
        assert len(endpoints.get("runtimeEndpoints", [])) >= 1


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.mark.skip(reason="Requires full deployment and sample data")
    def test_data_ingestion_workflow(
        self, aws_region: str, environment: str, account_id: str
    ) -> None:
        """Test complete data ingestion workflow."""
        # Upload sample data
        bucket_name = f"agent-eval-data-{environment}-{account_id}-{aws_region}"
        s3 = boto3.client("s3", region_name=aws_region)

        sample_data = {
            "vehicles": [
                {
                    "id": "test-001",
                    "make": "Toyota",
                    "model": "Camry",
                    "year": 2023,
                    "price": 30000,
                }
            ]
        }

        import json

        s3.put_object(
            Bucket=bucket_name,
            Key="raw/test_sample.json",
            Body=json.dumps(sample_data),
        )

        # Invoke Lambda
        lambda_client = boto3.client("lambda", region_name=aws_region)
        function_name = f"agent-eval-data-ingestion-{environment}"

        response = lambda_client.invoke(
            FunctionName=function_name, InvocationType="RequestResponse", Payload="{}"
        )

        assert response["StatusCode"] == 200

        # Check output in S3
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix="lancedb/")
        assert objects["KeyCount"] > 0
