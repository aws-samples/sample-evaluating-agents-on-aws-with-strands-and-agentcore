# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data Pipeline Stack - Amazon EventBridge + AWS Lambda + Amazon S3.

Mock BigQuery integration: Instead of calling BigQuery API, we upload
sample vehicle data to S3 and process it through the pipeline.

Cleanup:
    Important: Destroying this stack deletes resources and their data.
    The Amazon S3 bucket holds versioned data. If you need to preserve any
    data, create a backup first.
    These resources are billable: Amazon S3 bucket (versioned data),
    AWS Lambda function, Amazon EventBridge rule, Amazon SQS
    dead-letter queue.
    Remove them with:
        cdk destroy DataPipelineStack -c environment=<env>
    In non-dev environments the S3 bucket uses RemovalPolicy.RETAIN
    and survives ``cdk destroy``. Empty and delete it manually to stop
    charges.
"""

from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_sqs as sqs,
)
from constructs import Construct


class DataPipelineStack(Stack):
    """Stack for data ingestion pipeline with mocked BigQuery."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Get environment from context
        env_name = self.node.try_get_context("environment") or "dev"

        # S3 Bucket for access logs (security best practice: S3_BUCKET_LOGGING_ENABLED)
        self.access_logs_bucket = s3.Bucket(
            self,
            "AccessLogsBucket",
            bucket_name=f"agent-eval-logs-{env_name}-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireAccessLogs",
                    expiration=Duration.days(90),
                ),
            ],
            removal_policy=(RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN),
            auto_delete_objects=(env_name == "dev"),
        )

        # S3 Bucket for LanceDB data (mocked BigQuery destination)
        self.data_bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"agent-eval-data-{env_name}-{self.account}-{self.region}",  # real, globally-unique bucket name (account+region); not the amzn-s3-demo-bucket doc placeholder
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            server_access_logs_bucket=self.access_logs_bucket,
            server_access_logs_prefix="data-bucket-access-logs/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    noncurrent_version_expiration=Duration.days(30),
                ),
                s3.LifecycleRule(
                    id="DeleteOldData",
                    expiration=Duration.days(90),
                    prefix="raw/",
                ),
            ],
            removal_policy=(RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN),
            auto_delete_objects=(env_name == "dev"),
        )

        # IAM Role for Lambda
        lambda_role = iam.Role(
            self,
            "DataIngestionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )

        # Grant S3 permissions
        self.data_bucket.grant_read_write(lambda_role)

        # Grant Amazon Bedrock permissions for embeddings and contextualization
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.nova-lite-v1:0",
                ],
            )
        )

        # CloudWatch LogGroup (create before Lambda to use new API)
        log_group = logs.LogGroup(
            self,
            "DataPipelineLogGroup",
            log_group_name=f"/aws/lambda/agent-eval-data-ingestion-{env_name}",
            retention=(
                logs.RetentionDays.ONE_WEEK if env_name == "dev" else logs.RetentionDays.ONE_MONTH
            ),
            removal_policy=(RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN),
        )

        # Lambda function for data ingestion (mocked BigQuery)
        self.ingestion_function = lambda_.Function(
            self,
            "DataIngestionFunction",
            function_name=f"agent-eval-data-ingestion-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambda/functions/data_ingestion"),
            role=lambda_role,
            timeout=Duration.minutes(10),
            memory_size=2048,
            environment={
                "DATA_BUCKET": self.data_bucket.bucket_name,
                "ENVIRONMENT": env_name,
                "REGION": self.region,
                # Mock BigQuery - we'll read from S3 instead
                "MOCK_BIGQUERY": "true",
                "SAMPLE_DATA_KEY": "raw/sample_vehicles.json",
            },
            log_group=log_group,
        )

        # EventBridge rule for daily data refresh (23-hour cycle)
        # Runs at 1 AM UTC daily (before 24-hour auction starts)
        refresh_rule = events.Rule(
            self,
            "DailyRefreshRule",
            rule_name=f"agent-eval-daily-refresh-{env_name}",
            schedule=events.Schedule.cron(
                minute="0",
                hour="1",
                month="*",
                week_day="*",
                year="*",
            ),
            enabled=True,
            description="Triggers daily vehicle data refresh for agent evaluation",
        )

        # Dead-letter queue for failed EventBridge invocations
        dlq = sqs.Queue(
            self,
            "DataIngestionDLQ",
            queue_name=f"agent-eval-ingestion-dlq-{env_name}",
            retention_period=Duration.days(14),
            enforce_ssl=True,
            encryption=sqs.QueueEncryption.KMS_MANAGED,
        )

        # Add Lambda as target
        refresh_rule.add_target(
            targets.LambdaFunction(
                self.ingestion_function,
                retry_attempts=2,
                dead_letter_queue=dlq,
            )
        )

        # Outputs
        cdk.CfnOutput(
            self,
            "DataBucketName",
            value=self.data_bucket.bucket_name,
            description="S3 bucket for LanceDB data storage",
            export_name=f"{env_name}-data-bucket-name",
        )

        cdk.CfnOutput(
            self,
            "IngestionFunctionArn",
            value=self.ingestion_function.function_arn,
            description="Data ingestion Lambda function ARN",
            export_name=f"{env_name}-ingestion-function-arn",
        )

        cdk.CfnOutput(
            self,
            "RefreshSchedule",
            value="Daily at 1:00 AM UTC",
            description="EventBridge schedule for data refresh",
        )

        cdk.CfnOutput(
            self,
            "IngestionDLQUrl",
            value=dlq.queue_url,
            description="Dead-letter queue URL for failed data ingestion events",
        )
