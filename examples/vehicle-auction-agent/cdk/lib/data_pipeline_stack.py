# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Data Pipeline Stack - Amazon EventBridge + AWS Lambda + Amazon S3.

Mock BigQuery integration: Instead of calling BigQuery API, we upload
sample vehicle data to S3 and process it through the pipeline.

Cleanup requires the repository retention manifest, explicit profile/account/
region verification, a reviewed destroy change, and approval for the exact
stack and retained-data deletion sets. The versioned buckets, application
logs, and KMS key are retained in every environment and continue to incur
storage/key charges.
"""

from pathlib import Path
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
    aws_kms as kms,
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

from .security import (
    explicit_kms_key_policy,
    finalize_explicit_kms_actions,
    grant_cloudwatch_logs_encryption,
)

_TLS_S3_ACTIONS = [
    "s3:AbortMultipartUpload",
    "s3:DeleteObject",
    "s3:DeleteObjectVersion",
    "s3:GetBucketAcl",
    "s3:GetBucketLocation",
    "s3:GetObject",
    "s3:GetObjectAttributes",
    "s3:GetObjectVersion",
    "s3:ListBucket",
    "s3:ListBucketMultipartUploads",
    "s3:ListMultipartUploadParts",
    "s3:PutObject",
]
_INGESTION_FUNCTION_DIR = (
    Path(__file__).resolve().parents[2] / "lambda" / "functions" / "data_ingestion"
)


def _deny_insecure_s3_transport(bucket: s3.Bucket) -> None:
    """Require TLS for every S3 operation used by this application."""
    bucket.add_to_resource_policy(
        iam.PolicyStatement(
            sid="DenyInsecureTransport",
            effect=iam.Effect.DENY,
            actions=_TLS_S3_ACTIONS,
            principals=[iam.AnyPrincipal()],
            resources=[bucket.bucket_arn, bucket.arn_for_objects("*")],
            conditions={"Bool": {"aws:SecureTransport": "false"}},
        )
    )


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
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireAccessLogs",
                    expiration=Duration.days(90),
                    noncurrent_version_expiration=Duration.days(30),
                ),
            ],
            removal_policy=RemovalPolicy.RETAIN,
        )
        _deny_insecure_s3_transport(self.access_logs_bucket)

        # S3 Bucket for LanceDB data (mocked BigQuery destination)
        self.data_bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"agent-eval-data-{env_name}-{self.account}-{self.region}",  # real, globally-unique bucket name (account+region); not the amzn-s3-demo-bucket doc placeholder
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
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
            removal_policy=RemovalPolicy.RETAIN,
        )
        _deny_insecure_s3_transport(self.data_bucket)

        data_pipeline_key_policy = explicit_kms_key_policy()
        data_pipeline_key = kms.Key(
            self,
            "DataPipelineKey",
            description=f"Encrypts ingestion Lambda settings and logs ({env_name})",
            enable_key_rotation=True,
            policy=data_pipeline_key_policy,
            removal_policy=RemovalPolicy.RETAIN,
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

        # Ingestion reads the fixed raw input prefix, publishes immutable
        # candidates plus the manifest under lancedb/, and records refresh
        # status under metadata/. It never lists or deletes bucket contents.
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[self.data_bucket.arn_for_objects("raw/*")],
            )
        )
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[
                    self.data_bucket.arn_for_objects("lancedb/*"),
                    self.data_bucket.arn_for_objects("metadata/*"),
                ],
            )
        )
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                actions=["kms:Decrypt", "kms:DescribeKey"],
                resources=[data_pipeline_key.key_arn],
            )
        )

        # Grant Amazon Bedrock permission for Titan embedding generation.
        lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                ],
            )
        )

        # One queue handles both EventBridge target delivery failures and Lambda
        # asynchronous invocation failures. Direct synchronous callers receive
        # the Lambda error response instead.
        dlq = sqs.Queue(
            self,
            "DataIngestionDLQ",
            queue_name=f"agent-eval-ingestion-dlq-{env_name}",
            retention_period=Duration.days(14),
            enforce_ssl=False,
            encryption=sqs.QueueEncryption.KMS_MANAGED,
        )
        dlq.add_to_resource_policy(
            iam.PolicyStatement(
                sid="DenyInsecureTransport",
                effect=iam.Effect.DENY,
                actions=[
                    "sqs:ChangeMessageVisibility",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                    "sqs:GetQueueUrl",
                    "sqs:ReceiveMessage",
                    "sqs:SendMessage",
                ],
                principals=[iam.AnyPrincipal()],
                resources=[dlq.queue_arn],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
            )
        )

        # CloudWatch LogGroup (create before Lambda to use new API)
        log_group_name = f"/aws/lambda/agent-eval-data-ingestion-{env_name}"
        log_group = logs.LogGroup(
            self,
            "DataPipelineLogGroup",
            log_group_name=log_group_name,
            retention=(
                logs.RetentionDays.ONE_WEEK if env_name == "dev" else logs.RetentionDays.ONE_MONTH
            ),
            encryption_key=data_pipeline_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        grant_cloudwatch_logs_encryption(
            data_pipeline_key,
            [log_group_name],
        )

        # Lambda function for data ingestion (mocked BigQuery)
        self.ingestion_function = lambda_.Function(
            self,
            "DataIngestionFunction",
            function_name=f"agent-eval-data-ingestion-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_14,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset(str(_INGESTION_FUNCTION_DIR)),
            role=lambda_role,
            timeout=Duration.minutes(10),
            memory_size=2048,
            reserved_concurrent_executions=2,
            dead_letter_queue=dlq,
            environment_encryption=data_pipeline_key,
            environment={
                "DATA_BUCKET": self.data_bucket.bucket_name,
                "ENVIRONMENT": env_name,
                "REGION": self.region,
                # Mock BigQuery - we'll read from S3 instead
                "MOCK_BIGQUERY": "true",
                "SAMPLE_DATA_KEY": "raw/sample_vehicles.json",
                "MIN_EMBEDDING_SUCCESS_RATIO": "0.95",
                "EXPECTED_EMBEDDING_DIMENSION": "1024",
                "LANCEDB_MANIFEST_KEY": "lancedb/manifest.json",
            },
            log_group=log_group,
        )

        # EventBridge rule for daily data refresh (24-hour schedule)
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

        # Add Lambda as target
        refresh_rule.add_target(
            targets.LambdaFunction(
                self.ingestion_function,
                retry_attempts=2,
                dead_letter_queue=dlq,
            )
        )

        finalize_explicit_kms_actions(data_pipeline_key, data_pipeline_key_policy)

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
