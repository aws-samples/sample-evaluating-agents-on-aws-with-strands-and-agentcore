# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Evaluation Stack - CloudWatch + SNS + IAM.

Provides monitoring infrastructure for agent evaluation:
- CloudWatch Logs for evaluation results
- SNS topics for alerts
- IAM roles for evaluation access

Cleanup requires the repository retention manifest, explicit profile/account/
region verification, a reviewed destroy change, and approval for the exact
stack and retained-data deletion sets. Evaluation logs and the KMS key are
retained and continue to incur charges after stack deletion.
"""

from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
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
    aws_sns as sns,
)
from constructs import Construct

from .security import (
    explicit_kms_key_policy,
    finalize_explicit_kms_actions,
    grant_cloudwatch_logs_encryption,
)


class EvaluationStack(Stack):
    """Stack for evaluation monitoring and alerting."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        data_bucket: s3.IBucket,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Get environment from context
        env_name = self.node.try_get_context("environment") or "dev"

        evaluation_key_policy = explicit_kms_key_policy()
        evaluation_key = kms.Key(
            self,
            "EvaluationKey",
            description=f"Encrypts evaluation alerts and logs ({env_name})",
            enable_key_rotation=True,
            policy=evaluation_key_policy,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # SNS Topic for evaluation alerts
        self.alert_topic = sns.Topic(
            self,
            "EvaluationAlertTopic",
            topic_name=f"agent-eval-alerts-{env_name}",
            display_name=f"Agent Evaluation Alerts ({env_name})",
            # CDK API requires this specific parameter name.
            master_key=evaluation_key,
        )

        # CloudWatch Log Group for evaluation results
        evaluation_log_group_name = f"/aws/agent-evaluation/{env_name}"
        self.evaluation_log_group = logs.LogGroup(
            self,
            "EvaluationLogGroup",
            log_group_name=evaluation_log_group_name,
            retention=(
                logs.RetentionDays.ONE_WEEK if env_name == "dev" else logs.RetentionDays.ONE_MONTH
            ),
            encryption_key=evaluation_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        grant_cloudwatch_logs_encryption(
            evaluation_key,
            [evaluation_log_group_name],
        )

        # IAM Role for Amazon Bedrock AgentCore Evaluations to access logs and metrics
        self.evaluation_role = iam.Role(
            self,
            "AgentCoreEvaluationRole",
            role_name=f"agent-eval-agentcore-{env_name}",
            assumed_by=iam.ServicePrincipal(
                "bedrock.amazonaws.com",
                conditions={
                    # self.account resolves to your AWS account ID (e.g., 123456789012) at deploy time
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{self.region}:{self.account}:evaluation-job/*"
                    },
                },
            ),
            description="Role for Bedrock AgentCore Evaluations to access logs and metrics",
        )

        # Grant read access to evaluation logs
        self.evaluation_log_group.grant_read(self.evaluation_role)

        # Grant read access to data bucket for evaluation, scoped to the data
        # prefixes the evaluation actually reads (processed LanceDB artifact and
        # raw inputs) rather than the whole bucket — least privilege.
        self.evaluation_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[
                    data_bucket.arn_for_objects("lancedb/*"),
                    data_bucket.arn_for_objects("raw/*"),
                ],
            )
        )

        # An encrypted SNS topic requires both publish and data-key access.
        # Keep these explicit because the L2 grant emits GenerateDataKey*.
        self.evaluation_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sns:Publish"],
                resources=[self.alert_topic.topic_arn],
            )
        )
        self.evaluation_role.add_to_policy(
            iam.PolicyStatement(
                actions=["kms:Decrypt", "kms:GenerateDataKey"],
                resources=[evaluation_key.key_arn],
            )
        )

        # IAM Role for build-time evaluation (strands-agents-evals)
        self.build_eval_role = iam.Role(
            self,
            "BuildTimeEvaluationRole",
            role_name=f"agent-eval-buildtime-{env_name}",
            # Confused-deputy guards: each service may assume this role only on
            # behalf of resources in this account (aws:SourceAccount) and only
            # for the resource type that legitimately runs build-time eval
            # (aws:SourceArn scoped to this account's lambda functions /
            # codebuild projects). Without these, any account could trick the
            # service into assuming the role on its behalf.
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal(
                    "lambda.amazonaws.com",
                    conditions={
                        "StringEquals": {"aws:SourceAccount": self.account},
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:lambda:{self.region}:{self.account}:function:*"
                        },
                    },
                ),
                iam.ServicePrincipal(
                    "codebuild.amazonaws.com",
                    conditions={
                        "StringEquals": {"aws:SourceAccount": self.account},
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:codebuild:{self.region}:{self.account}:project/*"
                        },
                    },
                ),
            ),
            description="Role for build-time evaluation in CI/CD",
        )

        # Grant permissions to invoke Amazon Bedrock models for LLM-as-judge
        self.build_eval_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-sonnet-4-6",
                    f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v2:0",
                ],
            )
        )

        # Grant read access to data bucket, scoped to the data prefixes the
        # build-time evaluation reads (processed LanceDB artifact and raw inputs).
        self.build_eval_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[
                    data_bucket.arn_for_objects("lancedb/*"),
                    data_bucket.arn_for_objects("raw/*"),
                ],
            )
        )

        # Grant write access to evaluation logs
        self.evaluation_log_group.grant_write(self.build_eval_role)

        finalize_explicit_kms_actions(evaluation_key, evaluation_key_policy)

        # Outputs
        cdk.CfnOutput(
            self,
            "AlertTopicArn",
            value=self.alert_topic.topic_arn,
            description="SNS topic ARN for evaluation alerts",
            export_name=f"{env_name}-alert-topic-arn",
        )

        cdk.CfnOutput(
            self,
            "EvaluationLogGroupName",
            value=self.evaluation_log_group.log_group_name,
            description="CloudWatch log group for evaluation results",
            export_name=f"{env_name}-evaluation-log-group",
        )

        cdk.CfnOutput(
            self,
            "EvaluationRoleArn",
            value=self.evaluation_role.role_arn,
            description="IAM role ARN for Amazon Bedrock AgentCore Evaluations",
            export_name=f"{env_name}-evaluation-role-arn",
        )

        cdk.CfnOutput(
            self,
            "BuildEvalRoleArn",
            value=self.build_eval_role.role_arn,
            description="IAM role ARN for build-time evaluation",
            export_name=f"{env_name}-build-eval-role-arn",
        )
