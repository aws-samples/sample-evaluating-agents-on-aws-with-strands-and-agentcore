# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Evaluation Stack - CloudWatch + SNS + IAM.

Provides monitoring infrastructure for agent evaluation:
- CloudWatch Logs for evaluation results
- SNS topics for alerts
- IAM roles for evaluation access

Cleanup:
    These resources are billable. Remove them with:
        cdk destroy EvaluationStack -c environment=<env>
    In non-dev environments some resources use RemovalPolicy.RETAIN and
    survive ``cdk destroy`` (delete them manually to stop charges).
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

        # SNS Topic for evaluation alerts
        self.alert_topic = sns.Topic(
            self,
            "EvaluationAlertTopic",
            topic_name=f"agent-eval-alerts-{env_name}",
            display_name=f"Agent Evaluation Alerts ({env_name})",
            # CDK API requires this specific parameter name; this is an immutable
            # AWS CDK construct property and cannot use inclusive terminology.
            # See: https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_sns/Topic.html
            master_key=kms.Alias.from_alias_name(self, "SnsManagedKey", "alias/aws/sns"),
        )

        # In dev we create the topic without subscriptions and emit the
        # subscribe command as a stack output so an operator can opt in by
        # email manually. Non-dev environments manage subscriptions out of band.
        if env_name == "dev":
            cdk.CfnOutput(
                self,
                "AlertTopicSubscriptionCommand",
                value=f"aws sns subscribe --topic-arn {self.alert_topic.topic_arn} --protocol email --notification-endpoint YOUR_EMAIL@example.com",
                description="Command to subscribe to alerts (replace YOUR_EMAIL)",
            )

        # CloudWatch Log Group for evaluation results
        self.evaluation_log_group = logs.LogGroup(
            self,
            "EvaluationLogGroup",
            log_group_name=f"/aws/agent-evaluation/{env_name}",
            retention=(
                logs.RetentionDays.ONE_WEEK if env_name == "dev" else logs.RetentionDays.ONE_MONTH
            ),
            removal_policy=(RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN),
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
        data_bucket.grant_read(self.evaluation_role, "lancedb/*")
        data_bucket.grant_read(self.evaluation_role, "raw/*")

        # Grant permissions to write CloudWatch metrics
        # SECURITY NOTE: Resource "*" is required by AWS for CloudWatch PutMetricData.
        # This is a non-resource-level action that does not support resource-level ARNs.
        # Scope is enforced by the namespace condition below.
        # See: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazoncloudwatch.html
        self.evaluation_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutMetricData",
                ],
                resources=[
                    "*"
                ],  # AWS service limitation: CloudWatch PutMetricData is a non-resource-level action and does not support resource-level ARNs; scope is enforced by the namespace condition below.
                conditions={
                    "StringEquals": {"cloudwatch:namespace": f"AgentEvaluation/{env_name}"}
                },
            )
        )

        # Grant permissions to publish to SNS
        self.alert_topic.grant_publish(self.evaluation_role)

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
        data_bucket.grant_read(self.build_eval_role, "lancedb/*")
        data_bucket.grant_read(self.build_eval_role, "raw/*")

        # Grant write access to evaluation logs
        self.evaluation_log_group.grant_write(self.build_eval_role)

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
