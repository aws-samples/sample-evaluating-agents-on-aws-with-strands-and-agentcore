#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK app for agent evaluation pipeline.

Stack deployment order:
1. data_pipeline - S3 bucket, EventBridge, ingestion Lambda
2. dealer_api - DynamoDB, API Gateway, dealer Lambda
3. agent_runtime - AgentCore Runtime; reads dealer profiles directly from the
   DynamoDB table and LanceDB from the S3 data bucket
4. evaluation - CloudWatch logs, SNS, IAM roles
5. monitoring - Dashboard, alarms

Cleanup and cost:
This app deploys billable resources across all five stacks (Amazon S3,
Amazon DynamoDB, AWS Lambda, Amazon API Gateway, Amazon EventBridge,
Amazon Bedrock AgentCore runtime, and Amazon CloudWatch dashboards/alarms).
Estimated dev cost is roughly $50-100/month and varies with usage.

Important: The following steps delete data in Amazon S3 buckets,
Amazon DynamoDB tables, and Amazon CloudWatch logs. If you need to preserve
any data, create a backup before running them.

To remove everything, follow these steps in order:
1. Identify retained resources: buckets or tables using RemovalPolicy.RETAIN
   survive ``cdk destroy`` and must be emptied or deleted manually first.
   List them with ``cdk list`` and check the AWS Console.
2. Empty retained S3 buckets before destroying (non-empty buckets block stack
   deletion): ``aws s3 rm s3://<bucket-name>/ --recursive``
3. Destroy all stacks:
       cdk destroy --all -c environment=<env>
4. Verify removal with ``cdk list`` and the AWS Console. Retained DynamoDB
   tables must be deleted manually to stop charges.
Always destroy resources when not in use to avoid ongoing charges.
"""

import os

import aws_cdk as cdk
from lib.agent_runtime_stack import AgentRuntimeStack
from lib.data_pipeline_stack import DataPipelineStack
from lib.dealer_api_stack import DealerApiStack
from lib.evaluation_stack import EvaluationStack
from lib.monitoring_stack import MonitoringStack

app = cdk.App()

env_name = app.node.try_get_context("environment") or os.environ.get("ENVIRONMENT", "dev")
region = os.environ.get("AWS_REGION", "eu-west-1")
account = os.environ.get("AWS_ACCOUNT_ID", os.environ.get("CDK_DEFAULT_ACCOUNT"))
if not account:
    raise ValueError("Set AWS_ACCOUNT_ID or CDK_DEFAULT_ACCOUNT environment variable")

env = cdk.Environment(account=account, region=region)
stack_prefix = f"agent-eval-{env_name}"

# Agent container image URI (built by CodeBuild, no local Docker).
# Pass with: `cdk deploy ... -c agent_image_uri=<account>.dkr.ecr.<region>...:tag`
# or set AGENT_IMAGE_URI env var. Without it, the agent_runtime stack is
# skipped (use --build-image in deploy_stack.py to populate it automatically).
agent_image_uri = app.node.try_get_context("agent_image_uri") or os.environ.get("AGENT_IMAGE_URI")
skip_agent_runtime = (
    str(app.node.try_get_context("skip_agent_runtime") or "").lower() == "true"
    or not agent_image_uri
)

# Optional Cognito JWT inbound auth for the AgentCore runtime. Default OFF keeps
# IAM SigV4 auth (a valid mechanism); enable with `-c enable_cognito=true`.
enable_cognito = str(app.node.try_get_context("enable_cognito") or "").lower() == "true"

# 1. Data Pipeline Stack - EventBridge + Lambda + S3
data_pipeline = DataPipelineStack(
    app,
    f"{stack_prefix}-data-pipeline",
    env=env,
    description=f"Data ingestion pipeline ({env_name})",
)

# 2. Dealer API Stack - DynamoDB + API Gateway + SSM params
dealer_api = DealerApiStack(
    app,
    f"{stack_prefix}-dealer-api",
    env=env,
    description=f"Dealer API for AgentCore Gateway ({env_name})",
)

# 3. Agent Runtime Stack - AgentCore Runtime (managed serverless compute).
#    The agent uses AgentCore Memory (cross-session dealer memory) and reaches
#    dealer profiles through the AgentCore Gateway owned by the dealer-api
#    stack, so the Gateway construct and URL are passed in here.
agent_runtime = None
if not skip_agent_runtime:
    agent_runtime = AgentRuntimeStack(
        app,
        f"{stack_prefix}-agent-runtime",
        image_uri=agent_image_uri,
        data_bucket=data_pipeline.data_bucket,
        dealer_gateway=dealer_api.gateway,
        gateway_url=dealer_api.gateway_url,
        enable_cognito=enable_cognito,
        env=env,
        description=f"AgentCore Runtime ({env_name})",
    )
    agent_runtime.add_dependency(data_pipeline)
    agent_runtime.add_dependency(dealer_api)

# 4. Evaluation Stack - CloudWatch + SNS + IAM
evaluation = EvaluationStack(
    app,
    f"{stack_prefix}-evaluation",
    data_bucket=data_pipeline.data_bucket,
    env=env,
    description=f"Evaluation monitoring and alerting ({env_name})",
)

# 5. Monitoring Stack - Dashboards + Alarms
monitoring = MonitoringStack(
    app,
    f"{stack_prefix}-monitoring",
    evaluation_topic=evaluation.alert_topic,
    agent_runtime_arn=(agent_runtime.runtime_arn if agent_runtime is not None else None),
    env=env,
    description=f"CloudWatch dashboards and alarms ({env_name})",
)
if agent_runtime is not None:
    monitoring.add_dependency(agent_runtime)

# Tags
_tagged = [data_pipeline, dealer_api, evaluation, monitoring]
if agent_runtime is not None:
    _tagged.append(agent_runtime)
for stack in _tagged:
    cdk.Tags.of(stack).add("Project", "AgentEvaluation")
    cdk.Tags.of(stack).add("Environment", env_name)
    cdk.Tags.of(stack).add("ManagedBy", "CDK")

app.synth()
