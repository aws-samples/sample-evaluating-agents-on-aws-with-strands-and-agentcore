# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""AgentCore Memory strategy and retrieval namespace contracts."""

import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_s3 as s3
from aws_cdk.assertions import Template

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.agent_runtime_stack import AgentRuntimeStack  # noqa: E402


def _template() -> Template:
    app = cdk.App(context={"environment": "dev"})
    env = cdk.Environment(account="111122223333", region="eu-west-1")
    data_stack = cdk.Stack(app, "data-stack", env=env)
    data_bucket = s3.Bucket(data_stack, "DataBucket")
    stack = AgentRuntimeStack(
        app,
        "runtime-stack",
        image_uri=(
            "111122223333.dkr.ecr.eu-west-1.amazonaws.com/agent-eval-runtime@sha256:" + ("a" * 64)
        ),
        data_bucket=data_bucket,
        env=env,
    )
    return Template.from_stack(stack)


def test_memory_strategy_namespaces_match_runtime_retrieval_config() -> None:
    _template().has_resource_properties(
        "AWS::BedrockAgentCore::Memory",
        {
            "MemoryStrategies": [
                {
                    "UserPreferenceMemoryStrategy": {
                        "Name": "dealer_preferences",
                        "Namespaces": ["/preferences/{actorId}/"],
                        "Type": "USER_PREFERENCE",
                    }
                },
                {
                    "SemanticMemoryStrategy": {
                        "Name": "dealer_facts",
                        "Namespaces": ["/facts/{actorId}/"],
                        "Type": "SEMANTIC",
                    }
                },
            ]
        },
    )


def test_agentcore_runtime_log_groups_have_bounded_retention() -> None:
    template = _template()
    resources = template.find_resources("Custom::LogRetention")

    assert len(resources) == 2
    assert {resource["Properties"]["RetentionInDays"] for resource in resources.values()} == {7}

    provider_functions = template.find_resources(
        "AWS::Lambda::Function",
        {
            "Properties": {
                "FunctionName": "agent-eval-log-retention-dev",
                "LoggingConfig": {
                    "LogFormat": "JSON",
                },
                "ReservedConcurrentExecutions": 2,
            }
        },
    )
    assert len(provider_functions) == 1

    provider_log_groups = template.find_resources(
        "AWS::Logs::LogGroup",
        {
            "Properties": {
                "LogGroupName": "/agent-eval/control-plane/log-retention/dev",
                "RetentionInDays": 7,
            }
        },
    )
    assert len(provider_log_groups) == 1
    provider_log_group_id = next(iter(provider_log_groups))
    provider_function = next(iter(provider_functions.values()))
    assert provider_function["Properties"]["LoggingConfig"]["LogGroup"] == {
        "Ref": provider_log_group_id
    }

    policies = template.find_resources("AWS::IAM::Policy")
    retention_statements = [
        statement
        for policy in policies.values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if "logs:DeleteRetentionPolicy"
        in (statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]])
    ]
    assert len(retention_statements) == 1
    assert set(retention_statements[0]["Action"]) == {
        "logs:CreateLogGroup",
        "logs:DeleteRetentionPolicy",
        "logs:PutRetentionPolicy",
    }
    assert retention_statements[0]["Resource"] != "*"
    assert all(
        resource["Fn::Join"][1][-1].endswith(":*")
        for resource in retention_statements[0]["Resource"]
    )
    provider_group_statements = [
        statement
        for policy in policies.values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if set(
            statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
        )
        == {"logs:CreateLogGroup", "logs:PutRetentionPolicy"}
    ]
    assert len(provider_group_statements) == 1
    assert provider_group_statements[0]["Resource"]["Fn::Join"][1][-1].endswith(
        ":log-group:/aws/lambda/agent-eval-log-retention-dev:*"
    )

    provider_roles = template.find_resources("AWS::IAM::Role")
    provider_role = next(
        role
        for role in provider_roles.values()
        if role["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]
        == {"Service": "lambda.amazonaws.com"}
    )
    assert "ManagedPolicyArns" not in provider_role["Properties"]

    log_write_statements = [
        statement
        for policy in policies.values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if "logs:PutLogEvents"
        in (statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]])
        and statement["Resource"].get("Fn::Join", [None, []])[1][0]
        == {"Fn::GetAtt": [provider_log_group_id, "Arn"]}
    ]
    assert len(log_write_statements) == 1
    assert log_write_statements[0]["Resource"]["Fn::Join"][1][-1] == ":*"
