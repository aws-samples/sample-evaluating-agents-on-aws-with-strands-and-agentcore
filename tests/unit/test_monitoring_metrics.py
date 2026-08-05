# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CloudWatch metric contracts for the AgentCore runtime."""

import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_sns as sns
from aws_cdk.assertions import Template

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.monitoring_stack import MonitoringStack  # noqa: E402

_RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:eu-west-1:111122223333:runtime/agent_eval_runtime_dev-example"
)
_MEMORY_ARN = (
    "arn:aws:bedrock-agentcore:eu-west-1:111122223333:memory/agent_eval_memory_dev-example"
)
_DIMENSIONS = [
    {"Name": "Name", "Value": "agent_eval_runtime_dev::DEFAULT"},
    {"Name": "Operation", "Value": "InvokeAgentRuntime"},
    {"Name": "Resource", "Value": _RUNTIME_ARN},
]


def _template() -> Template:
    app = cdk.App(context={"environment": "dev"})
    topic_stack = cdk.Stack(
        app,
        "topic-stack",
        env=cdk.Environment(account="111122223333", region="eu-west-1"),
    )
    topic = sns.Topic(topic_stack, "Topic")
    stack = MonitoringStack(
        app,
        "monitoring-stack",
        evaluation_topic=topic,
        agent_runtime_arn=_RUNTIME_ARN,
        agent_memory_arn=_MEMORY_ARN,
        env=cdk.Environment(account="111122223333", region="eu-west-1"),
    )
    return Template.from_stack(stack)


def test_agentcore_error_alarm_uses_emitted_service_metric() -> None:
    _template().has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "AlarmName": "agent-eval-agent-error-rate-dev",
            "Namespace": "AWS/Bedrock-AgentCore",
            "MetricName": "Errors",
            "Dimensions": _DIMENSIONS,
        },
    )


def test_agentcore_latency_alarm_uses_duration_metric() -> None:
    _template().has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "AlarmName": "agent-eval-agent-latency-p99-dev",
            "Namespace": "AWS/Bedrock-AgentCore",
            "MetricName": "Duration",
            "ExtendedStatistic": "p99",
            "Dimensions": _DIMENSIONS,
        },
    )


def test_agentcore_memory_extraction_errors_are_alarmed() -> None:
    _template().has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "AlarmName": "agent-eval-memory-extraction-error-dev",
            "Namespace": "AWS/Bedrock-AgentCore",
            "MetricName": "Errors",
            "Dimensions": [
                {"Name": "Operation", "Value": "Extraction"},
                {"Name": "Resource", "Value": _MEMORY_ARN},
            ],
        },
    )
