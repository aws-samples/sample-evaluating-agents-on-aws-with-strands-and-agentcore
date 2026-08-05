# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""IAM contract tests for the data-ingestion pipeline."""

import json
import sys
from pathlib import Path

import aws_cdk as cdk
from aws_cdk import aws_s3 as s3
from aws_cdk.assertions import Template

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.data_pipeline_stack import DataPipelineStack  # noqa: E402
from lib.agent_runtime_stack import (  # noqa: E402
    AgentRuntimeStack,
    _GUARDRAIL_POLICY_REVISION,
)


def test_ingestion_can_publish_lancedb_and_refresh_metadata() -> None:
    app = cdk.App(context={"environment": "dev"})
    stack = DataPipelineStack(
        app,
        "test-data-pipeline",
        env=cdk.Environment(account="111122223333", region="eu-west-1"),
    )
    policies = Template.from_stack(stack).find_resources("AWS::IAM::Policy")

    put_object_resources: list[str] = []
    for policy in policies.values():
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions = statement["Action"]
            if isinstance(actions, str):
                actions = [actions]
            if "s3:PutObject" in actions and statement["Effect"] == "Allow":
                put_object_resources.append(json.dumps(statement["Resource"], sort_keys=True))

    resources = " ".join(put_object_resources)
    assert "lancedb/*" in resources
    assert "metadata/*" in resources


def test_guardrail_policy_is_pinned_to_its_fingerprint() -> None:
    app = cdk.App(context={"environment": "dev"})
    storage_stack = cdk.Stack(
        app,
        "test-storage",
        env=cdk.Environment(account="111122223333", region="eu-west-1"),
    )
    data_bucket = s3.Bucket(storage_stack, "DataBucket")
    runtime_stack = AgentRuntimeStack(
        app,
        "test-runtime",
        image_uri=(
            "111122223333.dkr.ecr.eu-west-1.amazonaws.com/"
            "agent-eval-runtime@sha256:"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
        data_bucket=data_bucket,
        env=cdk.Environment(account="111122223333", region="eu-west-1"),
    )
    template = Template.from_stack(runtime_stack)

    guardrail = next(iter(template.find_resources("AWS::Bedrock::Guardrail").values()))
    properties = guardrail["Properties"]
    assert "TopicPolicyConfig" not in properties
    assert properties["ContentPolicyConfig"]["FiltersConfig"] == [
        {"Type": "PROMPT_ATTACK", "InputStrength": "HIGH", "OutputStrength": "NONE"},
        {"Type": "MISCONDUCT", "InputStrength": "HIGH", "OutputStrength": "HIGH"},
    ]

    version = next(iter(template.find_resources("AWS::Bedrock::GuardrailVersion").values()))
    assert version["Properties"]["Description"] == (
        f"Pinned dev policy {_GUARDRAIL_POLICY_REVISION}"
    )
