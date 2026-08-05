# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Infrastructure regression tests for the Dealer API stage."""

import sys
from pathlib import Path
from unittest.mock import patch

import aws_cdk as cdk
from aws_cdk import aws_lambda as lambda_
from aws_cdk.assertions import Template

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.dealer_api_stack import DealerApiStack  # noqa: E402


def test_access_log_destination_uses_api_gateway_canonical_arn() -> None:
    app = cdk.App(context={"environment": "dev"})
    with patch(
        "lib.dealer_api_stack.lambda_.Code.from_asset",
        return_value=lambda_.Code.from_inline("def lambda_handler(event, context): return {}"),
    ):
        stack = DealerApiStack(
            app,
            "dealer-api-stack",
            env=cdk.Environment(account="111122223333", region="eu-west-1"),
        )
    template = Template.from_stack(stack)
    stages = template.find_resources("AWS::ApiGateway::Stage")
    access_log_groups = template.find_resources(
        "AWS::Logs::LogGroup",
        {
            "Properties": {
                "LogGroupName": "/aws/apigateway/agent-eval-dealer-api-dev",
            }
        },
    )

    assert len(stages) == 1
    assert len(access_log_groups) == 1
    destination_arn = next(iter(stages.values()))["Properties"]["AccessLogSetting"][
        "DestinationArn"
    ]
    log_group_logical_id = next(iter(access_log_groups))
    assert destination_arn == {
        "Fn::Join": [
            "",
            [
                "arn:",
                {"Ref": "AWS::Partition"},
                ":logs:eu-west-1:111122223333:log-group:",
                {"Ref": log_group_logical_id},
            ],
        ]
    }
