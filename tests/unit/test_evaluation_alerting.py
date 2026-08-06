# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Alert-delivery contract for the evaluation stack.

Every alarm in ``MonitoringStack`` publishes to ``EvaluationStack``'s SNS topic, so
an unsubscribed topic means the alarms notify nobody. These tests pin the opt-in
``alert_email`` context wiring that closes that gap.
"""

import sys
from pathlib import Path
from typing import Any

import aws_cdk as cdk
import pytest
from aws_cdk import aws_s3 as s3
from aws_cdk.assertions import Template

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.evaluation_stack import EvaluationStack  # noqa: E402

_ENV = cdk.Environment(account="111122223333", region="eu-west-1")


def _template(**context: Any) -> Template:
    app = cdk.App(context={"environment": "dev", **context})
    bucket_stack = cdk.Stack(app, "bucket-stack", env=_ENV)
    bucket = s3.Bucket.from_bucket_arn(
        bucket_stack, "DataBucket", "arn:aws:s3:::amzn-s3-demo-bucket"
    )
    stack = EvaluationStack(app, "evaluation-stack", data_bucket=bucket, env=_ENV)
    return Template.from_stack(stack)


def test_alert_topic_has_no_subscription_by_default() -> None:
    """Omitting the context value must leave the published quickstart deploy unchanged."""
    _template().resource_count_is("AWS::SNS::Subscription", 0)


def test_alert_email_context_subscribes_the_operator() -> None:
    _template(alert_email="ops@example.com").has_resource_properties(
        "AWS::SNS::Subscription",
        {"Protocol": "email", "Endpoint": "ops@example.com"},
    )


def test_non_email_alert_email_context_is_rejected() -> None:
    """Fail at synth rather than letting CloudFormation reject the endpoint mid-deploy."""
    with pytest.raises(ValueError, match="alert_email context must be an email address"):
        _template(alert_email="not-an-address")
