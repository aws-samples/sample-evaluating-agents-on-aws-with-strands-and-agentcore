# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Synthesis guards for the repository-wide Lambda invocation boundary."""

import sys
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.security import LambdaInvokeBoundary  # noqa: E402


def _guarded_app() -> tuple[cdk.App, cdk.Stack, lambda_.Function]:
    app = cdk.App()
    stack = cdk.Stack(
        app,
        "test-stack",
        env=cdk.Environment(account="111122223333", region="eu-west-1"),
    )
    function = lambda_.Function(
        stack,
        "Function",
        runtime=lambda_.Runtime.PYTHON_3_14,
        handler="index.handler",
        code=lambda_.Code.from_inline("def handler(event, context): return {}"),
    )
    cdk.Aspects.of(app).add(LambdaInvokeBoundary())
    return app, stack, function


def test_lambda_function_urls_are_rejected_even_with_iam_auth() -> None:
    app, _, function = _guarded_app()
    function.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM)

    with pytest.raises((RuntimeError, ValueError), match="Lambda Function URLs are prohibited"):
        app.synth()


def test_public_lambda_invocation_is_rejected() -> None:
    app, _, function = _guarded_app()
    function.add_permission(
        "PublicInvoke",
        principal=iam.AnyPrincipal(),
        action="lambda:InvokeFunction",
    )

    with pytest.raises((RuntimeError, ValueError), match="specific AWS service principal"):
        app.synth()


def test_account_principal_lambda_invocation_is_rejected() -> None:
    app, _, function = _guarded_app()
    function.add_permission(
        "AccountInvoke",
        principal=iam.AccountPrincipal("111122223333"),
        action="lambda:InvokeFunction",
    )

    with pytest.raises((RuntimeError, ValueError), match="specific AWS service principal"):
        app.synth()


def test_unscoped_service_invocation_is_rejected() -> None:
    app, _, function = _guarded_app()
    function.add_permission(
        "UnscopedEventBridgeInvoke",
        principal=iam.ServicePrincipal("events.amazonaws.com"),
        action="lambda:InvokeFunction",
    )

    with pytest.raises((RuntimeError, ValueError), match="scoped SourceArn"):
        app.synth()


def test_wildcard_source_arn_is_rejected() -> None:
    app, _, function = _guarded_app()
    function.add_permission(
        "WildcardEventBridgeInvoke",
        principal=iam.ServicePrincipal("events.amazonaws.com"),
        action="lambda:InvokeFunction",
        source_arn="*",
    )

    with pytest.raises((RuntimeError, ValueError), match="scoped SourceArn"):
        app.synth()


def test_function_url_only_invocation_condition_is_rejected() -> None:
    app, stack, function = _guarded_app()
    lambda_.CfnPermission(
        stack,
        "FunctionUrlOnlyInvoke",
        action="lambda:InvokeFunction",
        function_name=function.function_name,
        invoked_via_function_url=True,
        principal="cloudfront.amazonaws.com",
        source_arn=stack.format_arn(
            service="cloudfront",
            region="",
            resource="distribution",
            resource_name="EXAMPLE",
        ),
    )

    with pytest.raises((RuntimeError, ValueError), match="never Function URL access"):
        app.synth()


def test_scoped_service_invocation_is_allowed() -> None:
    app, stack, function = _guarded_app()
    function.add_permission(
        "ScopedEventBridgeInvoke",
        principal=iam.ServicePrincipal("events.amazonaws.com"),
        action="lambda:InvokeFunction",
        source_arn=stack.format_arn(
            service="events",
            resource="rule",
            resource_name="expected-rule",
        ),
    )

    app.synth()
