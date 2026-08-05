# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared security helpers for the reference CDK stacks."""

from typing import Any

import jsii
from aws_cdk import ArnFormat, IAspect, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from constructs import IConstruct

# KMS key policies must use Resource "*" because the policy is attached to the
# key itself. Actions remain explicit so account IAM permissions cannot delegate
# unrelated KMS operations through the key policy.
_ACCOUNT_KEY_ACTIONS = [
    "kms:CancelKeyDeletion",
    "kms:CreateGrant",
    "kms:Decrypt",
    "kms:DescribeKey",
    "kms:DisableKey",
    "kms:DisableKeyRotation",
    "kms:EnableKey",
    "kms:EnableKeyRotation",
    "kms:Encrypt",
    "kms:GenerateDataKey",
    "kms:GenerateDataKeyWithoutPlaintext",
    "kms:GetKeyPolicy",
    "kms:GetKeyRotationStatus",
    "kms:ListGrants",
    "kms:ListKeyPolicies",
    "kms:ListKeyRotations",
    "kms:ListResourceTags",
    "kms:ListRetirableGrants",
    "kms:PutKeyPolicy",
    "kms:ReEncryptFrom",
    "kms:ReEncryptTo",
    "kms:RetireGrant",
    "kms:RevokeGrant",
    "kms:RotateKeyOnDemand",
    "kms:ScheduleKeyDeletion",
    "kms:TagResource",
    "kms:UntagResource",
    "kms:UpdateKeyDescription",
]

_KMS_ACTION_EXPANSIONS = {
    "kms:*": _ACCOUNT_KEY_ACTIONS,
    "kms:GenerateDataKey*": [
        "kms:GenerateDataKey",
        "kms:GenerateDataKeyWithoutPlaintext",
    ],
    "kms:ReEncrypt*": ["kms:ReEncryptFrom", "kms:ReEncryptTo"],
}


@jsii.implements(IAspect)
class LambdaInvokeBoundary:
    """Reject Lambda URLs and unscoped resource-based invoke permissions."""

    def visit(self, node: IConstruct) -> None:
        if isinstance(node, lambda_.CfnUrl):
            raise ValueError(
                "Lambda Function URLs are prohibited; use an authenticated API Gateway"
            )

        if not isinstance(node, lambda_.CfnPermission):
            return

        if (
            node.action != "lambda:InvokeFunction"
            or node.function_url_auth_type is not None
            or node.invoked_via_function_url is not None
        ):
            raise ValueError(
                "Lambda permissions may grant only lambda:InvokeFunction, never Function URL access"
            )

        principal = node.principal
        service_suffixes = (".amazonaws.com", ".amazonaws.com.cn")
        if not isinstance(principal, str) or not principal.endswith(service_suffixes):
            raise ValueError(
                "Lambda invocation may be granted only to a specific AWS service principal"
            )

        if node.source_arn is None or (
            isinstance(node.source_arn, str) and node.source_arn.strip() in {"", "*"}
        ):
            raise ValueError(
                "Cross-service Lambda invocation permissions require a scoped SourceArn"
            )


def explicit_kms_key_policy() -> iam.PolicyDocument:
    """Allow account IAM delegation for only the operations these stacks use."""
    return iam.PolicyDocument(
        statements=[
            iam.PolicyStatement(
                sid="EnableAccountKeyAdministrationAndUsage",
                effect=iam.Effect.ALLOW,
                principals=[iam.AccountRootPrincipal()],
                actions=_ACCOUNT_KEY_ACTIONS,
                resources=["*"],
            )
        ]
    )


def grant_cloudwatch_logs_encryption(key: kms.Key, log_group_names: list[str]) -> None:
    """Permit regional CloudWatch Logs encryption for named log groups only."""
    stack = Stack.of(key)
    log_group_arns = [
        stack.format_arn(
            service="logs",
            resource="log-group",
            resource_name=log_group_name,
            arn_format=ArnFormat.COLON_RESOURCE_NAME,
        )
        for log_group_name in log_group_names
    ]
    key.add_to_resource_policy(
        iam.PolicyStatement(
            sid="AllowCloudWatchLogsEncryption",
            effect=iam.Effect.ALLOW,
            principals=[iam.ServicePrincipal(f"logs.{stack.region}.{stack.url_suffix}")],
            actions=[
                "kms:Decrypt",
                "kms:DescribeKey",
                "kms:Encrypt",
                "kms:GenerateDataKey",
                "kms:GenerateDataKeyWithoutPlaintext",
                "kms:ReEncryptFrom",
                "kms:ReEncryptTo",
            ],
            resources=["*"],
            conditions={
                "ArnEquals": {
                    "kms:EncryptionContext:aws:logs:arn": log_group_arns,
                }
            },
        )
    )


def finalize_explicit_kms_actions(key: kms.Key, policy: iam.PolicyDocument) -> None:
    """Expand wildcard actions added by CDK grants before template synthesis."""
    rendered_policy: dict[str, Any] = policy.to_json()
    statements = rendered_policy.get("Statement", [])

    for statement in statements:
        action_value = statement.get("Action")
        actions = action_value if isinstance(action_value, list) else [action_value]
        explicit_actions: list[str] = []
        for action in actions:
            expanded = _KMS_ACTION_EXPANSIONS.get(action, [action])
            for explicit_action in expanded:
                if explicit_action not in explicit_actions:
                    explicit_actions.append(explicit_action)
        statement["Action"] = (
            explicit_actions[0] if len(explicit_actions) == 1 else explicit_actions
        )

    cfn_key = key.node.default_child
    if not isinstance(cfn_key, kms.CfnKey):
        raise TypeError("Expected kms.Key default child to be kms.CfnKey")
    cfn_key.key_policy = rendered_policy
