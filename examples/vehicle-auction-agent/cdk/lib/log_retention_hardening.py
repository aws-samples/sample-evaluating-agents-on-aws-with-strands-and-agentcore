# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Bound retention on service-created log groups without a permissive helper.

AgentCore creates the runtime's log groups on first invocation, so they cannot be
declared up front. ``logs.LogRetention`` safely creates or adopts them and stops
unbounded service-log accumulation — but it does so through a CDK-generated
singleton Lambda that ships with ``AWSLambdaBasicExecutionRole`` and log-retention
permissions on every log group in the account.

This module applies the retention and then constrains that generated provider:
its IAM resources are scoped to the groups it actually acts on, its concurrency
is bounded, and its own log group is pre-created and encrypted so the control does
not introduce a logging gap of its own. Everything here reaches into resources CDK
generated rather than ones the project declared, which is why it is quarantined in
one module.
"""

from aws_cdk import ArnFormat, CfnResource, RemovalPolicy, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from constructs import Construct

from lib.security import grant_cloudwatch_logs_encryption


def configure_runtime_log_retention(
    stack: Stack, env_name: str, encryption_key: kms.Key, log_groups: dict[str, str]
) -> None:
    """Set one-week retention on the runtime's log groups, safely.

    Args:
        stack: Stack the retention constructs are created in.
        env_name: Environment suffix in the provider and log-group names.
        encryption_key: Key encrypting the provider's own log group.
        log_groups: Construct id to log-group name for each group to bound.
    """
    retention_constructs = [
        logs.LogRetention(
            stack,
            construct_id,
            log_group_name=log_group_name,
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.RETAIN,
        )
        for construct_id, log_group_name in log_groups.items()
    ]

    provider_log_group = _harden_provider(stack, env_name, encryption_key, log_groups)
    provider_log_group_resource = _default_cfn_resource(provider_log_group)
    for retention_construct in retention_constructs:
        _default_cfn_resource(retention_construct).add_dependency(provider_log_group_resource)


def _harden_provider(
    stack: Stack, env_name: str, encryption_key: kms.Key, log_groups: dict[str, str]
) -> logs.LogGroup:
    """Constrain the singleton Lambda that CDK's LogRetention generates.

    Args:
        stack: Stack holding the generated provider.
        env_name: Environment suffix in the provider and log-group names.
        encryption_key: Key encrypting the provider's log group.
        log_groups: The groups the provider is allowed to act on.

    Returns:
        The provider's own log group, which the retention custom resources must
        depend on so it exists before the provider first writes to it.
    """
    provider, provider_role, provider_policy = _find_provider(stack)

    log_group_name = f"/agent-eval/control-plane/log-retention/{env_name}"
    provider_log_group = logs.LogGroup(
        stack,
        "LogRetentionProviderLogGroup",
        log_group_name=log_group_name,
        retention=logs.RetentionDays.ONE_WEEK,
        encryption_key=encryption_key,
        # This group belongs only to the deployment helper. Deleting it on
        # rollback prevents a retained fixed name from blocking a retry.
        removal_policy=RemovalPolicy.DESTROY,
    )
    grant_cloudwatch_logs_encryption(encryption_key, [log_group_name])

    provider.add_property_override("FunctionName", f"agent-eval-log-retention-{env_name}")
    provider.add_property_override("ReservedConcurrentExecutions", 2)
    provider.add_property_override(
        "LoggingConfig",
        {
            "LogFormat": "JSON",
            "LogGroup": provider_log_group.log_group_name,
        },
    )

    # The generated AWSLambdaBasicExecutionRole policy writes to every log group.
    # The managed group exists before the function, so replace it with
    # stream-write access scoped to that group.
    provider_role.add_deletion_override("Properties.ManagedPolicyArns")
    _scope_provider_policy(stack, provider_policy, log_groups)
    _grant_provider_self_logging(stack, provider_policy, env_name, provider_log_group.log_group_arn)
    return provider_log_group


def _find_provider(stack: Stack) -> tuple[CfnResource, iam.CfnRole, iam.CfnPolicy]:
    """Locate the function, role, and policy CDK generated for LogRetention.

    Args:
        stack: Stack to search.

    Returns:
        The provider's function, role, and policy resources.

    Raises:
        RuntimeError: The tree does not hold exactly one of each, so the
            overrides applied to them would harden the wrong resource.
    """
    functions = [
        node
        for node in stack.node.find_all()
        if isinstance(node, CfnResource)
        and node.cfn_resource_type == "AWS::Lambda::Function"
        and "/LogRetention" in node.node.path
    ]
    policies = [
        node
        for node in stack.node.find_all()
        if isinstance(node, iam.CfnPolicy) and "/LogRetention" in node.node.path
    ]
    roles = [
        node
        for node in stack.node.find_all()
        if isinstance(node, iam.CfnRole) and "/LogRetention" in node.node.path
    ]
    if len(functions) != 1 or len(policies) != 1 or len(roles) != 1:
        raise RuntimeError("Expected one CDK LogRetention provider, role, and policy")
    return functions[0], roles[0], policies[0]


def _scope_provider_policy(
    stack: Stack, provider_policy: iam.CfnPolicy, log_groups: dict[str, str]
) -> None:
    """Restrict the provider to retention actions on the runtime's groups.

    Args:
        stack: Stack the ARNs are formatted for.
        provider_policy: The generated provider policy.
        log_groups: The groups the provider may set retention on.
    """
    provider_policy.add_property_override(
        "PolicyDocument.Statement.0.Action",
        [
            "logs:CreateLogGroup",
            "logs:DeleteRetentionPolicy",
            "logs:PutRetentionPolicy",
        ],
    )
    provider_policy.add_property_override(
        "PolicyDocument.Statement.0.Resource",
        [
            stack.format_arn(
                service="logs",
                resource="log-group",
                # CloudWatch Logs authorizes PutRetentionPolicy against a
                # generated ``:log-stream:`` child ARN. The group-scoped ``:*``
                # suffix is therefore required by the service.
                resource_name=f"{log_group_name}:*",
                arn_format=ArnFormat.COLON_RESOURCE_NAME,
            )
            for log_group_name in log_groups.values()
        ],
    )


def _grant_provider_self_logging(
    stack: Stack, provider_policy: iam.CfnPolicy, env_name: str, provider_log_group_arn: str
) -> None:
    """Let the provider write its own logs, and only its own.

    Args:
        stack: Stack the ARNs are formatted for.
        provider_policy: The generated provider policy.
        env_name: Environment suffix in the provider's conventional Lambda
            log-group name.
        provider_log_group_arn: ARN of the managed group the provider's output is
            routed to.
    """
    provider_policy.add_property_override(
        "PolicyDocument.Statement.1",
        {
            # CDK's provider always creates its conventional Lambda group and
            # sets one-day retention, even when LoggingConfig routes function
            # output to the managed group below.
            "Action": [
                "logs:CreateLogGroup",
                "logs:PutRetentionPolicy",
            ],
            "Effect": "Allow",
            "Resource": stack.format_arn(
                service="logs",
                resource="log-group",
                resource_name=f"/aws/lambda/agent-eval-log-retention-{env_name}:*",
                arn_format=ArnFormat.COLON_RESOURCE_NAME,
            ),
        },
    )
    provider_policy.add_property_override(
        "PolicyDocument.Statement.2",
        {
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Effect": "Allow",
            "Resource": f"{provider_log_group_arn}:*",
        },
    )


def _default_cfn_resource(construct: Construct) -> CfnResource:
    """Return the CloudFormation resource an L2 construct wraps.

    Args:
        construct: The L2 construct.

    Returns:
        Its default child.

    Raises:
        TypeError: The construct does not wrap a single CloudFormation resource,
            which would silently make the caller's dependency a no-op.
    """
    resource = construct.node.default_child
    if not isinstance(resource, CfnResource):
        raise TypeError(f"{construct.node.path} does not wrap a CloudFormation resource")
    return resource
