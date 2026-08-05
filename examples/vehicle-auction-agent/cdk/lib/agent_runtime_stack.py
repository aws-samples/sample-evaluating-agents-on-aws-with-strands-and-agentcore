# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Amazon Bedrock AgentCore Runtime Stack — points at an ECR image built by CodeBuild.

Deploys the agent to Amazon Bedrock AgentCore Runtime using a container image.

No local Docker bundling. Uses ``AgentRuntimeArtifact.from_image_uri``
so the container image is built outside CDK (by CodeBuild) and the stack
just registers the runtime.

The agent uses Amazon Bedrock AgentCore Memory for cross-session dealer memory
and reaches dealer profiles through an AgentCore Gateway (owned by the dealer-api
stack) rather than calling Amazon DynamoDB directly.

Security hardening:
- IAM execution role is pre-created with ``aws:SourceAccount`` and
  ``aws:SourceArn`` confused-deputy guards in the trust policy.
- Optional Cognito JWT authorizer when a ``user_pool`` is supplied;
  otherwise IAM auth is used.
- Least-privilege resource ARNs for Amazon Bedrock / S3 / ECR, scoped
  AgentCore Memory grants, and a Gateway-scoped invoke grant.

Cleanup requires the repository retention manifest, explicit profile/account/
region verification, a reviewed destroy change, and approval for the exact
stack and retained-data deletion sets. Agent memory, identity data, the
evaluation secret, and its KMS key are retained after stack deletion.
"""

import hashlib
import json
from typing import Any, Optional

from aws_cdk import ArnFormat, CfnOutput, CfnResource, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk.aws_bedrockagentcore import (
    AgentRuntimeArtifact,
    IGateway,
    Memory,
    MemoryStrategy,
    ProtocolType,
    Runtime,
    RuntimeAuthorizerConfiguration,
    RuntimeNetworkConfiguration,
)
from constructs import Construct

from .security import (
    explicit_kms_key_policy,
    finalize_explicit_kms_actions,
    grant_cloudwatch_logs_encryption,
)

_GUARDRAIL_POLICY_SPEC = {
    "blocked_input_message": (
        "This request was blocked by content safety policy. "
        "I'm a vehicle search assistant; please rephrase your request."
    ),
    "blocked_output_message": "The response was blocked by content safety policy.",
    "content_filters": [
        {"type": "PROMPT_ATTACK", "input": "HIGH", "output": "NONE"},
        {"type": "MISCONDUCT", "input": "HIGH", "output": "HIGH"},
    ],
    "pii_entities": [
        {"type": "EMAIL", "action": "ANONYMIZE"},
        {"type": "PHONE", "action": "ANONYMIZE"},
    ],
    "denied_topics": [],
}
_GUARDRAIL_POLICY_REVISION = hashlib.sha256(
    json.dumps(_GUARDRAIL_POLICY_SPEC, separators=(",", ":"), sort_keys=True).encode()
).hexdigest()[:12]


class AgentRuntimeStack(Stack):
    """Deploys the vehicle search agent to Amazon Bedrock AgentCore Runtime.

    Constructor params
    ------------------
    image_uri : str
        Full ECR image URI built by CodeBuild, e.g.
        ``123456789012.dkr.ecr.eu-west-1.amazonaws.com/agent-eval-runtime@sha256:<digest>``.
    data_bucket : s3.IBucket
        S3 bucket holding LanceDB / vehicle data.
    dealer_gateway : optional IGateway
        AgentCore Gateway (from the dealer-api stack) that fronts the Dealer
        API as an MCP tool. When supplied, the runtime role is granted invoke
        access to it.
    gateway_url : optional str
        The Gateway MCP endpoint, passed to the agent as ``GATEWAY_URL``.
    user_pool / user_pool_client : optional Cognito resources for JWT auth.
        If omitted, the runtime falls back to IAM authorization.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        image_uri: str,
        data_bucket: s3.IBucket,
        dealer_gateway: Optional[IGateway] = None,
        gateway_url: Optional[str] = None,
        user_pool: Optional[cognito.IUserPool] = None,
        user_pool_client: Optional[cognito.IUserPoolClient] = None,
        enable_cognito: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        env_name = self.node.try_get_context("environment") or "dev"

        # ── Optional Cognito JWT inbound auth ───────────────────────────────
        # AgentCore Runtime is invoked via the InvokeAgentRuntime data-plane API;
        # its inbound control is the authorizer (IAM SigV4 by default). When
        # enable_cognito is set and no external pool was supplied, provision a
        # User Pool + app client for JWT auth. Authorization-code grant only
        # (no implicit grant), self sign-up disabled, strong password policy.
        if enable_cognito and user_pool is None:
            user_pool = cognito.UserPool(
                self,
                "AgentUserPool",
                user_pool_name=f"agent-eval-runtime-{env_name}",
                self_sign_up_enabled=False,
                password_policy=cognito.PasswordPolicy(
                    min_length=12,
                    require_lowercase=True,
                    require_uppercase=True,
                    require_digits=True,
                    require_symbols=True,
                ),
                custom_attributes={
                    "dealer_id": cognito.StringAttribute(min_len=1, max_len=64, mutable=True)
                },
                removal_policy=RemovalPolicy.RETAIN,
            )
            user_pool_client = user_pool.add_client(
                "AgentUserPoolClient",
                user_pool_client_name=f"agent-eval-runtime-{env_name}",
                auth_flows=cognito.AuthFlow(admin_user_password=True, user_srp=True),
                o_auth=cognito.OAuthSettings(
                    flows=cognito.OAuthFlows(
                        authorization_code_grant=True, implicit_code_grant=False
                    ),
                    scopes=[cognito.OAuthScope.OPENID],
                ),
                generate_secret=False,
            )

        # ── Pre-create execution role with confused-deputy guards ───────────
        execution_role = iam.Role(
            self,
            "AgentRuntimeRole",
            role_name=f"agent-eval-runtime-role-{env_name}",
            assumed_by=iam.ServicePrincipal(
                "bedrock-agentcore.amazonaws.com",
                conditions={
                    # self.account resolves to your AWS account ID (e.g., 123456789012) at deploy time
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {
                        "aws:SourceArn": (
                            f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:runtime/*"
                        )
                    },
                },
            ),
            description=f"AgentCore Runtime execution role ({env_name})",
        )

        # ── Amazon Bedrock Guardrail ────────────────────────────────────────
        # Defence-in-depth for the agent's model calls. Encodes the same intent
        # as the SYSTEM_PROMPT "Safety guardrails" section as an enforced policy:
        #   - block prompt-injection attempts on input
        #   - block misconduct on input and output
        #   - anonymise dealer PII (email/phone) in model output
        # Bid placement remains structurally unavailable because the runtime has
        # no write/bid tool. Let benign capability questions reach the model so
        # it can provide the explicit refusal defined in the system prompt.
        guardrail = bedrock.CfnGuardrail(
            self,
            "AgentGuardrail",
            name=f"agent-eval-guardrail-{env_name}",
            blocked_input_messaging=_GUARDRAIL_POLICY_SPEC["blocked_input_message"],
            blocked_outputs_messaging=_GUARDRAIL_POLICY_SPEC["blocked_output_message"],
            description=f"Vehicle search agent guardrail ({env_name})",
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type=config["type"],
                        input_strength=config["input"],
                        output_strength=config["output"],
                    )
                    for config in _GUARDRAIL_POLICY_SPEC["content_filters"]
                ]
            ),
            sensitive_information_policy_config=(
                bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                    pii_entities_config=[
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type=config["type"],
                            action=config["action"],
                        )
                        for config in _GUARDRAIL_POLICY_SPEC["pii_entities"]
                    ]
                )
            ),
        )
        guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            "AgentGuardrailVersion",
            guardrail_identifier=guardrail.attr_guardrail_id,
            # Description is create-only. Including the deterministic policy
            # fingerprint forces a new immutable version whenever the policy
            # specification changes, so the runtime never remains on a stale
            # version after the Guardrail draft is updated.
            description=f"Pinned {env_name} policy {_GUARDRAIL_POLICY_REVISION}",
        )
        self.guardrail_id = guardrail.attr_guardrail_id
        self.guardrail_arn = guardrail.attr_guardrail_arn
        self.guardrail_version = guardrail_version.attr_version

        # ── Amazon Bedrock AgentCore Memory ─────────────────────────────────
        # Long-term memory so the agent recalls dealer preferences and facts
        # across sessions. The built-in user-preference and semantic strategies
        # extract structured memory from raw conversation events; the agent's
        # session manager retrieves from the /preferences/{actorId}/ and
        # /facts/{actorId}/ namespaces (see agent/app.py). Short-term (session)
        # transcript persistence is included by default.
        self.memory = Memory(
            self,
            "AgentMemory",
            memory_name=f"agent_eval_memory_{env_name}",
            description=f"Dealer long-term memory ({env_name})",
            memory_strategies=[
                MemoryStrategy.using_user_preference(
                    strategy_name="dealer_preferences",
                    description="Extract dealer preferences for cross-session personalization",
                    namespaces=["/preferences/{actorId}/"],
                ),
                MemoryStrategy.using_semantic(
                    strategy_name="dealer_facts",
                    description="Extract durable dealer facts for cross-session context",
                    namespaces=["/facts/{actorId}/"],
                ),
            ],
        )
        memory_resource = self.memory.node.find_child("Memory")
        if not isinstance(memory_resource, CfnResource):
            raise TypeError("AgentCore Memory did not create the expected CloudFormation resource")
        memory_resource.apply_removal_policy(RemovalPolicy.RETAIN)
        self.memory_id = self.memory.memory_id
        self.memory_arn = self.memory.memory_arn

        # Privileged evaluation telemetry can contain dealer profile data and
        # tool arguments. A generated token gates that response path; only the
        # runtime and explicitly authorized evaluators may read it.
        evaluation_secret_key_policy = explicit_kms_key_policy()
        evaluation_secret_key = kms.Key(
            self,
            "EvaluationSecretKey",
            description=f"Encrypts privileged evaluation authorization ({env_name})",
            enable_key_rotation=True,
            policy=evaluation_secret_key_policy,
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.evaluation_trace_secret = secretsmanager.Secret(
            self,
            "EvaluationTraceSecret",
            secret_name=f"agent-eval/evaluation-trace/{env_name}",
            description=f"Authorizes privileged AgentCore evaluation telemetry ({env_name})",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True,
                password_length=48,
            ),
            encryption_key=evaluation_secret_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        self.evaluation_trace_secret.grant_read(execution_role)

        # ── Reference the existing ECR image (no local Docker) ──────────────
        agent_artifact = AgentRuntimeArtifact.from_image_uri(image_uri)

        # ── Authorizer: Cognito JWT if pool provided, else IAM ──────────────
        if user_pool and user_pool_client:
            authorizer = RuntimeAuthorizerConfiguration.using_cognito(user_pool, [user_pool_client])
        else:
            authorizer = RuntimeAuthorizerConfiguration.using_iam()

        # ── L2 Runtime construct ────────────────────────────────────────────
        # Cost control: AgentCore Harness supports maxTokens, maxIterations, and
        # timeoutSeconds as per-invocation limits. Configure via:
        #   aws bedrock-agentcore-control update-harness --max-tokens 8192
        # These are service-managed knobs — no application-level token tracking needed.
        self.runtime = Runtime(
            self,
            "AgentRuntime",
            runtime_name=f"agent_eval_runtime_{env_name}",
            agent_runtime_artifact=agent_artifact,
            execution_role=execution_role,
            environment_variables={
                "DATA_BUCKET": data_bucket.bucket_name,
                "ENVIRONMENT": env_name,
                "LANCEDB_PATH": "lancedb/manifest.json",
                "LANCEDB_REFRESH_INTERVAL_SECONDS": "60",
                "LANCEDB_CACHE_GENERATIONS": "3",
                "EXPECTED_EMBEDDING_DIMENSION": "1024",
                "GUARDRAIL_ID": self.guardrail_id,
                "GUARDRAIL_VERSION": self.guardrail_version,
                # AgentCore Memory (cross-session dealer memory) and Gateway
                # (dealer-profile tool). The agent reads dealer profiles through
                # the Gateway, not DynamoDB, so no DEALERS_TABLE var is needed.
                "MEMORY_ID": self.memory_id,
                "GATEWAY_URL": gateway_url or "",
                "IDENTITY_MODE": "jwt_claim" if user_pool and user_pool_client else "single_tenant",
                "DEFAULT_ACTOR_ID": (self.node.try_get_context("default_actor_id") or "default"),
                "ACTOR_ID_CLAIM": "custom:dealer_id",
                "EVALUATION_TRACE_SECRET_ID": self.evaluation_trace_secret.secret_arn,
            },
            authorizer_configuration=authorizer,
            network_configuration=RuntimeNetworkConfiguration.using_public_network(),
            protocol_configuration=ProtocolType.HTTP,
            description=f"Vehicle search agent ({env_name})",
        )

        # ── ECR pull permissions (from_image_uri does not auto-grant) ───────
        # URI shape: {account}.dkr.ecr.{region}.amazonaws.com/{repo}[:{tag}|@{digest}]
        # Split on the registry host so namespaced repos (foo/bar) survive, then
        # strip the tag or digest suffix.
        repo_path = image_uri.split(".amazonaws.com/", 1)[-1]
        ecr_repo_name = repo_path.split("@", 1)[0].split(":", 1)[0]
        ecr_repo = ecr.Repository.from_repository_name(self, "EcrRepo", ecr_repo_name)
        ecr_repo.grant_pull(execution_role)

        # The runtime reads only promoted LanceDB manifests and snapshots. Use
        # explicit actions because the L2 grant also includes broad List/Get
        # wildcard actions that this direct-key access path does not need.
        execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[data_bucket.arn_for_objects("lancedb/*")],
            )
        )

        # ── Amazon Bedrock model invocation (cross-region inference profile) ──
        self.runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                # The chat model is a cross-region inference profile
                # (``eu.`` prefix): at invoke time it fans out to the
                # underlying foundation-model ARN in whichever EU region it
                # routes to, so the foundation-model resources are scoped to
                # the ``eu-*`` partition rather than a single region. Pinning
                # to one region would deny requests the profile routes
                # elsewhere in the EU; ``eu-*`` still excludes every other
                # partition (us/ap/etc.).
                # AWS does not expose which EU region a cross-region inference profile routes to,
                # so eu-* is the tightest possible scope.
                resources=[
                    f"arn:aws:bedrock:{self.region}:{self.account}"
                    ":inference-profile/eu.anthropic.claude-sonnet-4-6",
                    "arn:aws:bedrock:eu-*::foundation-model/anthropic.claude-sonnet-4-6",
                    "arn:aws:bedrock:eu-*::foundation-model/amazon.titan-embed-text-v2:0",
                ],
            )
        )

        # ── Apply the guardrail on model invocation ─────────────────────────
        # bedrock:ApplyGuardrail is the action Bedrock requires to enforce a
        # guardrail during Converse/InvokeModel; scope it to this guardrail ARN.
        self.runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:ApplyGuardrail"],
                resources=[self.guardrail_arn],
            )
        )

        # ── AgentCore Memory (read + write + short-term delete) ─────────────
        # The Strands session manager writes conversation events, reads back
        # extracted long-term memory, and DELETES+recreates short-term events
        # when redacting a message — so DeleteEvent is required in addition to
        # read/write. All grants are scoped to this memory resource.
        self.memory.grant_read(execution_role)
        self.memory.grant_write(execution_role)
        self.memory.grant_delete_short_term_memory(execution_role)

        # ── AgentCore Gateway invocation (dealer-profile tool over MCP) ─────
        # The runtime signs its MCP calls to the Gateway with SigV4. Use the
        # Gateway L2's own grant so the exact IAM action stays owned by the
        # construct (least-privilege, scoped to this Gateway ARN).
        if dealer_gateway is not None:
            dealer_gateway.grant_invoke(execution_role)

        # ── HTTP endpoint ───────────────────────────────────────────────────
        self.endpoint = self.runtime.add_endpoint(
            f"agent_eval_endpoint_{env_name}",
            description=f"HTTP endpoint for vehicle search agent ({env_name})",
        )

        # AgentCore creates these groups on first invocation. LogRetention safely
        # creates or adopts them and prevents unbounded service-log accumulation.
        runtime_log_groups = {
            "DefaultRuntimeLogRetention": self.runtime.application_log_group.log_group_name,
            "NamedEndpointLogRetention": (
                f"/aws/bedrock-agentcore/runtimes/{self.runtime.agent_runtime_id}-"
                f"agent_eval_endpoint_{env_name}"
            ),
        }
        retention_constructs: list[logs.LogRetention] = []
        for construct_name, log_group_name in runtime_log_groups.items():
            retention_constructs.append(
                logs.LogRetention(
                    self,
                    construct_name,
                    log_group_name=log_group_name,
                    retention=logs.RetentionDays.ONE_WEEK,
                    removal_policy=RemovalPolicy.RETAIN,
                )
            )

        # LogRetention uses a singleton Lambda. Harden that generated provider:
        # scope its IAM resources, bound concurrency, and pre-create its own
        # encrypted log group so the control does not introduce a logging gap.
        retention_provider_functions = [
            node
            for node in self.node.find_all()
            if isinstance(node, CfnResource)
            and node.cfn_resource_type == "AWS::Lambda::Function"
            and "/LogRetention" in node.node.path
        ]
        retention_provider_policies = [
            node
            for node in self.node.find_all()
            if isinstance(node, iam.CfnPolicy) and "/LogRetention" in node.node.path
        ]
        retention_provider_roles = [
            node
            for node in self.node.find_all()
            if isinstance(node, iam.CfnRole) and "/LogRetention" in node.node.path
        ]
        if (
            len(retention_provider_functions) != 1
            or len(retention_provider_policies) != 1
            or len(retention_provider_roles) != 1
        ):
            raise RuntimeError("Expected one CDK LogRetention provider, role, and policy")

        retention_provider = retention_provider_functions[0]
        retention_provider_log_group_name = f"/agent-eval/control-plane/log-retention/{env_name}"
        retention_provider_log_group = logs.LogGroup(
            self,
            "LogRetentionProviderLogGroup",
            log_group_name=retention_provider_log_group_name,
            retention=logs.RetentionDays.ONE_WEEK,
            encryption_key=evaluation_secret_key,
            # This group belongs only to the deployment helper. Deleting it on
            # rollback prevents a retained fixed name from blocking a retry.
            removal_policy=RemovalPolicy.DESTROY,
        )
        grant_cloudwatch_logs_encryption(
            evaluation_secret_key,
            [retention_provider_log_group_name],
        )

        retention_provider.add_property_override(
            "FunctionName", f"agent-eval-log-retention-{env_name}"
        )
        retention_provider.add_property_override("ReservedConcurrentExecutions", 2)
        retention_provider.add_property_override(
            "LoggingConfig",
            {
                "LogFormat": "JSON",
                "LogGroup": retention_provider_log_group.log_group_name,
            },
        )

        # The generated AWSLambdaBasicExecutionRole policy writes to every log
        # group. The managed group exists before the function, so replace it
        # with stream-write access scoped to that group.
        retention_provider_roles[0].add_deletion_override("Properties.ManagedPolicyArns")
        retention_provider_policies[0].add_property_override(
            "PolicyDocument.Statement.0.Action",
            [
                "logs:CreateLogGroup",
                "logs:DeleteRetentionPolicy",
                "logs:PutRetentionPolicy",
            ],
        )
        retention_provider_policies[0].add_property_override(
            "PolicyDocument.Statement.0.Resource",
            [
                Stack.of(self).format_arn(
                    service="logs",
                    resource="log-group",
                    # CloudWatch Logs authorizes PutRetentionPolicy against a
                    # generated ``:log-stream:`` child ARN. The group-scoped
                    # ``:*`` suffix is therefore required by the service.
                    resource_name=f"{log_group_name}:*",
                    arn_format=ArnFormat.COLON_RESOURCE_NAME,
                )
                for log_group_name in runtime_log_groups.values()
            ],
        )
        retention_provider_policies[0].add_property_override(
            "PolicyDocument.Statement.1",
            {
                # CDK's provider always creates its conventional Lambda group
                # and sets one-day retention, even when LoggingConfig routes
                # function output to the managed group below.
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:PutRetentionPolicy",
                ],
                "Effect": "Allow",
                "Resource": Stack.of(self).format_arn(
                    service="logs",
                    resource="log-group",
                    resource_name=f"/aws/lambda/agent-eval-log-retention-{env_name}:*",
                    arn_format=ArnFormat.COLON_RESOURCE_NAME,
                ),
            },
        )
        retention_provider_policies[0].add_property_override(
            "PolicyDocument.Statement.2",
            {
                "Action": [
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Effect": "Allow",
                "Resource": f"{retention_provider_log_group.log_group_arn}:*",
            },
        )
        for retention_construct in retention_constructs:
            retention_resource = retention_construct.node.default_child
            if not isinstance(retention_resource, CfnResource):
                raise TypeError("Expected LogRetention to create a custom resource")
            retention_resource.add_dependency(retention_provider_log_group.node.default_child)

        # Expose for cross-stack references
        self.runtime_arn = self.runtime.agent_runtime_arn
        self.runtime_id = self.runtime.agent_runtime_id
        finalize_explicit_kms_actions(evaluation_secret_key, evaluation_secret_key_policy)
        CfnOutput(
            self,
            "EvaluationTraceSecretArn",
            value=self.evaluation_trace_secret.secret_arn,
            description="Secrets Manager ARN for authorized evaluation callers",
        )
