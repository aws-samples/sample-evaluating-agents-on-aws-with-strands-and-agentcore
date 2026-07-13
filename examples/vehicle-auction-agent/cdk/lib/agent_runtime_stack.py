# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Amazon Bedrock AgentCore Runtime Stack — points at an ECR image built by CodeBuild.

Deploys the agent to Amazon Bedrock AgentCore Runtime using a container image.

No local Docker bundling. Uses ``AgentRuntimeArtifact.from_image_uri``
so the container image is built outside CDK (by CodeBuild) and the stack
just registers the runtime.

Security hardening:
- IAM execution role is pre-created with ``aws:SourceAccount`` and
  ``aws:SourceArn`` confused-deputy guards in the trust policy.
- Optional Cognito JWT authorizer when a ``user_pool`` is supplied;
  otherwise IAM auth is used.
- Least-privilege resource ARNs for Amazon Bedrock / DynamoDB / S3 / ECR.

Cleanup:
    Important: Destroying this stack removes the AgentCore Runtime and its
    endpoint configuration. Any in-flight requests fail immediately.
    Amazon Bedrock AgentCore Runtime incurs per-request charges while active.
    Remove all resources with:
        cdk destroy AgentRuntimeStack -c environment=<env>
    After destroy, verify in the Amazon Bedrock console that no active
    runtimes remain; retained runtimes continue to incur charges.
"""

from typing import Any, Optional

from aws_cdk import RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from aws_cdk.aws_bedrockagentcore import (
    AgentRuntimeArtifact,
    ProtocolType,
    Runtime,
    RuntimeAuthorizerConfiguration,
    RuntimeNetworkConfiguration,
)
from constructs import Construct


class AgentRuntimeStack(Stack):
    """Deploys the vehicle search agent to Amazon Bedrock AgentCore Runtime.

    Constructor params
    ------------------
    image_uri : str
        Full ECR image URI built by CodeBuild, e.g.
        ``123456789012.dkr.ecr.eu-west-1.amazonaws.com/agent-eval-runtime:latest``.
    data_bucket : s3.IBucket
        S3 bucket holding LanceDB / vehicle data.
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
                removal_policy=RemovalPolicy.DESTROY if env_name == "dev" else RemovalPolicy.RETAIN,
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
        #   - deny bid-placement / auction-outcome manipulation as a topic
        #   - anonymise dealer PII (email/phone) in model output
        guardrail = bedrock.CfnGuardrail(
            self,
            "AgentGuardrail",
            name=f"agent-eval-guardrail-{env_name}",
            blocked_input_messaging=(
                "This request was blocked by content safety policy. "
                "I'm a vehicle search assistant — please rephrase your request."
            ),
            blocked_outputs_messaging=("The response was blocked by content safety policy."),
            description=f"Vehicle search agent guardrail ({env_name})",
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE"
                    ),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="MISCONDUCT", input_strength="HIGH", output_strength="HIGH"
                    ),
                ]
            ),
            topic_policy_config=bedrock.CfnGuardrail.TopicPolicyConfigProperty(
                topics_config=[
                    bedrock.CfnGuardrail.TopicConfigProperty(
                        name="BidPlacementOrOutcome",
                        type="DENY",
                        definition=(
                            "Placing or submitting bids on behalf of a user, or predicting, "
                            "estimating, or commenting on auction winning probabilities or "
                            "bid outcomes."
                        ),
                        examples=[
                            "Place a bid of 20000 on vehicle V003 for me.",
                            "What are my chances of winning this auction?",
                            "How likely am I to win if I bid 15000?",
                        ],
                    )
                ]
            ),
            sensitive_information_policy_config=(
                bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                    pii_entities_config=[
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type="EMAIL", action="ANONYMIZE"
                        ),
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type="PHONE", action="ANONYMIZE"
                        ),
                    ]
                )
            ),
        )
        guardrail_version = bedrock.CfnGuardrailVersion(
            self,
            "AgentGuardrailVersion",
            guardrail_identifier=guardrail.attr_guardrail_id,
            description=f"Pinned version for {env_name}",
        )
        self.guardrail_id = guardrail.attr_guardrail_id
        self.guardrail_arn = guardrail.attr_guardrail_arn
        self.guardrail_version = guardrail_version.attr_version

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
                "LANCEDB_PATH": "lancedb/latest.json",
                "DEALERS_TABLE": f"agent-eval-dealers-{env_name}",
                "GUARDRAIL_ID": self.guardrail_id,
                "GUARDRAIL_VERSION": self.guardrail_version,
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

        # ── S3 grant (data bucket holds LanceDB + bids JSON) ────────────────
        data_bucket.grant_read(execution_role)

        # ── Amazon Bedrock model invocation (cross-region inference profile) ──
        self.runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse",
                    "bedrock:ConverseStream",
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

        # ── DynamoDB dealer profiles ────────────────────────────────────────
        self.runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:GetItem", "dynamodb:Query"],
                resources=[
                    f"arn:aws:dynamodb:{self.region}:{self.account}"
                    f":table/agent-eval-dealers-{env_name}",
                ],
            )
        )

        # ── HTTP endpoint ───────────────────────────────────────────────────
        self.endpoint = self.runtime.add_endpoint(
            f"agent_eval_endpoint_{env_name}",
            description=f"HTTP endpoint for vehicle search agent ({env_name})",
        )

        # Expose for cross-stack references
        self.runtime_arn = self.runtime.agent_runtime_arn
        self.runtime_id = self.runtime.agent_runtime_id
