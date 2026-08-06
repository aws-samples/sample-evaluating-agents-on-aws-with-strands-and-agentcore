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
  otherwise IAM auth is used (see ``lib.runtime_integrations``).
- Least-privilege resource ARNs for Amazon Bedrock / S3 / ECR, scoped
  AgentCore Memory grants, and a Gateway-scoped invoke grant.
- A Bedrock Guardrail on every model call (see ``lib.runtime_guardrail``) and
  bounded retention on the service-created log groups (see
  ``lib.log_retention_hardening``).

Cleanup requires the repository retention manifest, explicit profile/account/
region verification, a reviewed destroy change, and approval for the exact
stack and retained-data deletion sets. Agent memory, identity data, the
evaluation secret, and its KMS key are retained after stack deletion.
"""

from typing import Any

from aws_cdk import CfnOutput, CfnResource, RemovalPolicy, Stack
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk.aws_bedrockagentcore import (
    AgentRuntimeArtifact,
    Memory,
    MemoryStrategy,
    ProtocolType,
    Runtime,
    RuntimeEndpoint,
    RuntimeNetworkConfiguration,
)
from constructs import Construct

from lib.log_retention_hardening import configure_runtime_log_retention
from lib.runtime_guardrail import GuardrailRefs, create_agent_guardrail
from lib.runtime_integrations import (
    NO_INTEGRATIONS,
    InboundAuth,
    RuntimeIntegrations,
    inbound_auth,
)
from lib.security import explicit_kms_key_policy, finalize_explicit_kms_actions


class AgentRuntimeStack(Stack):
    """Deploys the vehicle search agent to Amazon Bedrock AgentCore Runtime."""

    # Cross-stack surface: the monitoring stack alarms on the runtime and memory
    # ARNs, and the evaluation harness reads the trace secret.
    guardrail: GuardrailRefs
    memory: Memory
    memory_id: str
    memory_arn: str
    evaluation_trace_secret: secretsmanager.Secret
    runtime: Runtime
    endpoint: RuntimeEndpoint
    runtime_arn: str
    runtime_id: str

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        image_uri: str,
        data_bucket: s3.IBucket,
        integrations: RuntimeIntegrations = NO_INTEGRATIONS,
        **kwargs: Any,
    ) -> None:
        """Register the agent runtime and everything it needs to run.

        Args:
            scope: The parent construct, normally the CDK ``App``.
            construct_id: Logical id of the stack.
            image_uri: Full ECR image URI built by CodeBuild, in the form
                ``<account>.dkr.ecr.<region>.amazonaws.com/<repo>@sha256:<digest>``.
            data_bucket: Bucket holding the LanceDB vehicle data the agent reads.
            integrations: Optional Gateway and Cognito collaborators. Defaults to
                none, which yields a runtime with IAM inbound auth and no
                dealer-profile tool.
            **kwargs: Passed through to ``Stack``, notably ``env``. The target
                environment is read from the ``environment`` context value.
        """
        super().__init__(scope, construct_id, **kwargs)

        # Get environment from context
        env_name = self.node.try_get_context("environment") or "dev"

        user_pool, user_pool_client = self._resolve_user_pool(env_name, integrations)
        execution_role = self._create_execution_role(env_name)
        self.guardrail = create_agent_guardrail(self, env_name)
        self._create_memory(env_name)

        evaluation_secret_key_policy = explicit_kms_key_policy()
        evaluation_secret_key = self._create_evaluation_secret(
            env_name, execution_role, evaluation_secret_key_policy
        )

        auth = inbound_auth(user_pool, user_pool_client)
        self.runtime = self._create_runtime(
            env_name,
            execution_role,
            auth,
            image_uri,
            self._runtime_environment(
                env_name, data_bucket, integrations.gateway_url, auth.identity_mode
            ),
        )
        self._grant_runtime_access(execution_role, image_uri, data_bucket, integrations)

        # ── HTTP endpoint ───────────────────────────────────────────────────
        self.endpoint = self.runtime.add_endpoint(
            f"agent_eval_endpoint_{env_name}",
            description=f"HTTP endpoint for vehicle search agent ({env_name})",
        )

        configure_runtime_log_retention(
            self,
            env_name,
            evaluation_secret_key,
            {
                "DefaultRuntimeLogRetention": self.runtime.application_log_group.log_group_name,
                "NamedEndpointLogRetention": (
                    f"/aws/bedrock-agentcore/runtimes/{self.runtime.agent_runtime_id}-"
                    f"agent_eval_endpoint_{env_name}"
                ),
            },
        )

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

    def _resolve_user_pool(
        self, env_name: str, integrations: RuntimeIntegrations
    ) -> tuple[cognito.IUserPool | None, cognito.IUserPoolClient | None]:
        """Return the user pool to authorize with, provisioning one if asked.

        AgentCore Runtime is invoked through the InvokeAgentRuntime data-plane
        API, so its inbound control is the authorizer (IAM SigV4 by default).
        When ``enable_cognito`` is set and no external pool was supplied,
        provision a User Pool and app client for JWT auth: authorization-code
        grant only (no implicit grant), self sign-up disabled, and a strong
        password policy.

        Args:
            env_name: Environment suffix in the pool and client names.
            integrations: The stack's optional collaborators.

        Returns:
            The pool and app client to validate inbound JWTs against, or
            ``(None, None)`` when the runtime should fall back to IAM auth.
        """
        if not (integrations.enable_cognito and integrations.user_pool is None):
            return integrations.user_pool, integrations.user_pool_client

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
                flows=cognito.OAuthFlows(authorization_code_grant=True, implicit_code_grant=False),
                scopes=[cognito.OAuthScope.OPENID],
            ),
            generate_secret=False,
        )
        return user_pool, user_pool_client

    def _create_execution_role(self, env_name: str) -> iam.Role:
        """Pre-create the runtime's execution role with confused-deputy guards.

        Args:
            env_name: Environment suffix in the role name.

        Returns:
            The role AgentCore assumes to run the agent.
        """
        return iam.Role(
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

    def _create_memory(self, env_name: str) -> None:
        """Create the long-term memory store the agent recalls dealers from.

        The built-in user-preference and semantic strategies extract structured
        memory from raw conversation events; the agent's session manager
        retrieves from the ``/preferences/{actorId}/`` and ``/facts/{actorId}/``
        namespaces (see ``agent/app.py``). Short-term (session) transcript
        persistence is included by default.

        Args:
            env_name: Environment suffix in the memory name.

        Raises:
            TypeError: The L2 construct did not create the CloudFormation
                resource the retention policy has to be applied to.
        """
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
        # Memory.apply_removal_policy() raises CannotApplyRemovalPolicy: the L2
        # names its child "Memory" instead of the "Resource"/"Default" id that
        # Resource.applyRemovalPolicy looks for, so there is no default child.
        # CDK's own error directs you to the CfnResource, as done here.
        memory_resource = self.memory.node.find_child("Memory")
        if not isinstance(memory_resource, CfnResource):
            raise TypeError("AgentCore Memory did not create the expected CloudFormation resource")
        memory_resource.apply_removal_policy(RemovalPolicy.RETAIN)
        self.memory_id = self.memory.memory_id
        self.memory_arn = self.memory.memory_arn

    def _create_evaluation_secret(
        self, env_name: str, execution_role: iam.Role, key_policy: iam.PolicyDocument
    ) -> kms.Key:
        """Create the token that gates privileged evaluation telemetry.

        Privileged evaluation telemetry can contain dealer profile data and tool
        arguments. A generated token gates that response path, so only the
        runtime and explicitly authorized evaluators may read it.

        Args:
            env_name: Environment suffix in the secret and key names.
            execution_role: Role granted read access to the secret.
            key_policy: Explicit-action policy for the encryption key. The caller
                keeps it to expand wildcard actions after every grant is added.

        Returns:
            The key the secret is encrypted with, reused for the control-plane
            log group created later in this stack.
        """
        evaluation_secret_key = kms.Key(
            self,
            "EvaluationSecretKey",
            description=f"Encrypts privileged evaluation authorization ({env_name})",
            enable_key_rotation=True,
            policy=key_policy,
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
        return evaluation_secret_key

    def _runtime_environment(
        self,
        env_name: str,
        data_bucket: s3.IBucket,
        gateway_url: str | None,
        identity_mode: str,
    ) -> dict[str, str]:
        """Build the environment the agent container reads its configuration from.

        Args:
            env_name: Environment name, also the ``ENVIRONMENT`` value.
            data_bucket: Bucket the agent loads LanceDB artifacts from.
            gateway_url: MCP endpoint of the dealer-profile Gateway, if deployed.
            identity_mode: How the agent derives the actor id for memory.

        Returns:
            The runtime's environment variables.
        """
        return {
            "DATA_BUCKET": data_bucket.bucket_name,
            "ENVIRONMENT": env_name,
            "LANCEDB_PATH": "lancedb/manifest.json",
            "LANCEDB_REFRESH_INTERVAL_SECONDS": "60",
            "LANCEDB_CACHE_GENERATIONS": "3",
            "EXPECTED_EMBEDDING_DIMENSION": "1024",
            "GUARDRAIL_ID": self.guardrail.identifier,
            "GUARDRAIL_VERSION": self.guardrail.version,
            # AgentCore Memory (cross-session dealer memory) and Gateway
            # (dealer-profile tool). The agent reads dealer profiles through
            # the Gateway, not DynamoDB, so no DEALERS_TABLE var is needed.
            "MEMORY_ID": self.memory_id,
            "GATEWAY_URL": gateway_url or "",
            "IDENTITY_MODE": identity_mode,
            "DEFAULT_ACTOR_ID": (self.node.try_get_context("default_actor_id") or "default"),
            "ACTOR_ID_CLAIM": "custom:dealer_id",
            "EVALUATION_TRACE_SECRET_ID": self.evaluation_trace_secret.secret_arn,
        }

    def _create_runtime(
        self,
        env_name: str,
        execution_role: iam.Role,
        auth: InboundAuth,
        image_uri: str,
        environment: dict[str, str],
    ) -> Runtime:
        """Register the runtime against an image CodeBuild already published.

        Cost control: AgentCore Harness supports maxTokens, maxIterations, and
        timeoutSeconds as per-invocation limits. Configure them with
        ``aws bedrock-agentcore-control update-harness --max-tokens 8192``.
        These are service-managed knobs — no application-level token tracking is
        needed.

        Args:
            env_name: Environment suffix in the runtime name.
            execution_role: Role the runtime assumes.
            auth: Inbound authorization, Cognito JWT or IAM.
            image_uri: ECR image the runtime runs, referenced rather than built
                so no local Docker is involved.
            environment: Environment variables for the agent container.

        Returns:
            The registered runtime.
        """
        return Runtime(
            self,
            "AgentRuntime",
            runtime_name=f"agent_eval_runtime_{env_name}",
            agent_runtime_artifact=AgentRuntimeArtifact.from_image_uri(image_uri),
            execution_role=execution_role,
            environment_variables=environment,
            authorizer_configuration=auth.configuration,
            network_configuration=RuntimeNetworkConfiguration.using_public_network(),
            protocol_configuration=ProtocolType.HTTP,
            description=f"Vehicle search agent ({env_name})",
        )

    def _grant_runtime_access(
        self,
        execution_role: iam.Role,
        image_uri: str,
        data_bucket: s3.IBucket,
        integrations: RuntimeIntegrations,
    ) -> None:
        """Grant the runtime exactly the access the agent exercises.

        Args:
            execution_role: Role every grant is attached to.
            image_uri: ECR image URI, parsed for the repository to pull from.
            data_bucket: Bucket holding the LanceDB artifacts.
            integrations: Supplies the Gateway to grant invoke access to, if any.
        """
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
                resources=[self.guardrail.arn],
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
        if integrations.dealer_gateway is not None:
            integrations.dealer_gateway.grant_invoke(execution_role)
