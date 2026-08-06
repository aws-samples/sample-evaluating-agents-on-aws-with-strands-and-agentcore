# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""What the agent runtime optionally connects to, and the auth that follows.

An environment may deploy a subset of the project: the agent still runs without
the AgentCore Gateway (it just loses the dealer-profile tool) and without Amazon
Cognito (inbound auth falls back to IAM SigV4). This module holds that optional
surface as one value object so the runtime stack takes a single ``integrations``
argument instead of five loosely related flags, and derives the inbound auth
decision from it in one place.
"""

from dataclasses import dataclass
from typing import NamedTuple

from aws_cdk import aws_cognito as cognito
from aws_cdk.aws_bedrockagentcore import IGateway, RuntimeAuthorizerConfiguration


@dataclass(frozen=True, slots=True)
class RuntimeIntegrations:
    """Optional collaborators the runtime wires to when they exist.

    Attributes:
        dealer_gateway: AgentCore Gateway, owned by the dealer-api stack, that
            fronts the Dealer API as an MCP tool. When supplied, the runtime
            role is granted invoke access to it.
        gateway_url: The Gateway MCP endpoint, passed to the agent as
            ``GATEWAY_URL``.
        user_pool: Existing user pool to validate inbound JWTs against.
        user_pool_client: App client within that pool.
        enable_cognito: Create a user pool and app client in the runtime stack
            when no existing ``user_pool`` was supplied. Ignored when one was.
    """

    dealer_gateway: IGateway | None = None
    gateway_url: str | None = None
    user_pool: cognito.IUserPool | None = None
    user_pool_client: cognito.IUserPoolClient | None = None
    enable_cognito: bool = False


# Referenced as a default argument rather than constructed in the signature,
# which would be a function call in a default (ruff B008). Safe to share: frozen.
NO_INTEGRATIONS = RuntimeIntegrations()


class InboundAuth(NamedTuple):
    """How callers authenticate to the runtime, and how the agent reads identity.

    Both follow from the same fact — whether a user pool is in play — so they are
    decided together, which keeps the runtime's authorizer and its
    ``IDENTITY_MODE`` from drifting apart.

    Attributes:
        configuration: Authorizer the runtime enforces on inbound invocations.
        identity_mode: How the agent derives an actor id, either from a JWT claim
            or from the single-tenant default.
    """

    configuration: RuntimeAuthorizerConfiguration
    identity_mode: str


def inbound_auth(
    user_pool: cognito.IUserPool | None, user_pool_client: cognito.IUserPoolClient | None
) -> InboundAuth:
    """Choose inbound authorization for the runtime.

    Args:
        user_pool: Pool to validate JWTs against, if one exists.
        user_pool_client: App client within that pool.

    Returns:
        A Cognito JWT authorizer with per-caller identity when both are present,
        otherwise IAM SigV4 with a single-tenant actor id.
    """
    if user_pool and user_pool_client:
        return InboundAuth(
            RuntimeAuthorizerConfiguration.using_cognito(user_pool, [user_pool_client]),
            "jwt_claim",
        )
    return InboundAuth(RuntimeAuthorizerConfiguration.using_iam(), "single_tenant")
