# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Amazon Bedrock AgentCore Gateway client for the dealer-profile tool.

The agent reaches the Dealer API through an AgentCore Gateway rather than
calling Amazon DynamoDB directly. The Gateway exposes the Dealer API as an
MCP (Model Context Protocol) tool, so the agent consumes it as a standard
Strands ``MCPClient`` tool provider.

Authentication uses SigV4 over the Gateway's MCP endpoint via the first-party
``aws_iam_streamablehttp_client`` transport (from ``mcp-proxy-for-aws``); the
runtime's execution role is the signing principal. No bearer tokens or secrets
are handled here.
"""

import logging

from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)

# AgentCore Gateway MCP endpoints are SigV4-signed against this service name.
_GATEWAY_SERVICE = "bedrock-agentcore"


def build_gateway_mcp_client(gateway_url: str, region: str) -> MCPClient:
    """Build a Strands ``MCPClient`` bound to an AgentCore Gateway endpoint.

    Args:
        gateway_url: The Gateway MCP endpoint, e.g.
            ``https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp``.
        region: AWS region used to sign the SigV4 request.

    Returns:
        An ``MCPClient`` that signs each request with the caller's IAM
        credentials. Call ``start()`` (or use it as a context manager) before
        listing or invoking tools.
    """
    logger.info("Connecting to AgentCore Gateway MCP endpoint")
    return MCPClient(
        lambda: aws_iam_streamablehttp_client(
            endpoint=gateway_url,
            aws_region=region,
            aws_service=_GATEWAY_SERVICE,
        )
    )
