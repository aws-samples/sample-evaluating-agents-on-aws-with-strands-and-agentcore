# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The content-safety policy the agent runtime enforces on every model call.

Defence-in-depth for the agent's model calls. Encodes the same intent as the
SYSTEM_PROMPT "Safety guardrails" section as an enforced Amazon Bedrock
Guardrail: block prompt-injection attempts on input, block misconduct on input
and output, and anonymise dealer PII (email/phone) in model output. Bid placement
remains structurally unavailable because the runtime has no write/bid tool.
Benign capability questions are left to reach the model so it can give the
explicit refusal defined in the system prompt.

The policy lives here, apart from the runtime stack, so a change to what the
agent is allowed to say is a change to one file with one reason to change.
"""

import hashlib
import json
from typing import NamedTuple

from aws_cdk import aws_bedrock as bedrock
from constructs import Construct

_POLICY_SPEC = {
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

# A deterministic fingerprint of the policy above, embedded in the version
# description so a policy edit forces a new immutable Guardrail version.
_POLICY_REVISION = hashlib.sha256(
    json.dumps(_POLICY_SPEC, separators=(",", ":"), sort_keys=True).encode()
).hexdigest()[:12]


class GuardrailRefs(NamedTuple):
    """References the runtime needs to apply and enforce the guardrail.

    Attributes:
        identifier: Guardrail id, passed to the agent as ``GUARDRAIL_ID``.
        arn: Guardrail ARN, the resource the ``bedrock:ApplyGuardrail`` grant is
            scoped to.
        version: The pinned immutable version, passed as ``GUARDRAIL_VERSION``.
    """

    identifier: str
    arn: str
    version: str


def create_agent_guardrail(scope: Construct, env_name: str) -> GuardrailRefs:
    """Create the guardrail and pin an immutable version of it.

    Args:
        scope: Construct the guardrail is created in, normally the runtime stack.
        env_name: Environment suffix in the guardrail name.

    Returns:
        The guardrail's id, ARN, and pinned version.
    """
    guardrail = bedrock.CfnGuardrail(
        scope,
        "AgentGuardrail",
        name=f"agent-eval-guardrail-{env_name}",
        blocked_input_messaging=_POLICY_SPEC["blocked_input_message"],
        blocked_outputs_messaging=_POLICY_SPEC["blocked_output_message"],
        description=f"Vehicle search agent guardrail ({env_name})",
        content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
            filters_config=[
                bedrock.CfnGuardrail.ContentFilterConfigProperty(
                    type=content_filter["type"],
                    input_strength=content_filter["input"],
                    output_strength=content_filter["output"],
                )
                for content_filter in _POLICY_SPEC["content_filters"]
            ]
        ),
        sensitive_information_policy_config=(
            bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                pii_entities_config=[
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(
                        type=pii_entity["type"],
                        action=pii_entity["action"],
                    )
                    for pii_entity in _POLICY_SPEC["pii_entities"]
                ]
            )
        ),
    )
    version = bedrock.CfnGuardrailVersion(
        scope,
        "AgentGuardrailVersion",
        guardrail_identifier=guardrail.attr_guardrail_id,
        # Description is create-only, so embedding the policy fingerprint is what
        # makes an edit produce a new version rather than leaving the runtime
        # pinned to a stale one after the draft is updated.
        description=f"Pinned {env_name} policy {_POLICY_REVISION}",
    )
    return GuardrailRefs(
        identifier=guardrail.attr_guardrail_id,
        arn=guardrail.attr_guardrail_arn,
        version=version.attr_version,
    )
