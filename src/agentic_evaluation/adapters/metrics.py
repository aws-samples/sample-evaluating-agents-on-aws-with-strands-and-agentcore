# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Token accounting and the operational metrics every adapter reports.

:class:`~agentic_evaluation.evaluators.LatencyEvaluator` and
:class:`~agentic_evaluation.evaluators.CostEvaluator` read fixed key names off
``actual_environment_state``. :func:`base_metrics` is the single place those
names are written, so an adapter cannot silently emit a key the evaluators do
not look for.

.. versionadded:: 0.4.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one agent turn.

    Attributes:
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        total_tokens: Tokens billed overall. Providers that omit this report the
            sum of the other two — see :meth:`from_counts`.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(
        cls,
        input_tokens: Any,
        output_tokens: Any,
        total_tokens: Any = None,
    ) -> TokenUsage:
        """Build usage from a provider's raw counts, defaulting the total.

        Args:
            input_tokens: Prompt-token count, coercible to int.
            output_tokens: Completion-token count, coercible to int.
            total_tokens: Provider-reported total. When None, the sum of the
                other two is used, since not every provider reports it.

        Returns:
            The normalised counts.
        """
        prompt = int(input_tokens or 0)
        completion = int(output_tokens or 0)
        return cls(
            input_tokens=prompt,
            output_tokens=completion,
            total_tokens=int(total_tokens) if total_tokens is not None else prompt + completion,
        )


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """USD price per 1000 tokens, used to estimate the cost of a turn.

    Defaults match Anthropic Claude Sonnet 4.x list pricing ($3 / 1M input,
    $15 / 1M output). Pass an instance to an adapter's ``make_task_fn`` when you
    deploy a different model or have negotiated rates.

    Attributes:
        input_per_1k: USD per 1000 input tokens.
        output_per_1k: USD per 1000 output tokens.
    """

    input_per_1k: float = 0.003
    output_per_1k: float = 0.015

    def cost_usd(self, usage: TokenUsage) -> float:
        """Estimate what one turn cost.

        Args:
            usage: The turn's token counts.

        Returns:
            Estimated cost in USD.
        """
        return (
            usage.input_tokens / 1000 * self.input_per_1k
            + usage.output_tokens / 1000 * self.output_per_1k
        )


DEFAULT_PRICING = TokenPricing()


def base_metrics(
    latency_ms: float,
    usage: TokenUsage,
    pricing: TokenPricing,
) -> dict[str, Any]:
    """Build the latency/token/cost keys the domain evaluators read.

    Args:
        latency_ms: Wall-clock duration of the agent invocation.
        usage: The turn's token counts.
        pricing: Rates used to estimate cost.

    Returns:
        A new dict of metric keys. Adapters add their own identifiers (session
        id, runtime ARN, ...) on top.
    """
    return {
        "latency_ms": latency_ms,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": pricing.cost_usd(usage),
    }
