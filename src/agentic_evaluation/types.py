# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Public typing contracts for the evaluation SDK.

The most important contract is :class:`TaskFnResult` — the shape every
``task_fn`` must return. ``run_all_layers`` and the ``build_*_experiment``
helpers call your ``task_fn`` once per :class:`strands_evals.Case` and expect
this dict back.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TaskFnResult(TypedDict, total=False):
    """What a ``task_fn`` must return for each evaluated case.

    Only ``output`` and ``trajectory`` are required. ``environment_state`` is
    the recommended channel for operational metrics because ``strands_evals``
    propagates it onto ``EvaluationData`` — unlike task-result ``metadata``,
    which the framework **drops**. ``metadata`` is retained for static fixtures
    and tests that build ``EvaluationData(metadata=...)`` directly.

    Attributes:
        output: The agent's final textual response. Scored by the Layer 3
            output-quality judge.
        trajectory: Either an ordered ``list[str]`` of tool names, or a Strands
            ``Session`` reconstructed from the run. A ``Session`` unlocks the
            Session-level judges (Helpfulness, GoalSuccessRate); the
            deterministic graders accept either form. Use ``[]`` for cases that
            should call no tools (e.g. safety refusals).
        environment_state: List of ``strands_evals`` ``EnvironmentState``
            objects captured after execution. The domain evaluators read an
            entry named ``"metrics"`` whose ``state`` dict carries (all
            optional):

            * ``latency_ms`` (float) — wall-clock latency in **milliseconds**.
              Read by ``LatencyEvaluator``; absent => treated as 0.
            * ``total_tokens`` (int) — total tokens for the turn. Read by
              ``CostEvaluator``.
            * ``estimated_cost_usd`` (float) — estimated USD cost for the
              turn. Read by ``CostEvaluator``.
            * ``last_refresh_time`` (ISO-8601 str) — data recency. Read by
              ``DataFreshnessEvaluator``.
        metadata: Fallback for the same keys as ``environment_state["metrics"]``
            when building ``EvaluationData`` directly. Ignored on the live
            ``strands_evals`` path (the framework drops task-result metadata).
        interactions: Optional multi-agent interaction records.
        input: Optional replacement input for the evaluation (does not mutate
            the original case).
    """

    output: str
    trajectory: Any
    environment_state: list[Any]
    metadata: dict[str, Any]
    interactions: list[Any]
    input: Any
