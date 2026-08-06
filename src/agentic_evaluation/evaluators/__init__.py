# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Strands-agents-evals compatible evaluators.

Wraps domain-specific custom evaluators as strands_evals.Evaluator subclasses
so they integrate with the Experiment/Case framework while keeping
deterministic, code-based evaluation logic (no LLM calls).

The submodules group evaluators by what makes them change:

- :mod:`~agentic_evaluation.evaluators.tool_usage` — Layer 1 graders, tied to
  your agent's tool schema.
- :mod:`~agentic_evaluation.evaluators.data_quality` — freshness and scoping,
  tied to your data model.
- :mod:`~agentic_evaluation.evaluators.operational` — latency and cost, tied to
  your SLOs and pricing.
- :mod:`~agentic_evaluation.evaluators.safety` — forbidden actions and phrases,
  tied to policy.

Every evaluator is re-exported here, so ``from agentic_evaluation.evaluators
import CostEvaluator`` works regardless of which submodule it lives in.
"""

from agentic_evaluation.evaluators.data_quality import (
    DataFreshnessEvaluator,
    SchemaScopingEvaluator,
    SecondaryScope,
)
from agentic_evaluation.evaluators.operational import CostEvaluator, LatencyEvaluator
from agentic_evaluation.evaluators.safety import SafetyGuardrailEvaluator
from agentic_evaluation.evaluators.tool_usage import (
    ToolParameterGrader,
    ToolSelectionGrader,
    TrajectoryOrderGrader,
)

__all__ = [
    "CostEvaluator",
    "DataFreshnessEvaluator",
    "LatencyEvaluator",
    "SafetyGuardrailEvaluator",
    "SchemaScopingEvaluator",
    "SecondaryScope",
    "ToolParameterGrader",
    "ToolSelectionGrader",
    "TrajectoryOrderGrader",
]
