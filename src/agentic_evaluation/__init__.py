# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Reusable agent evaluation SDK.

A three-layer evaluation framework for any agent — local, deployed,
HTTP, Strands, LangChain, CrewAI, custom — driven by a single
``eval_config.yaml``.

- Layer 1 (deterministic): tool selection + trajectory order graders
- Layer 2 (LLM judge): helpfulness + reasoning quality
- Layer 3 (LLM judge): output quality + goal success
- Domain layer: configurable evaluators (latency, cost, safety, freshness,
  schema scoping) plus user-registered plugins

Quick start::

    from agentic_evaluation import run_all_layers, TaskFnResult

    def task_fn(case) -> TaskFnResult:
        return {"output": "...", "trajectory": [...], "metadata": {"latency_ms": 12}}

    results = run_all_layers(task_fn=task_fn)
    print(results["all_passed"])

The shape ``task_fn`` must return is :class:`TaskFnResult`. Framework
adapters that build a ``task_fn`` for you live under
:mod:`agentic_evaluation.adapters` — import them by submodule, e.g.
``from agentic_evaluation.adapters.agentcore import make_task_fn``.
"""

from agentic_evaluation.config import EvalConfig, load_config
from agentic_evaluation.evaluators import (
    CostEvaluator,
    DataFreshnessEvaluator,
    LatencyEvaluator,
    SafetyGuardrailEvaluator,
    SchemaScopingEvaluator,
    ToolSelectionGrader,
    TrajectoryOrderGrader,
)
from agentic_evaluation.exceptions import (
    ConfigError,
    EvaluationError,
    JudgeUnavailableError,
    PluginLoadError,
    PluginNotFoundError,
    TaskFnError,
)
from agentic_evaluation.judges import (
    JudgeBackend,
    NoOpJudgeBackend,
    StrandsJudgeBackend,
    build_judge,
)
from agentic_evaluation.plugins import build_evaluators_from_config, load_evaluator_plugin
from agentic_evaluation.run_experiment import (
    build_cases_from_registry,
    build_domain_experiment,
    build_layer1_experiment,
    build_layer2_experiment,
    build_layer3_experiment,
    run_all_layers,
)
from agentic_evaluation.test_cases import TestCaseRegistry
from agentic_evaluation.thresholds import EVALUATION_THRESHOLDS, EvaluationThresholds
from agentic_evaluation.types import TaskFnResult

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # Config
    "EvalConfig",
    "load_config",
    "EvaluationThresholds",
    "EVALUATION_THRESHOLDS",
    # Generic evaluators
    "DataFreshnessEvaluator",
    "SafetyGuardrailEvaluator",
    "SchemaScopingEvaluator",
    "LatencyEvaluator",
    "CostEvaluator",
    "ToolSelectionGrader",
    "TrajectoryOrderGrader",
    # Judges
    "JudgeBackend",
    "StrandsJudgeBackend",
    "NoOpJudgeBackend",
    "build_judge",
    # Plugins
    "load_evaluator_plugin",
    "build_evaluators_from_config",
    # Exceptions
    "EvaluationError",
    "ConfigError",
    "PluginNotFoundError",
    "PluginLoadError",
    "JudgeUnavailableError",
    "TaskFnError",
    # Experiment builders
    "build_cases_from_registry",
    "build_layer1_experiment",
    "build_layer2_experiment",
    "build_layer3_experiment",
    "build_domain_experiment",
    "run_all_layers",
    # Test infrastructure
    "TestCaseRegistry",
    # Typing contracts
    "TaskFnResult",
]
