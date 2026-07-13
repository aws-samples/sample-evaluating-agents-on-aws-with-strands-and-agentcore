# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Configuration loader for agent evaluation.

Reads eval_config.yaml and provides typed access to all settings.
This is the bridge between the single config file and the evaluation modules.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentic_evaluation.exceptions import ConfigError
from agentic_evaluation.test_cases import EvaluationLayer, TestCase, TestCategory
from agentic_evaluation.thresholds import EvaluationThresholds

logger = logging.getLogger(__name__)

# Region-agnostic default. Users prepend an Amazon Bedrock inference profile prefix
# (e.g. "eu." or "us.") via ``judge_region_prefix`` in eval_config.yaml.
DEFAULT_JUDGE_MODEL = "anthropic.claude-sonnet-4-6"


@dataclass
class EvalConfig:
    """Parsed evaluation configuration."""

    project_name: str = "agent-evaluation"
    project_description: str = ""
    region: str = "us-east-1"
    environment: str = "dev"
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_region_prefix: str = ""  # e.g. "eu." or "us." — prepended at runtime
    judge_backend: str = "strands"  # strands | noop | <plugin>
    plugin_evaluators: list[dict[str, Any]] = field(default_factory=list)
    tools: dict[str, str] = field(default_factory=dict)
    rubric_trajectory: str = ""
    rubric_output_quality: str = ""
    safety_forbidden_actions: set[str] = field(default_factory=set)
    safety_forbidden_phrases: list[str] = field(default_factory=list)
    thresholds: EvaluationThresholds = field(default_factory=EvaluationThresholds)
    domain_evaluators: dict[str, dict[str, Any]] = field(default_factory=dict)
    test_cases: list[TestCase] = field(default_factory=list)

    def resolved_judge_model(self) -> str:
        """Return judge_model with region prefix applied (if configured)."""
        if self.judge_region_prefix and not self.judge_model.startswith(self.judge_region_prefix):
            return f"{self.judge_region_prefix}{self.judge_model}"
        return self.judge_model


_CATEGORY_MAP = {
    "happy_path": TestCategory.HAPPY_PATH,
    "edge_case": TestCategory.EDGE_CASE,
    "safety": TestCategory.SAFETY,
    "multi_turn": TestCategory.MULTI_TURN,
    "performance": TestCategory.PERFORMANCE,
}

_LAYER_MAP = {
    # Canonical names (used by the accompanying blog post).
    "layer_1_tool_usage": EvaluationLayer.LAYER_1_TOOL_USAGE,
    "layer_2_reasoning": EvaluationLayer.LAYER_2_REASONING,
    "layer_3_output_quality": EvaluationLayer.LAYER_3_OUTPUT_QUALITY,
    # Industry-standard aliases (DeepEval / Ragas / Anthropic vocabulary).
    # These resolve to the same layers so configs can use either name.
    "tool_correctness": EvaluationLayer.LAYER_1_TOOL_USAGE,
    "process_evaluation": EvaluationLayer.LAYER_2_REASONING,
    "outcome_evaluation": EvaluationLayer.LAYER_3_OUTPUT_QUALITY,
}


def _parse_test_case(raw: dict[str, Any]) -> TestCase:
    """Convert a raw YAML dict to a TestCase object."""
    if "id" not in raw or "query" not in raw:
        raise ConfigError(f"test_case missing required keys 'id' and 'query': {raw!r}")
    category_str = raw.get("category", "")
    if category_str and category_str not in _CATEGORY_MAP:
        raise ConfigError(f"Unknown test category: {category_str!r}. Valid: {list(_CATEGORY_MAP)}")
    category = _CATEGORY_MAP.get(category_str, TestCategory.HAPPY_PATH)
    layers = []
    for layer in raw.get("evaluation_layers", []):
        if layer not in _LAYER_MAP:
            raise ConfigError(f"Unknown evaluation layer: {layer!r}. Valid: {list(_LAYER_MAP)}")
        layers.append(_LAYER_MAP[layer])
    return TestCase(
        id=raw["id"],
        query=raw["query"],
        category=category,
        expected_tools=raw.get("expected_tools", []),
        expected_behavior=raw.get("expected_behavior", ""),
        evaluation_layers=layers,
        tags=raw.get("tags", []),
        reference_solution=raw.get("reference_solution"),
        expected_assertion=raw.get("expected_assertion"),
    )


def _load_test_cases(raw: dict[str, Any], config_path: Path | None) -> list[TestCase]:
    """Resolve test cases from either inline ``test_cases`` or ``test_cases_path``.

    ``test_cases_path`` is resolved relative to the config file's directory.
    The referenced file may be a bare list of cases or a mapping with a
    ``test_cases:`` key.
    """
    inline = raw.get("test_cases")
    path_ref = raw.get("test_cases_path")

    if inline and path_ref:
        raise ConfigError("Set either 'test_cases' (inline) or 'test_cases_path' (file), not both")

    if path_ref:
        base = config_path.parent if config_path else Path.cwd()
        tc_path = (base / path_ref).resolve()
        if not tc_path.exists():
            raise ConfigError(f"test_cases_path not found: {tc_path}")
        try:
            with open(tc_path) as f:
                tc_raw = yaml.safe_load(f)
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark
            loc = f"{tc_path}:{mark.line + 1}:{mark.column + 1}" if mark else str(tc_path)
            raise ConfigError(f"YAML parse error in {loc}: {exc.problem}") from exc

        if isinstance(tc_raw, list):
            cases_raw = tc_raw
        elif isinstance(tc_raw, dict):
            cases_raw = tc_raw.get("test_cases", [])
        else:
            raise ConfigError(
                f"{tc_path}: top-level must be a list of cases or a mapping with 'test_cases:'"
            )
        return [_parse_test_case(tc) for tc in cases_raw]

    return [_parse_test_case(tc) for tc in (inline or [])]


def load_config(config_path: str | Path | None = None) -> EvalConfig:
    """Load evaluation config from YAML file.

    Resolution order:
      1. Explicit path argument
      2. EVAL_CONFIG_PATH environment variable
      3. eval_config.yaml in the current working directory
      4. eval_config.yaml in the repo root (relative to this file)
    """
    if config_path is None:
        config_path = os.environ.get("EVAL_CONFIG_PATH")

    if config_path is None:
        cwd_path = Path.cwd() / "eval_config.yaml"
        if cwd_path.exists():
            config_path = cwd_path

    if config_path is None:
        repo_path = Path(__file__).parent.parent / "eval_config.yaml"
        if repo_path.exists():
            config_path = repo_path

    if config_path is None:
        logger.warning(
            "No eval_config.yaml found (checked the EVAL_CONFIG_PATH env var, the "
            "current working directory, and the package root). Falling back to "
            "built-in defaults with 0 test cases — run_all_layers will vacuously "
            "pass. Create one with `agentic-eval init` or set EVAL_CONFIG_PATH."
        )
        return EvalConfig()

    config_path = Path(config_path)
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        loc = f"{config_path}:{mark.line + 1}:{mark.column + 1}" if mark else str(config_path)
        raise ConfigError(f"YAML parse error in {loc}: {exc.problem}") from exc

    if not raw:
        return EvalConfig()
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")

    project = raw.get("project", {})
    rubrics = raw.get("rubrics", {})
    safety = raw.get("safety", {})
    thresh_raw = raw.get("thresholds", {})
    tools_raw = raw.get("tools", {})

    tool_descriptions = {
        name: info.get("description", "") if isinstance(info, dict) else str(info)
        for name, info in tools_raw.items()
    }

    thresholds = EvaluationThresholds(
        tool_selection_accuracy=thresh_raw.get("tool_selection_accuracy", 0.95),
        tool_parameter_accuracy=thresh_raw.get("tool_parameter_accuracy", 0.95),
        helpfulness_score=thresh_raw.get("helpfulness_score", 0.83),
        reasoning_coherence=thresh_raw.get("reasoning_coherence", 0.85),
        goal_success_rate=thresh_raw.get("goal_success_rate", 0.90),
        output_quality_score=thresh_raw.get("output_quality_score", 0.90),
        domain_aggregate_score=thresh_raw.get("domain_aggregate_score", 0.90),
        task_completion_rate=thresh_raw.get("task_completion_rate", 0.95),
        hallucination_rate=thresh_raw.get("hallucination_rate", 0.02),
        response_latency_p50_ms=thresh_raw.get("response_latency_p50_ms", 12000),
        response_latency_p99_ms=thresh_raw.get("response_latency_p99_ms", 25000),
        alert_task_completion=thresh_raw.get("alert_task_completion", 0.80),
        alert_tool_selection=thresh_raw.get("alert_tool_selection", 0.90),
        alert_helpfulness=thresh_raw.get("alert_helpfulness", 0.58),
        alert_latency_p99_ms=thresh_raw.get("alert_latency_p99_ms", 30000),
        alert_hallucination_rate=thresh_raw.get("alert_hallucination_rate", 0.05),
    )

    test_cases = _load_test_cases(raw, config_path)

    plugin_evaluators_raw = raw.get("plugin_evaluators", []) or []
    if not isinstance(plugin_evaluators_raw, list):
        raise ConfigError("'plugin_evaluators' must be a list of {name, params} mappings")

    judge_model = raw.get("judge_model", DEFAULT_JUDGE_MODEL)
    judge_region_prefix = raw.get("judge_region_prefix", "")
    # Allow a region-prefixed judge_model (e.g. "eu.anthropic...") as a
    # shorthand: split the prefix out so resolved_judge_model() doesn't
    # double-prefix when judge_region_prefix is also set.
    if not judge_region_prefix:
        for prefix in ("eu.", "us.", "ap.", "apac."):
            if judge_model.startswith(prefix):
                judge_region_prefix = prefix
                judge_model = judge_model[len(prefix) :]
                break

    return EvalConfig(
        project_name=project.get("name", "agent-evaluation"),
        project_description=project.get("description", ""),
        region=project.get("region", "us-east-1"),
        environment=project.get("environment", "dev"),
        judge_model=judge_model,
        judge_region_prefix=judge_region_prefix,
        judge_backend=raw.get("judge_backend", "strands"),
        plugin_evaluators=plugin_evaluators_raw,
        tools=tool_descriptions,
        rubric_trajectory=rubrics.get("trajectory", ""),
        rubric_output_quality=rubrics.get("output_quality", ""),
        safety_forbidden_actions=set(safety.get("forbidden_actions", [])),
        safety_forbidden_phrases=safety.get("forbidden_phrases", []),
        thresholds=thresholds,
        domain_evaluators=raw.get("domain_evaluators", {}),
        test_cases=test_cases,
    )
