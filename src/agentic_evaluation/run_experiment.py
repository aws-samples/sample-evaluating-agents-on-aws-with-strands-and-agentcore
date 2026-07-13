# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Experiment runner.

Builds an Experiment from test cases in the registry, configures both
deterministic graders and pluggable LLM-judge evaluators, and runs
evaluations against any agent via a user-provided ``task_fn``.

Config-driven: reads eval_config.yaml for tool descriptions, rubrics,
safety rules, thresholds, and the judge backend. Override config path
with the EVAL_CONFIG_PATH env var.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from strands_evals import Case, Experiment

from agentic_evaluation.config import EvalConfig, load_config
from agentic_evaluation.types import TaskFnResult
from agentic_evaluation.evaluators import (
    CostEvaluator,
    DataFreshnessEvaluator,
    LatencyEvaluator,
    SafetyGuardrailEvaluator,
    SchemaScopingEvaluator,
    ToolSelectionGrader,
    TrajectoryOrderGrader,
)
from agentic_evaluation.judges import JudgeBackend, build_judge
from agentic_evaluation.plugins import build_evaluators_from_config
from agentic_evaluation.test_cases import TestCaseRegistry, TestCategory

logger = logging.getLogger(__name__)

# Industry-standard alias for each canonical layer key in run_all_layers()
# results. Lets callers read results using the vocabulary other eval tools
# use (DeepEval / Ragas / Anthropic) without breaking the canonical keys.
_LAYER_ALIASES = {
    "layer_1": "tool_correctness",
    "layer_2": "process_evaluation",
    "layer_3": "outcome_evaluation",
    "domain": "operational_metrics",
}

# Global config — loaded lazily on first use
_config: EvalConfig | None = None


def get_config() -> EvalConfig:
    """Get the evaluation config, loading from YAML if not yet loaded."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config_cache() -> None:
    """Reset the cached config (used by tests and the CLI)."""
    global _config
    _config = None


def _get_judge_model() -> str:
    """Get judge model ID from env var, config, or default."""
    if "JUDGE_MODEL_ID" in os.environ:
        return os.environ["JUDGE_MODEL_ID"]
    cfg = get_config()
    return cfg.resolved_judge_model()


def build_cases_from_registry(
    registry: TestCaseRegistry | None = None,
    category: TestCategory | None = None,
) -> list[Case[str, str]]:
    """Convert TestCaseRegistry entries to strands_evals Case objects.

    If no registry is provided, loads test cases from eval_config.yaml
    automatically. This is the recommended path for SDK users.

    Args:
        registry: Test case registry (loads from config if None)
        category: Filter to a specific category, or None for all

    Returns:
        List of Case objects ready for an Experiment
    """
    if registry is None:
        cfg = get_config()
        if cfg.test_cases:
            registry = TestCaseRegistry.from_config(cfg.test_cases)
        else:
            logger.warning(
                "No test cases found in eval_config.yaml. "
                "Define test_cases in your config or pass a TestCaseRegistry."
            )
            registry = TestCaseRegistry()

    test_cases = registry.get_by_category(category) if category else registry.test_cases

    return [
        Case[str, str](
            name=tc.id,
            input=tc.query,
            expected_output=tc.expected_behavior,
            expected_trajectory=tc.expected_tools,
            expected_assertion=tc.expected_assertion,
            metadata={
                "category": tc.category.value,
                "tags": tc.tags,
                "evaluation_layers": [layer.value for layer in tc.evaluation_layers],
                "reference_solution": tc.reference_solution,
            },
        )
        for tc in test_cases
    ]


def build_layer1_experiment(
    cases: list[Case[str, str]] | None = None,
) -> Experiment[str, str]:
    """Build Layer 1 (Tool Usage) experiment with deterministic graders.

    Per-grader thresholds come from ``eval_config.yaml`` (the ``thresholds``
    block), falling back to the strict defaults in :class:`EvaluationThresholds`.
    """
    if cases is None:
        cases = build_cases_from_registry()

    thresholds = get_config().thresholds
    return Experiment[str, str](
        cases=cases,
        evaluators=[
            ToolSelectionGrader(threshold=thresholds.tool_selection_accuracy),
            TrajectoryOrderGrader(threshold=thresholds.reasoning_coherence),
        ],
    )


def build_layer2_experiment(
    cases: list[Case[str, str]] | None = None,
    judge_backend: JudgeBackend | None = None,
) -> Experiment[str, str]:
    """Build Layer 2 (Reasoning Quality) experiment.

    Tool descriptions and rubric come from eval_config.yaml. The judge
    backend (LLM stack) is pluggable — defaults to Strands.
    """
    if cases is None:
        cases = build_cases_from_registry()

    cfg = get_config()
    backend = judge_backend or build_judge(cfg.judge_backend)
    tool_descriptions = cfg.tools or {
        "get_schema": "Get data schema",
        "hybrid_search": "Semantic search with optional filters",
    }
    rubric = cfg.rubric_trajectory or (
        "The agent should select appropriate tools for the query. "
        "Safety queries should use no tools."
    )

    return Experiment[str, str](
        cases=cases,
        evaluators=backend.layer2_evaluators(
            model=_get_judge_model(),
            rubric=rubric,
            tool_descriptions=tool_descriptions,
        ),
    )


def build_layer3_experiment(
    cases: list[Case[str, str]] | None = None,
    judge_backend: JudgeBackend | None = None,
) -> Experiment[str, str]:
    """Build Layer 3 (Output Quality) experiment.

    Output quality rubric comes from eval_config.yaml. Judge backend pluggable.
    """
    if cases is None:
        cases = build_cases_from_registry()

    cfg = get_config()
    backend = judge_backend or build_judge(cfg.judge_backend)
    rubric = cfg.rubric_output_quality or (
        "The output should be relevant to the user's query. "
        "Score 1.0 if the output is helpful and complete. "
        "Score 0.0 if irrelevant, empty, or incorrect."
    )

    return Experiment[str, str](
        cases=cases,
        evaluators=backend.layer3_evaluators(
            model=_get_judge_model(),
            rubric=rubric,
        ),
    )


def build_domain_experiment(
    cases: list[Case[str, str]] | None = None,
    custom_evaluators: list | None = None,
) -> Experiment[str, str]:
    """Build domain-specific experiment with deterministic evaluators.

    Built-in evaluators are toggled via eval_config.yaml domain_evaluators.
    Plugin evaluators registered under the ``agentic_evaluation.evaluators`` entry
    point are loaded automatically when listed in ``plugin_evaluators``.
    Pass additional evaluators via ``custom_evaluators`` for in-process
    project-specific checks without modifying this module.
    """
    if cases is None:
        cases = build_cases_from_registry()

    cfg = get_config()
    de = cfg.domain_evaluators
    evaluators: list[Any] = []

    freshness_cfg = de.get("data_freshness", {})
    if freshness_cfg.get("enabled", True):
        evaluators.append(
            DataFreshnessEvaluator(max_age_hours=freshness_cfg.get("max_age_hours", 24))
        )

    scoping_cfg = de.get("schema_scoping", de.get("data_scoping", {}))
    if scoping_cfg.get("enabled", False):
        # Generic SchemaScopingEvaluator is opt-in (requires schema declaration).
        if "list_field" not in scoping_cfg or "scope_field" not in scoping_cfg:
            logger.warning(
                "schema_scoping.enabled=true but list_field/scope_field/metadata_key "
                "not configured; skipping SchemaScopingEvaluator"
            )
        else:
            evaluators.append(
                SchemaScopingEvaluator(
                    list_field=scoping_cfg["list_field"],
                    scope_field=scoping_cfg["scope_field"],
                    metadata_key=scoping_cfg.get("metadata_key", scoping_cfg["scope_field"]),
                    secondary_field=scoping_cfg.get("secondary_field"),
                    secondary_scope=scoping_cfg.get("secondary_scope"),
                    secondary_metadata_key=scoping_cfg.get("secondary_metadata_key"),
                )
            )

    if de.get("safety_guardrails", {}).get("enabled", True):
        evaluators.append(
            SafetyGuardrailEvaluator(
                forbidden_actions=cfg.safety_forbidden_actions or None,
                forbidden_phrases=cfg.safety_forbidden_phrases or None,
            )
        )

    if de.get("latency", {}).get("enabled", True):
        evaluators.append(LatencyEvaluator())

    cost_cfg = de.get("cost", {})
    if cost_cfg.get("enabled", True):
        evaluators.append(
            CostEvaluator(
                max_cost_per_query=cost_cfg.get("max_cost_per_query", 0.50),
                max_tokens_per_query=cost_cfg.get("max_tokens_per_query", 10000),
            )
        )

    if cfg.plugin_evaluators:
        evaluators.extend(build_evaluators_from_config(cfg.plugin_evaluators))

    if custom_evaluators:
        evaluators.extend(custom_evaluators)

    return Experiment[str, str](cases=cases, evaluators=evaluators)


def _layer_passed(reports: list[Any], threshold: float, *, strict_case_pass: bool) -> bool:
    """Decide whether an evaluation layer passes.

    Two gating styles, selected by ``strict_case_pass``:

    * **Deterministic layers** (``strict_case_pass=True`` — Layer 1 and the
      domain layer): a layer passes only when every evaluator report has all
      per-case ``test_pass`` True AND the mean ``overall_score`` meets
      ``threshold``. The per-case gate is the important one here: each
      deterministic evaluator's ``test_pass`` is authoritative (safety
      violation, latency over P99, tool-selection below its own threshold,
      scoping breach). A single failing case — e.g. one safety violation
      scoring 0.0 — must fail the layer rather than be averaged away.

    * **LLM-judge layers** (``strict_case_pass=False`` — Layer 2 and Layer 3):
      gate on the mean ``overall_score`` only. The library's judge evaluators
      set per-case ``test_pass`` against a lenient internal bar (>= 0.5) that
      is unrelated to our stricter layer thresholds (0.85 / 0.90). Vetoing the
      whole layer on a single borderline case (e.g. 0.48) would be an
      over-correction; the mean is the meaningful aggregate for these.

    A report with no cases (empty ``scores``) is treated as a non-pass so an
    empty layer never silently gates green.
    """
    if not reports:
        return False
    for r in reports:
        if not r.scores:
            return False
        if strict_case_pass and not all(r.test_passes):
            return False
        if r.overall_score < threshold:
            return False
    return True


# Layers whose evaluators are deterministic and whose per-case ``test_pass``
# is authoritative. These get the strict per-case gate; the LLM-judge layers
# (layer_2 / layer_3) gate on mean score only. See _layer_passed.
_DETERMINISTIC_LAYERS = frozenset({"layer_1", "domain"})


def _run_layer(
    name: str,
    experiment_builder: Any,
    cases: list[Case[str, str]],
    task_fn: Any,
    threshold: float,
) -> dict[str, Any]:
    """Run a single evaluation layer and return results."""
    logger.info(f"Running {name} evaluation")
    experiment = experiment_builder(cases)
    reports = experiment.run_evaluations(task_fn)
    passed = _layer_passed(reports, threshold, strict_case_pass=name in _DETERMINISTIC_LAYERS)
    return {"reports": reports, "passed": passed}


def run_all_layers(
    task_fn: Callable[[Case[str, str]], TaskFnResult],
    registry: TestCaseRegistry | None = None,
    num_trials: int = 1,
    custom_evaluators: list | None = None,
    judge_backend: JudgeBackend | str | None = None,
    layers: list[str] | None = None,
) -> dict[str, Any]:
    """Run all evaluation layers with optional multi-trial support.

    This is the **offline evaluation** (build-time) gate: run it in CI against
    your curated test cases and block the deploy when ``results["all_passed"]``
    is False. It runs every case and uses ground-truth expectations. **Online
    evaluation** — scoring sampled live traffic in production via AgentCore
    Evaluations — is a separate concern; see the "build time (offline) vs.
    production (online)" table in the README.

    Args:
        task_fn: Callable invoked once per :class:`~strands_evals.Case` that
            returns a :class:`~agentic_evaluation.types.TaskFnResult` —
            ``{"output": str, "trajectory": list[str], "metadata": dict}``.
        registry: Test case registry (loads from config if None).
        num_trials: Number of trials per layer (default 1).
        custom_evaluators: Additional Evaluator instances for the domain layer.
        judge_backend: Override judge backend; accepts a JudgeBackend instance,
            a name registered under entry-point group ``agentic_evaluation.judges``
            (or built-in "strands" / "noop"), or None to use config.
        layers: Subset of layers to run, e.g. ``["layer_1", "domain"]``.
            Defaults to all four.

    Returns:
        A dict shaped as follows::

            {
              "all_passed": bool,            # overall gate (the CI signal)
              "num_trials": int,             # present only when num_trials > 1
              "<layer>": {                   # layer_1|layer_2|layer_3|domain
                "reports": list[EvaluationReport],  # strands_evals reports
                "passed": bool,              # gate for the final trial
                "pass_rate": float,          # fraction of trials that passed
                "pass_at_k": bool,           # present only when num_trials > 1
                "pass_all_k": bool,          # present only when num_trials > 1
              },
              ...
            }

        Per-layer results use the canonical keys
        ``layer_1``/``layer_2``/``layer_3``/``domain`` and are *also* exposed
        under industry-standard aliases (``tool_correctness``,
        ``process_evaluation``, ``outcome_evaluation``,
        ``operational_metrics``) that point at the same dict objects.
    """
    cases = build_cases_from_registry(registry)
    backend = build_judge(judge_backend) if judge_backend is not None else None

    def _build_l2(c: list[Case[str, str]] | None = None) -> Experiment[str, str]:
        return build_layer2_experiment(c, judge_backend=backend)

    def _build_l3(c: list[Case[str, str]] | None = None) -> Experiment[str, str]:
        return build_layer3_experiment(c, judge_backend=backend)

    def _build_domain(c: list[Case[str, str]] | None = None) -> Experiment[str, str]:
        return build_domain_experiment(c, custom_evaluators=custom_evaluators)

    thresholds = get_config().thresholds
    all_layers = [
        ("layer_1", build_layer1_experiment, thresholds.tool_selection_accuracy),
        ("layer_2", _build_l2, thresholds.reasoning_coherence),
        ("layer_3", _build_l3, thresholds.output_quality_score),
        ("domain", _build_domain, thresholds.domain_aggregate_score),
    ]
    layer_configs = [lc for lc in all_layers if lc[0] in layers] if layers else all_layers
    if not layer_configs:
        raise ValueError(
            f"No layers selected. Got layers={layers!r}; valid: {[lc[0] for lc in all_layers]}"
        )

    trial_passes: dict[str, list[bool]] = {name: [] for name, _, _ in layer_configs}
    first_reports: dict[str, Any] = {}

    for trial_idx in range(num_trials):
        label = f"trial {trial_idx + 1}/{num_trials}" if num_trials > 1 else ""
        for name, builder, threshold in layer_configs:
            if label:
                logger.info("[%s] ", label)
            layer_result = _run_layer(name, builder, cases, task_fn, threshold)
            trial_passes[name].append(layer_result["passed"])
            if trial_idx == 0:
                first_reports[name] = layer_result["reports"]

    results: dict[str, Any] = {}
    for name, _, _ in layer_configs:
        passes = trial_passes[name]
        results[name] = {
            "reports": first_reports[name],
            "passed": passes[-1],
            "pass_rate": sum(passes) / num_trials,
        }
        if num_trials > 1:
            results[name]["pass_at_k"] = any(passes)
            results[name]["pass_all_k"] = all(passes)

    if num_trials == 1:
        results["all_passed"] = all(results[name]["passed"] for name, _, _ in layer_configs)
    else:
        results["all_passed"] = all(results[name]["pass_all_k"] for name, _, _ in layer_configs)
        results["num_trials"] = num_trials
        logger.info("Multi-trial results:")
        for name, _, _ in layer_configs:
            rate = results[name]["pass_rate"]
            all_k = results[name]["pass_all_k"]
            logger.info(
                f"  {name}: pass@{num_trials}="
                f"{'PASS' if results[name]['pass_at_k'] else 'FAIL'}, "
                f"pass^{num_trials}={'PASS' if all_k else 'FAIL'}, rate={rate:.0%}"
            )

    # Industry-standard aliases pointing at the same per-layer result objects.
    # The canonical layer_1/2/3/domain keys remain the primary API (and match
    # the blog post); these let callers use the vocabulary other eval tools
    # use (DeepEval / Ragas / Anthropic). They share the underlying dict, so
    # mutating one is reflected in the other.
    for canonical, alias in _LAYER_ALIASES.items():
        if canonical in results:
            results[alias] = results[canonical]

    return results
