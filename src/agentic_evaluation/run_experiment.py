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
import uuid
from collections.abc import Callable
from copy import deepcopy
from functools import cache
from typing import Any

from strands_evals import Case, Experiment
from strands_evals.types.trace import Session

from agentic_evaluation.config import EvalConfig, load_config
from agentic_evaluation.evaluators import (
    CostEvaluator,
    DataFreshnessEvaluator,
    LatencyEvaluator,
    SafetyGuardrailEvaluator,
    SchemaScopingEvaluator,
    SecondaryScope,
    ToolParameterGrader,
    ToolSelectionGrader,
    TrajectoryOrderGrader,
)
from agentic_evaluation.judges import JudgeBackend, build_judge
from agentic_evaluation.plugins import build_evaluators_from_config
from agentic_evaluation.test_cases import TestCaseRegistry, TestCategory
from agentic_evaluation.types import TaskFnResult

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
_CASE_LAYER_VALUES = {
    "layer_1": "layer_1_tool_usage",
    "layer_2": "layer_2_reasoning",
    "layer_3": "layer_3_output_quality",
}

# One layer selected for a run: its canonical name, the experiment builder to
# call with that layer's cases, and the score threshold(s) its evaluators must
# clear (one float shared by all, or one per evaluator in declaration order).
LayerPlan = tuple[str, Callable[..., Experiment[str, str]], float | list[float]]


@cache
def get_config() -> EvalConfig:
    """Get the evaluation config, loading from YAML on first use.

    Returns:
        The parsed config. Cached, so the YAML is read at most once per process
        until :func:`reset_config_cache` is called.
    """
    return load_config()


def reset_config_cache() -> None:
    """Drop the cached config so the next :func:`get_config` re-reads the YAML.

    Used by the CLI after setting ``EVAL_CONFIG_PATH``, and by tests that point
    at a fixture config.
    """
    get_config.cache_clear()


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

    cases: list[Case[str, str]] = []
    for tc in test_cases:
        layer_values = [layer.value for layer in tc.evaluation_layers]
        base_metadata = {
            "category": tc.category.value,
            "tags": tc.tags,
            "evaluation_layers": layer_values,
            "source_case_id": tc.id,
            "conversation_id": tc.id if tc.reference_solution else None,
            "turn_index": 1,
            "expected_tool_parameters": tc.expected_tool_parameters,
        }
        turns: list[tuple[str, str, list[str], str | None, dict[str, Any] | None]] = [
            (
                tc.id,
                tc.query,
                tc.expected_tools,
                tc.expected_behavior,
                tc.expected_tool_parameters,
            )
        ]
        reference = tc.reference_solution or {}
        turn_index = 2
        while f"turn_{turn_index}_query" in reference:
            turns.append(
                (
                    f"{tc.id}__turn_{turn_index}",
                    str(reference[f"turn_{turn_index}_query"]),
                    list(reference.get(f"turn_{turn_index}_expected_tools", [])),
                    reference.get(f"turn_{turn_index}_behavior"),
                    reference.get(f"turn_{turn_index}_expected_parameters"),
                )
            )
            turn_index += 1

        turn_count = len(turns)
        for index, (name, query, tools, behavior, parameters) in enumerate(turns, start=1):
            metadata = {
                **base_metadata,
                "turn_index": index,
                "turn_count": turn_count,
                "expected_tool_parameters": parameters,
            }
            cases.append(
                Case[str, str](
                    name=name,
                    input=query,
                    expected_output=behavior or tc.expected_behavior,
                    expected_trajectory=tools,
                    expected_assertion=(
                        tc.expected_assertion if index == 1 else behavior or tc.expected_assertion
                    ),
                    metadata=metadata,
                )
            )
    return cases


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
            ToolParameterGrader(threshold=thresholds.tool_parameter_accuracy),
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


def _secondary_scope(scoping_cfg: dict[str, Any]) -> SecondaryScope | None:
    """Build the optional secondary-scope check from a ``schema_scoping`` block.

    The three ``secondary_*`` keys are only meaningful together, so this is the
    boundary where a partial declaration is resolved to "no secondary check"
    rather than carried into the evaluator as three loose optionals.

    Args:
        scoping_cfg: The ``domain_evaluators.schema_scoping`` config block.

    Returns:
        A :class:`SecondaryScope` when all three keys are set, else None.
    """
    field = scoping_cfg.get("secondary_field")
    scope_field = scoping_cfg.get("secondary_scope")
    metadata_key = scoping_cfg.get("secondary_metadata_key")
    if field and scope_field and metadata_key:
        return SecondaryScope(field=field, scope_field=scope_field, metadata_key=metadata_key)
    return None


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
                    secondary=_secondary_scope(scoping_cfg),
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


def _layer_passed(
    reports: list[Any], threshold: float | list[float], *, strict_case_pass: bool
) -> bool:
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
    report_thresholds = (
        [float(threshold)] * len(reports) if isinstance(threshold, (int, float)) else threshold
    )
    if len(report_thresholds) < len(reports):
        report_thresholds = [
            *report_thresholds,
            *([report_thresholds[-1]] * (len(reports) - len(report_thresholds))),
        ]
    else:
        report_thresholds = report_thresholds[: len(reports)]
    # strict=True documents the invariant established immediately above:
    # report_thresholds has been padded or truncated to exactly len(reports).
    for r, report_threshold in zip(reports, report_thresholds, strict=True):
        if not r.scores:
            return False
        if strict_case_pass and not all(r.test_passes):
            return False
        if r.overall_score < report_threshold:
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
    threshold: float | list[float],
) -> dict[str, Any]:
    """Run a single evaluation layer and return results."""
    logger.info("Running %s evaluation", name)
    experiment = experiment_builder(cases)
    reports = experiment.run_evaluations(task_fn)
    passed = _layer_passed(reports, threshold, strict_case_pass=name in _DETERMINISTIC_LAYERS)
    logger.info(
        "Completed %s evaluation: passed=%s scores=%s",
        name,
        passed,
        [round(report.overall_score, 3) for report in reports],
    )
    return {"reports": reports, "passed": passed}


def _cases_for_layer(cases: list[Case[str, str]], layer_name: str) -> list[Case[str, str]]:
    """Return cases explicitly routed to a layer; domain metrics apply to all."""
    if layer_name == "domain":
        return cases
    configured_value = _CASE_LAYER_VALUES[layer_name]
    selected = []
    for case in cases:
        configured = (case.metadata or {}).get("evaluation_layers", [])
        if not configured or configured_value in configured:
            selected.append(case)
    return selected


def _case_key(case: Case[str, str]) -> str:
    """Identify a case for the duration of one trial.

    Args:
        case: The case being executed.

    Returns:
        The cache and bookkeeping key. ``Case.name`` is optional upstream, so
        coercing here keeps the per-trial cache and the priming set in agreement
        about what identifies a case.
    """
    return str(case.name)


def _make_cached_task_fn(
    task_fn: Callable[[Case[str, str]], TaskFnResult],
    run_artifacts: dict[str, TaskFnResult],
    conversation_traces: dict[str, list[Any]],
) -> Callable[[Case[str, str]], TaskFnResult]:
    """Wrap ``task_fn`` so each case executes at most once per trial.

    Every layer in a trial evaluates the same cases, so the raw ``task_fn``
    would be invoked once per layer per case. This memoises the first result in
    ``run_artifacts`` and hands out deep copies, so layers cannot observe each
    other's mutations. For cases carrying a ``conversation_id``, traces are
    accumulated across turns in ``conversation_traces`` so a multi-turn
    trajectory is evaluated as a whole.

    The caches are passed in rather than captured from an enclosing loop: one
    instance belongs to exactly one trial, and binding them as parameters makes
    that ownership explicit instead of dependent on rebinding order.

    Args:
        task_fn: The user's task function.
        run_artifacts: Per-trial cache, keyed by case name.
        conversation_traces: Per-trial trace accumulator, keyed by conversation.

    Returns:
        A single-argument callable with the same shape as ``task_fn``.
    """

    def cached_task_fn(case: Case[str, str]) -> TaskFnResult:
        key = _case_key(case)
        if key not in run_artifacts:
            artifact = deepcopy(task_fn(case))
            conversation_id = (case.metadata or {}).get("conversation_id")
            trajectory = artifact.get("trajectory")
            if conversation_id and isinstance(trajectory, Session):
                traces = conversation_traces.setdefault(str(conversation_id), [])
                traces.extend(deepcopy(trajectory.traces))
                trajectory.traces = deepcopy(traces)
            run_artifacts[key] = artifact
        return deepcopy(run_artifacts[key])

    return cached_task_fn


def _layer_plans(
    judge_backend: JudgeBackend | str | None,
    custom_evaluators: list | None,
    layers: list[str] | None,
) -> list[LayerPlan]:
    """Select and configure the layers a run will execute.

    Args:
        judge_backend: Judge override passed through to the LLM layers; None
            leaves each builder to read the config.
        custom_evaluators: Extra evaluators for the domain layer.
        layers: Subset of canonical layer names, or None for all four.

    Returns:
        One plan per selected layer, in canonical order.

    Raises:
        ValueError: ``layers`` matched none of the known layer names.
    """
    backend = build_judge(judge_backend) if judge_backend is not None else None

    def _build_l2(c: list[Case[str, str]] | None = None) -> Experiment[str, str]:
        return build_layer2_experiment(c, judge_backend=backend)

    def _build_l3(c: list[Case[str, str]] | None = None) -> Experiment[str, str]:
        return build_layer3_experiment(c, judge_backend=backend)

    def _build_domain(c: list[Case[str, str]] | None = None) -> Experiment[str, str]:
        return build_domain_experiment(c, custom_evaluators=custom_evaluators)

    thresholds = get_config().thresholds
    all_layers: list[LayerPlan] = [
        (
            "layer_1",
            build_layer1_experiment,
            [
                thresholds.tool_selection_accuracy,
                thresholds.tool_parameter_accuracy,
                thresholds.reasoning_coherence,
            ],
        ),
        (
            "layer_2",
            _build_l2,
            [thresholds.helpfulness_score, thresholds.reasoning_coherence],
        ),
        (
            "layer_3",
            _build_l3,
            [thresholds.output_quality_score, thresholds.goal_success_rate],
        ),
        ("domain", _build_domain, thresholds.domain_aggregate_score),
    ]
    plans = [plan for plan in all_layers if plan[0] in layers] if layers else all_layers
    if not plans:
        raise ValueError(
            f"No layers selected. Got layers={layers!r}; valid: {[p[0] for p in all_layers]}"
        )
    return plans


def _prime_task_fn(
    cached_task_fn: Callable[[Case[str, str]], TaskFnResult],
    trial_cases: list[Case[str, str]],
    execution_names: set[str],
) -> None:
    """Execute every case some layer will evaluate, exactly once, in case order.

    Priming up front does two things the per-layer runs cannot: it guarantees one
    agent invocation per case per trial (later layers hit the cache), and it
    fixes conversational turn order before any layer evaluates, so a multi-turn
    trajectory is assembled in the order the cases declare.

    Args:
        cached_task_fn: The memoising wrapper from :func:`_make_cached_task_fn`.
        trial_cases: This trial's cases, in declaration order.
        execution_names: Names of the cases at least one layer will evaluate.
    """
    case_count = sum(_case_key(case) in execution_names for case in trial_cases)
    case_index = 0
    for case in trial_cases:
        if _case_key(case) in execution_names:
            case_index += 1
            logger.info("Executing evaluation case %d/%d: %s", case_index, case_count, case.name)
            cached_task_fn(case)
            logger.info("Completed evaluation case %d/%d: %s", case_index, case_count, case.name)


def _run_trial(
    task_fn: Callable[[Case[str, str]], TaskFnResult],
    cases: list[Case[str, str]],
    plans: list[LayerPlan],
) -> dict[str, dict[str, Any]]:
    """Run every selected layer once against a fresh copy of the cases.

    Each trial gets its own deep copy of the cases, its own
    ``evaluation_run_id`` and its own task_fn cache, so trials cannot observe
    one another.

    Args:
        task_fn: The user's task function.
        cases: The cases to copy for this trial.
        plans: The layers to run.

    Returns:
        Per-layer ``{"reports": ..., "passed": ...}``, keyed by layer name. A
        layer with no routed cases reports ``passed=False`` rather than
        vacuously passing.
    """
    trial_cases = deepcopy(cases)
    run_id = uuid.uuid4().hex
    for case in trial_cases:
        case.metadata = {**(case.metadata or {}), "evaluation_run_id": run_id}

    layer_cases = {name: _cases_for_layer(trial_cases, name) for name, _, _ in plans}
    execution_names = {
        _case_key(case) for selected_cases in layer_cases.values() for case in selected_cases
    }
    run_artifacts: dict[str, TaskFnResult] = {}
    conversation_traces: dict[str, list[Any]] = {}
    cached_task_fn = _make_cached_task_fn(task_fn, run_artifacts, conversation_traces)
    _prime_task_fn(cached_task_fn, trial_cases, execution_names)

    trial_result: dict[str, dict[str, Any]] = {}
    for name, builder, threshold in plans:
        selected_cases = layer_cases[name]
        if not selected_cases:
            trial_result[name] = {"reports": [], "passed": False}
        else:
            trial_result[name] = _run_layer(
                name, builder, selected_cases, cached_task_fn, threshold
            )
    return trial_result


def _log_multi_trial(results: dict[str, Any], plans: list[LayerPlan], num_trials: int) -> None:
    """Log the pass@k / pass^k summary for a multi-trial run.

    Args:
        results: The aggregated results, already carrying the multi-trial keys.
        plans: The layers that ran.
        num_trials: How many trials were run.
    """
    logger.info("Multi-trial results:")
    for name, _, _ in plans:
        logger.info(
            "  %s: pass@%d=%s, pass^%d=%s, rate=%.0f%%",
            name,
            num_trials,
            "PASS" if results[name]["pass_at_k"] else "FAIL",
            num_trials,
            "PASS" if results[name]["pass_all_k"] else "FAIL",
            results[name]["pass_rate"] * 100,
        )


def _aggregate_trials(
    trial_results: list[dict[str, dict[str, Any]]],
    plans: list[LayerPlan],
) -> dict[str, Any]:
    """Fold per-trial layer results into the run's final result dict.

    Args:
        trial_results: One entry per trial, as returned by :func:`_run_trial`.
        plans: The layers that ran.

    Returns:
        The public result dict documented on :func:`run_all_layers`, including
        the industry-standard aliases.
    """
    num_trials = len(trial_results)
    results: dict[str, Any] = {}
    for name, _, _ in plans:
        passes = [trial[name]["passed"] for trial in trial_results]
        results[name] = {
            # Reports come from the first trial: they are a sample of what the
            # agent produced, while the gate below considers every trial.
            "reports": trial_results[0][name]["reports"],
            "passed": passes[-1],
            "pass_rate": sum(passes) / num_trials,
        }
        if num_trials > 1:
            results[name]["pass_at_k"] = any(passes)
            results[name]["pass_all_k"] = all(passes)

    # One trial gates on that trial; several gate on every trial passing.
    gate = "passed" if num_trials == 1 else "pass_all_k"
    results["all_passed"] = all(results[name][gate] for name, _, _ in plans)
    if num_trials > 1:
        results["num_trials"] = num_trials
        _log_multi_trial(results, plans, num_trials)

    # Industry-standard aliases pointing at the same per-layer result objects.
    # The canonical layer_1/2/3/domain keys remain the primary API (and match
    # the blog post); these let callers use the vocabulary other eval tools
    # use (DeepEval / Ragas / Anthropic). They share the underlying dict, so
    # mutating one is reflected in the other.
    for canonical, alias in _LAYER_ALIASES.items():
        if canonical in results:
            results[alias] = results[canonical]
    return results


# PLR0913 (6 > 5 args): the SDK's primary entry point, and every argument is an
# independent, documented knob that callers pass by name. Grouping them into a
# config object would move the same six values behind one more indirection and
# break every caller and code sample.
def run_all_layers(  # noqa: PLR0913
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

    Raises:
        ValueError: ``num_trials`` is below 1, or ``layers`` selected no layer.
    """
    if num_trials < 1:
        raise ValueError(f"num_trials must be at least 1, got {num_trials}")

    cases = build_cases_from_registry(registry)
    plans = _layer_plans(judge_backend, custom_evaluators, layers)

    trial_results = []
    for trial_idx in range(num_trials):
        if num_trials > 1:
            logger.info("Starting trial %d/%d", trial_idx + 1, num_trials)
        trial_results.append(_run_trial(task_fn, cases, plans))

    return _aggregate_trials(trial_results, plans)
