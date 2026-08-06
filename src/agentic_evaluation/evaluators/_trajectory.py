# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Trajectory and metrics readers shared by the deterministic evaluators.

Every evaluator in this package reads the agent's behaviour through one of the
three functions here rather than touching ``EvaluationData`` fields directly.
That keeps the "which shape did the task_fn return?" question answered in one
place: the readers accept every trajectory shape ``strands_evals`` may hand us,
so an evaluator never has to branch on it.

Private module: these are an implementation detail of
:mod:`agentic_evaluation.evaluators`, not part of the SDK's public surface.
"""

from typing import Any

from strands_evals.types.evaluation import EvaluationData
from strands_evals.types.trace import Session, ToolExecutionSpan


def extract_tool_names(trajectory: Any) -> list[str]:
    """Extract tool names from either a ``list[str]`` or a ``Session`` object.

    Deterministic evaluators store trajectories as ``list[str]`` for simplicity.
    LLM-based evaluators (HelpfulnessEvaluator, GoalSuccessRateEvaluator)
    require a Session object. This helper lets our custom evaluators work
    with both formats so ``run_all_layers()`` can use a single task function.

    Args:
        trajectory: A ``list[str]`` of tool names, a ``Session``, or None.

    Returns:
        The tool names in call order; empty for an unrecognised or absent
        trajectory.
    """
    if trajectory is None:
        return []
    if isinstance(trajectory, list):
        return [t for t in trajectory if isinstance(t, str)]
    if isinstance(trajectory, Session):
        return [
            span.tool_call.name
            for trace in trajectory.traces
            for span in trace.spans
            if isinstance(span, ToolExecutionSpan)
        ]
    return []


def extract_tool_calls(trajectory: Any) -> list[tuple[str, dict[str, Any]]]:
    """Extract ordered tool names and arguments from supported trajectories.

    Args:
        trajectory: A ``list[dict]`` of ``{"name": ..., "arguments": ...}``
            entries, or a ``Session``.

    Returns:
        ``(tool_name, arguments)`` pairs in call order; empty for an
        unrecognised or absent trajectory.
    """
    if isinstance(trajectory, list):
        calls = []
        for item in trajectory:
            if isinstance(item, dict):
                name = str(item.get("name", ""))
                arguments = item.get("arguments", {})
                calls.append((name, arguments if isinstance(arguments, dict) else {}))
        return calls
    if isinstance(trajectory, Session):
        if not trajectory.traces:
            return []
        # Multi-turn sessions accumulate traces; parameter expectations belong
        # to the current turn, which is always the final trace.
        return [
            (span.tool_call.name, span.tool_call.arguments or {})
            for span in trajectory.traces[-1].spans
            if isinstance(span, ToolExecutionSpan)
        ]
    return []


def case_metrics(evaluation_case: EvaluationData[Any, Any]) -> dict[str, Any]:
    """Read per-turn operational metrics for the case.

    ``strands_evals`` drops a task's result ``metadata`` but propagates
    ``environment_state``, so live adapters (e.g. Bedrock AgentCore) surface
    latency/token/cost there under an ``EnvironmentState(name="metrics")``.
    Prefer that; fall back to ``metadata`` for cases constructed directly with
    ``metadata=`` (tests, static fixtures).

    Args:
        evaluation_case: The case whose measured metrics to read.

    Returns:
        The metrics mapping, or an empty dict when neither channel carries one.
    """
    for env in evaluation_case.actual_environment_state or []:
        if getattr(env, "name", None) == "metrics" and isinstance(env.state, dict):
            return env.state
    return evaluation_case.metadata or {}
