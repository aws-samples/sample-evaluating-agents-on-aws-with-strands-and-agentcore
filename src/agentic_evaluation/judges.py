# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Pluggable LLM-judge backends.

JudgeBackend lets Layer 2 / Layer 3 evaluators run against any LLM stack.
Default is StrandsJudgeBackend (uses strands-agents-evals), but users on
non-Strands frameworks can pick NoOpJudgeBackend (skip LLM layers and rely
on Layer 1 + domain evaluators) or register a custom backend via the
``agentic_evaluation.judges`` entry-point group.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol, runtime_checkable

from botocore.config import Config as BotocoreConfig

from agentic_evaluation.exceptions import (
    JudgeUnavailableError,
    PluginLoadError,
    PluginNotFoundError,
)

_JUDGE_ENTRY_POINT_GROUP = "agentic_evaluation.judges"


@runtime_checkable
class JudgeBackend(Protocol):
    """Protocol every LLM-judge backend implements.

    Implementations may be heavy (real LLM calls) or lightweight (no-op).
    They must expose ``layer2_evaluators`` / ``layer3_evaluators`` returning
    a list of strands_evals.Evaluator-compatible instances. ``run_experiment``
    consumes those lists when building Layer 2 and Layer 3 experiments.
    """

    name: str

    def layer2_evaluators(
        self, *, model: str, rubric: str, tool_descriptions: dict[str, str]
    ) -> list[Any]:
        """Return Layer 2 (reasoning) evaluators for the given config."""
        ...

    def layer3_evaluators(self, *, model: str, rubric: str) -> list[Any]:
        """Return Layer 3 (output quality) evaluators for the given config."""
        ...


class StrandsJudgeBackend:
    """Default backend: wraps strands-agents-evals LLM evaluators.

    Imports from ``strands_evals`` are deferred to instantiation so the SDK
    core can be installed without the optional ``[strands]`` extra.
    """

    name = "strands"

    def __init__(
        self,
        *,
        connect_timeout_seconds: int = 5,
        read_timeout_seconds: int = 120,
        max_attempts: int = 3,
    ) -> None:
        try:
            from strands_evals.evaluators import (  # noqa: F401
                GoalSuccessRateEvaluator,
                HelpfulnessEvaluator,
                OutputEvaluator,
                TrajectoryEvaluator,
            )
        except ImportError as exc:  # pragma: no cover - exercised when extra missing
            raise JudgeUnavailableError(
                "StrandsJudgeBackend requires the 'strands' extra. "
                "Install with: pip install 'agentic-evaluation[strands]'"
            ) from exc
        for name, value in (
            ("connect_timeout_seconds", connect_timeout_seconds),
            ("read_timeout_seconds", read_timeout_seconds),
            ("max_attempts", max_attempts),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._boto_client_config = BotocoreConfig(
            connect_timeout=connect_timeout_seconds,
            read_timeout=read_timeout_seconds,
            retries={"max_attempts": max_attempts, "mode": "standard"},
        )

    def _configured_model(self, model_id: str) -> Any:
        """Build one bounded Bedrock model client for an evaluator group."""
        from strands.models import BedrockModel

        return BedrockModel(
            model_id=model_id,
            boto_client_config=self._boto_client_config,
        )

    def layer2_evaluators(
        self, *, model: str, rubric: str, tool_descriptions: dict[str, str]
    ) -> list[Any]:
        from strands_evals.evaluators import HelpfulnessEvaluator, TrajectoryEvaluator

        configured_model = self._configured_model(model)
        return [
            HelpfulnessEvaluator(model=configured_model),
            TrajectoryEvaluator(
                rubric=rubric,
                trajectory_description=tool_descriptions,
                model=configured_model,
            ),
        ]

    def layer3_evaluators(self, *, model: str, rubric: str) -> list[Any]:
        from strands_evals.evaluators import GoalSuccessRateEvaluator, OutputEvaluator

        configured_model = self._configured_model(model)
        return [
            OutputEvaluator(rubric=rubric, model=configured_model),
            GoalSuccessRateEvaluator(model=configured_model),
        ]


def _make_pass_through(label: str) -> Any:
    """Build an Evaluator subclass instance that always returns a pass.

    Subclasses ``strands_evals.evaluators.Evaluator`` so it plugs into the
    same async runner the LLM judges use. Used by NoOpJudgeBackend for
    quickstart demos and CI smoke tests.
    """
    from strands_evals.evaluators import Evaluator
    from strands_evals.types.evaluation import EvaluationOutput

    class _PassThroughEvaluator(Evaluator):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self._label = label

        @classmethod
        def get_type_name(cls) -> str:
            return f"NoOpJudge_{label}"

        def evaluate(self, evaluation_case: Any) -> list[Any]:  # noqa: ARG002
            return [
                EvaluationOutput(
                    score=1.0,
                    test_pass=True,
                    reason=f"judge=noop: {self._label} layer skipped",
                )
            ]

    return _PassThroughEvaluator()


class NoOpJudgeBackend:
    """No-LLM backend: skips Layer 2 and Layer 3 with passing scores.

    Use for quickstart demos, offline CI, or projects that only care about
    Layer 1 (deterministic tool usage) and domain evaluators.
    """

    name = "noop"

    def layer2_evaluators(
        self,
        *,
        model: str,  # noqa: ARG002
        rubric: str,  # noqa: ARG002
        tool_descriptions: dict[str, str],  # noqa: ARG002
    ) -> list[Any]:
        return [_make_pass_through("layer_2")]

    def layer3_evaluators(
        self,
        *,
        model: str,  # noqa: ARG002
        rubric: str,  # noqa: ARG002
    ) -> list[Any]:
        return [_make_pass_through("layer_3")]


def _discover_entry_points() -> dict[str, EntryPoint]:
    """Return registered judge backends keyed by name."""
    return {ep.name: ep for ep in entry_points(group=_JUDGE_ENTRY_POINT_GROUP)}


def build_judge(name: str | JudgeBackend | None = None) -> JudgeBackend:
    """Resolve a judge backend by name.

    - ``None`` or ``"strands"``: default StrandsJudgeBackend.
    - ``"noop"``: NoOpJudgeBackend (no LLM calls).
    - any other string: looked up via the ``agentic_evaluation.judges`` entry-point group.
    - JudgeBackend instance: returned as-is.
    """
    if name is None:
        return StrandsJudgeBackend()
    if isinstance(name, JudgeBackend) and not isinstance(name, str):
        return name  # type: ignore[return-value]
    if name == "strands":
        return StrandsJudgeBackend()
    if name == "noop":
        return NoOpJudgeBackend()

    eps = _discover_entry_points()
    if name not in eps:
        raise PluginNotFoundError(name, available=sorted(eps))
    try:
        cls = eps[name].load()
    except Exception as exc:  # noqa: BLE001
        raise PluginLoadError(name, exc) from exc
    return cls()
