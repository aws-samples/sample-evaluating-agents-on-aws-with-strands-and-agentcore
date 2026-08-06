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
        """Verify the optional extra is present and bound the Bedrock client.

        Args:
            connect_timeout_seconds: Bedrock connection timeout.
            read_timeout_seconds: Bedrock read timeout; judge calls are slow, so
                this is generously larger than the connect timeout.
            max_attempts: Total botocore attempts, standard retry mode.

        Raises:
            JudgeUnavailableError: The ``[strands]`` extra is not installed.
            ValueError: Any timeout or attempt count is not a positive integer.
        """
        try:
            # Local import: probes for the optional `[strands]` extra so the
            # failure surfaces here as JudgeUnavailableError rather than at
            # module import, letting the SDK core install without it.
            from strands_evals.evaluators import (  # noqa: F401, PLC0415
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
        # Local import: part of the optional `[strands]` extra, verified in
        # __init__. See the class docstring.
        from strands.models import BedrockModel  # noqa: PLC0415

        return BedrockModel(
            model_id=model_id,
            boto_client_config=self._boto_client_config,
        )

    def layer2_evaluators(
        self, *, model: str, rubric: str, tool_descriptions: dict[str, str]
    ) -> list[Any]:
        """Build the Layer 2 (reasoning quality) LLM evaluators.

        Args:
            model: Bedrock model ID for the judge.
            rubric: Trajectory rubric describing good tool selection.
            tool_descriptions: Tool name to description, given to the judge as
                the vocabulary for reasoning about the trajectory.

        Returns:
            A helpfulness evaluator and a trajectory evaluator, both bound to
            the same bounded Bedrock client.
        """
        # Local import: optional `[strands]` extra. See the class docstring.
        from strands_evals.evaluators import (  # noqa: PLC0415
            HelpfulnessEvaluator,
            TrajectoryEvaluator,
        )

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
        """Build the Layer 3 (output quality) LLM evaluators.

        Args:
            model: Bedrock model ID for the judge.
            rubric: Output-quality rubric describing a good final answer.

        Returns:
            An output evaluator and a goal-success-rate evaluator, both bound to
            the same bounded Bedrock client.
        """
        # Local import: optional `[strands]` extra. See the class docstring.
        from strands_evals.evaluators import (  # noqa: PLC0415
            GoalSuccessRateEvaluator,
            OutputEvaluator,
        )

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
    # Local imports: the base class and output type come from the optional
    # `[strands]` extra, and the subclass cannot be declared without them.
    from strands_evals.evaluators import Evaluator  # noqa: PLC0415
    from strands_evals.types.evaluation import EvaluationOutput  # noqa: PLC0415

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
        """Return a single always-pass evaluator standing in for Layer 2.

        Args:
            model: Ignored; kept to satisfy the JudgeBackend protocol.
            rubric: Ignored; kept to satisfy the JudgeBackend protocol.
            tool_descriptions: Ignored; kept to satisfy the JudgeBackend protocol.

        Returns:
            One pass-through evaluator labelled ``layer_2``.
        """
        return [_make_pass_through("layer_2")]

    def layer3_evaluators(
        self,
        *,
        model: str,  # noqa: ARG002
        rubric: str,  # noqa: ARG002
    ) -> list[Any]:
        """Return a single always-pass evaluator standing in for Layer 3.

        Args:
            model: Ignored; kept to satisfy the JudgeBackend protocol.
            rubric: Ignored; kept to satisfy the JudgeBackend protocol.

        Returns:
            One pass-through evaluator labelled ``layer_3``.
        """
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
    except Exception as exc:
        raise PluginLoadError(name, exc) from exc
    return cls()
