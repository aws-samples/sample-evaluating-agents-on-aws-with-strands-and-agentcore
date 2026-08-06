# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SDK exceptions.

Centralised error types so callers can distinguish configuration mistakes
(actionable by the user) from internal failures.
"""

from __future__ import annotations


class EvaluationError(Exception):
    """Base class for all SDK errors."""


class ConfigError(EvaluationError):
    """Raised when eval_config.yaml is malformed or references unknown features."""


class PluginNotFoundError(EvaluationError):
    """Raised when a YAML config references an entry-point name that isn't installed."""

    def __init__(self, name: str, available: list[str] | None = None) -> None:
        """Build the error, listing what *is* installed to make the typo obvious.

        Args:
            name: The plugin name the config asked for.
            available: Names actually registered under the entry-point group.
        """
        self.name = name
        self.available = available or []
        msg = f"Evaluator plugin {name!r} not found"
        if self.available:
            msg += f". Installed plugins: {', '.join(self.available)}"
        else:
            msg += ". No plugins are registered under entry-point group 'agentic_evaluation.evaluators'."
        super().__init__(msg)


class PluginLoadError(EvaluationError):
    """Raised when an entry-point is registered but importing it fails."""

    def __init__(self, name: str, original: BaseException) -> None:
        """Wrap the underlying import failure, preserving it for the traceback.

        Args:
            name: The plugin name whose import failed.
            original: The exception raised while loading the entry point.
        """
        self.name = name
        self.original = original
        super().__init__(f"Failed to load plugin {name!r}: {type(original).__name__}: {original}")


class JudgeUnavailableError(EvaluationError):
    """Raised when a JudgeBackend is requested but its dependencies aren't installed."""


class TaskFnError(EvaluationError):
    """Raised when the user-provided task_fn raises or returns a malformed result."""
