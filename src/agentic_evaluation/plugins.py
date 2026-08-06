# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom-evaluator plugin loader.

Third parties register their Evaluator subclasses under the entry-point
group ``agentic_evaluation.evaluators``. They can then be referenced by name in
``eval_config.yaml`` under ``plugin_evaluators`` and the SDK will load
and instantiate them automatically.

Example pyproject.toml::

    [project.entry-points."agentic_evaluation.evaluators"]
    my_compliance = "my_pkg.evaluators:ComplianceEvaluator"

Example eval_config.yaml::

    plugin_evaluators:
      - name: my_compliance
        params:
          policy_id: "PCI-DSS"
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any

from agentic_evaluation.exceptions import PluginLoadError, PluginNotFoundError

_EVALUATOR_ENTRY_POINT_GROUP = "agentic_evaluation.evaluators"


def _discover_entry_points() -> dict[str, EntryPoint]:
    return {ep.name: ep for ep in entry_points(group=_EVALUATOR_ENTRY_POINT_GROUP)}


def load_evaluator_plugin(name: str) -> type:
    """Resolve an Evaluator class by entry-point name.

    Raises:
        PluginNotFoundError: when no entry-point with that name is registered.
        PluginLoadError: when the entry-point exists but importing it fails
            (e.g. missing transitive dependency).
    """
    eps = _discover_entry_points()
    if name not in eps:
        raise PluginNotFoundError(name, available=sorted(eps))
    try:
        return eps[name].load()
    except Exception as exc:
        raise PluginLoadError(name, exc) from exc


def build_evaluators_from_config(plugin_specs: list[dict[str, Any]]) -> list[Any]:
    """Instantiate every plugin evaluator listed in ``cfg.plugin_evaluators``.

    Each spec must be a ``{"name": str, "params": dict}`` mapping. ``params``
    is optional and forwarded as ``**kwargs`` to the evaluator class.
    """
    evaluators: list[Any] = []
    for spec in plugin_specs:
        if "name" not in spec:
            raise PluginLoadError("<unnamed>", ValueError("plugin spec missing 'name' key"))
        cls = load_evaluator_plugin(spec["name"])
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise PluginLoadError(
                spec["name"], ValueError(f"'params' must be a mapping, got {type(params).__name__}")
            )
        evaluators.append(cls(**params))
    return evaluators
