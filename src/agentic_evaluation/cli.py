# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Command-line interface: ``agentic-eval`` / ``python -m agentic_evaluation``.

Subcommands:

- ``run``      Run all (or a subset of) layers against a user task_fn.
- ``init``     Scaffold a new project's eval_config.yaml + task_fn stub.
- ``validate`` Lint an eval_config.yaml without running anything.

Exit codes:
    0 — success / all layers passed
    1 — evaluation completed but some layers failed
    2 — user error (bad config / task_fn / arguments)
    3 — internal error (unexpected exception)
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from agentic_evaluation.config import load_config
from agentic_evaluation.exceptions import (
    ConfigError,
    JudgeUnavailableError,
    PluginLoadError,
    PluginNotFoundError,
    TaskFnError,
)
from agentic_evaluation.judges import build_judge

logger = logging.getLogger("agentic_evaluation.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_task_fn(spec: str) -> Callable[..., Any]:
    """Import ``module:attr`` and verify the attribute is callable.

    Raises a ConfigError with an actionable message on failure.
    """
    if ":" not in spec:
        raise ConfigError(
            f"--task-fn must be of the form 'module.path:function_name', got {spec!r}"
        )
    module_path, attr = spec.split(":", 1)
    # Make the user's working directory importable so a sibling task_fn.py resolves
    # without requiring PYTHONPATH. Mirrors pytest's rootdir / hatch / alembic behavior.
    import os

    cwd = os.getcwd()  # cwd: current working directory
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ConfigError(
            f"Module {module_path!r} not found. Looked in: cwd ({cwd}) and PYTHONPATH. ({exc})"
        ) from exc
    if not hasattr(module, attr):
        public = sorted(n for n in dir(module) if not n.startswith("_"))
        raise ConfigError(
            f"Module {module_path!r} has no attribute {attr!r}. Public attributes: {public[:20]}"
        )
    fn = getattr(module, attr)
    if not callable(fn):
        raise ConfigError(f"{spec!r} is not callable (got {type(fn).__name__})")
    return fn


def _setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from agentic_evaluation.run_experiment import reset_config_cache, run_all_layers

    if args.config:
        # load_config caches via run_experiment; reset so --config takes effect
        import os

        os.environ["EVAL_CONFIG_PATH"] = str(args.config)
        reset_config_cache()

    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("Config error: %s", exc)
        return 2

    try:
        task_fn = _import_task_fn(args.task_fn)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    try:
        judge_backend = build_judge(args.judge or cfg.judge_backend)
    except (PluginNotFoundError, PluginLoadError, JudgeUnavailableError) as exc:
        logger.error("Judge backend error: %s", exc)
        return 2

    layers = (
        [layer.strip() for layer in args.layers.split(",") if layer.strip()]
        if args.layers
        else None
    )

    try:
        results = run_all_layers(
            task_fn=task_fn,
            num_trials=args.trials,
            judge_backend=judge_backend,
            layers=layers,
        )
    except TaskFnError as exc:
        logger.error("task_fn failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Internal error during evaluation: %s", exc)
        return 3

    if args.output:
        payload = _build_output_payload(results, cfg, detail=args.detail)
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s results to %s", args.detail, args.output)

    print(_format_results(results))
    return 0 if results.get("all_passed") else 1


def _serialize_report(report: Any) -> dict[str, Any]:
    """Convert an EvaluationReport into a JSON-friendly dict.

    Reports come from strands_evals and expose: ``evaluator_name``,
    ``overall_score``, plus per-case lists ``cases``, ``scores``,
    ``test_passes``, ``reasons``.
    """
    name = getattr(report, "evaluator_name", "unknown")
    overall = float(getattr(report, "overall_score", 0.0))
    cases = getattr(report, "cases", []) or []
    scores = getattr(report, "scores", []) or []
    passes = getattr(report, "test_passes", []) or []
    reasons = getattr(report, "reasons", []) or []

    case_entries: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        if isinstance(case, dict):
            case_id = case.get("name") or case.get("input") or f"case_{i}"
        else:
            case_id = str(case)
        case_entries.append(
            {
                "id": str(case_id)[:200],
                "score": float(scores[i]) if i < len(scores) else None,
                "passed": bool(passes[i]) if i < len(passes) else None,
                "reason": reasons[i] if i < len(reasons) else None,
            }
        )
    return {"name": name, "overall": overall, "cases": case_entries}


def _build_output_payload(results: dict[str, Any], cfg: Any, detail: str) -> dict[str, Any]:
    """Build the JSON output payload at the requested detail level."""
    from datetime import datetime, timezone

    payload: dict[str, Any] = {
        "project": cfg.project_name,
        "run_id": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "all_passed": results.get("all_passed"),
        "layers": {},
    }
    for name in ("layer_1", "layer_2", "layer_3", "domain"):
        if name not in results:
            continue
        info = results[name]
        layer_entry: dict[str, Any] = {
            "passed": info.get("passed"),
            "pass_rate": info.get("pass_rate"),
        }
        if detail == "full":
            layer_entry["evaluators"] = [_serialize_report(r) for r in info.get("reports", [])]
        payload["layers"][name] = layer_entry
    return payload


def _format_results(results: dict[str, Any]) -> str:
    lines = ["", "Evaluation results", "==================="]
    for name in ("layer_1", "layer_2", "layer_3", "domain"):
        if name not in results:
            continue
        info = results[name]
        status = "PASS" if info.get("passed") else "FAIL"
        rate = info.get("pass_rate", 1.0)
        lines.append(f"  {name:<10} {status:<5}  pass_rate={rate:.0%}")
    lines.append("")
    lines.append("Overall: " + ("PASS" if results.get("all_passed") else "FAIL"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand: init
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Scaffold eval_config.yaml + a task_fn stub for a new project."""
    project_dir = Path(args.directory or ".")
    project_dir.mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "eval_config.yaml"
    if config_path.exists() and not args.force:
        logger.error("%s already exists. Re-run with --force to overwrite.", config_path)
        return 2

    tools_block = "\n".join(
        f'  {tool}:\n    description: "TODO: describe {tool}"'
        for tool in (args.tools or "search,answer").split(",")
    )

    config_path.write_text(f"""# Generated by `agentic-eval init`
project:
  name: "{args.name}"
  description: "TODO"
  region: "{args.region}"
  environment: "dev"

judge_backend: "{args.judge}"
judge_model: "anthropic.claude-sonnet-4-6"
judge_region_prefix: ""

tools:
{tools_block}

rubrics:
  trajectory: |
    TODO: describe what tool selection should look like.
  output_quality: |
    TODO: describe what a great answer looks like.

safety:
  forbidden_actions: []
  forbidden_phrases: []

domain_evaluators:
  data_freshness: {{enabled: false}}
  schema_scoping: {{enabled: false}}
  safety_guardrails: {{enabled: true}}
  latency: {{enabled: true}}
  cost: {{enabled: true, max_cost_per_query: 0.50, max_tokens_per_query: 10000}}

test_cases:
  - id: "hp_001"
    query: "TODO: a representative happy-path query"
    category: "happy_path"
    expected_tools: ["search"]
    expected_behavior: "TODO"
    evaluation_layers: ["layer_1_tool_usage", "layer_3_output_quality"]
    tags: []
""")

    task_fn_path = project_dir / "task_fn.py"
    if not task_fn_path.exists() or args.force:
        task_fn_path.write_text(
            '''"""Implement task_fn(case) for your agent and point the CLI at it:

    agentic-eval run --config eval_config.yaml --task-fn task_fn:run
"""

from typing import Any


def run(case: Any) -> dict[str, Any]:
    # TODO: call your agent with case.input and return the result
    return {
        "output": "TODO: replace with real agent output",
        "trajectory": [],
        "metadata": {"latency_ms": 0},
    }
'''
        )

    logger.info("Wrote %s and %s", config_path, task_fn_path)
    print(f"Initialized project '{args.name}' in {project_dir.resolve()}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("Config error: %s", exc)
        return 2

    # Resolve judge backend (catches typo'd names + missing plugins)
    try:
        build_judge(cfg.judge_backend)
    except (PluginNotFoundError, PluginLoadError) as exc:
        logger.error("%s", exc)
        return 2
    except JudgeUnavailableError as exc:
        logger.error("%s", exc)
        return 2

    # Verify each test_case.expected_tools is declared in cfg.tools (warn only)
    declared = set(cfg.tools)
    referenced: set[str] = set()
    for tc in cfg.test_cases:
        referenced.update(tc.expected_tools)
    unknown = referenced - declared
    if unknown:
        logger.warning(
            "test_cases reference tools not declared under 'tools': %s",
            sorted(unknown),
        )

    # Plugin evaluators referenced must resolve
    from agentic_evaluation.plugins import load_evaluator_plugin

    for spec in cfg.plugin_evaluators:
        name = spec.get("name", "")
        try:
            load_evaluator_plugin(name)
        except (PluginNotFoundError, PluginLoadError) as exc:
            logger.error("%s", exc)
            return 2

    print(
        f"OK: {len(cfg.test_cases)} test cases, {len(cfg.tools)} tools, judge={cfg.judge_backend}"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-eval",
        description="Evaluate any AI agent against the three-layer SDK framework.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run evaluations")
    run.add_argument("--config", type=Path, help="Path to eval_config.yaml")
    run.add_argument(
        "--task-fn",
        required=True,
        help="task_fn import spec, e.g. 'mypkg.tasks:run'",
    )
    run.add_argument("--trials", type=int, default=1, help="Number of trials per layer")
    run.add_argument(
        "--layers",
        help="Comma-separated subset of layers (layer_1,layer_2,layer_3,domain)",
    )
    run.add_argument("--judge", help="Override judge backend (strands|noop|<plugin>)")
    run.add_argument("--output", type=Path, help="Write JSON results here")
    run.add_argument(
        "--detail",
        choices=["summary", "full"],
        default="summary",
        help="JSON output detail: summary (gating only) or full (per-evaluator, per-case)",
    )
    run.set_defaults(func=_cmd_run)

    init = sub.add_parser("init", help="Scaffold a new evaluation project")
    init.add_argument("--name", required=True, help="Project name")
    init.add_argument("--tools", help="Comma-separated tool names")
    init.add_argument("--region", default="us-east-1", help="AWS region")
    init.add_argument("--judge", default="strands", help="Default judge backend")
    init.add_argument("--directory", help="Target directory (default: current)")
    init.add_argument("--force", action="store_true", help="Overwrite existing files")
    init.set_defaults(func=_cmd_init)

    validate = sub.add_parser("validate", help="Validate eval_config.yaml")
    validate.add_argument("--config", type=Path, help="Path to eval_config.yaml")
    validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.quiet)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
