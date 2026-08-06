#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Run the SDK quickstart against the mock agent.

Usage::

    python quickstart/run_demo.py

Exits 0 if every layer passes, 1 otherwise.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EVAL_CONFIG_PATH", str(Path(__file__).parent / "eval_config.yaml"))

from agentic_evaluation import load_config, run_all_layers  # noqa: E402
from quickstart.mock_task_fn import task_fn  # noqa: E402


def main() -> int:
    """Evaluate the mock task_fn through every layer and report the outcome.

    Returns:
        A process exit code: 0 when every layer passed, 1 otherwise.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = load_config()
    print(f"Loaded config: {cfg.project_name} (judge={cfg.judge_backend})")

    results = run_all_layers(task_fn=task_fn, judge_backend="noop")

    print()
    print("Layer results")
    print("-------------")
    for name in ("layer_1", "layer_2", "layer_3", "domain"):
        info = results.get(name)
        if info is None:
            continue
        status = "PASS" if info["passed"] else "FAIL"
        print(f"  {name:<10} {status}")

    overall = "PASS" if results["all_passed"] else "FAIL"
    print()
    print(f"Overall: {overall}")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
