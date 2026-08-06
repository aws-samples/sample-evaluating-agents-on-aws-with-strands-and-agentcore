# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""End-to-end test for the SDK quickstart demo (no AWS, no LLM)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.sdk
def test_quickstart_runs_and_passes():
    """`python quickstart/run_demo.py` must exit 0 with all-passes."""
    env = os.environ.copy()
    env.pop("EVAL_CONFIG_PATH", None)
    # S603/B603: the command is this interpreter plus a path derived from
    # REPO_ROOT — no external input, and shell=True is not used. check=False
    # because the exit code is what this test asserts on.
    result = subprocess.run(  # noqa: S603  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit  # nosec B603
        [sys.executable, str(REPO_ROOT / "quickstart" / "run_demo.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"Quickstart failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    for layer in ("layer_1", "layer_2", "layer_3", "domain"):
        assert f"{layer:<10} PASS" in result.stdout
    assert "Overall: PASS" in result.stdout
