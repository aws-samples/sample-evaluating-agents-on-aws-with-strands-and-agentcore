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
def test_quickstart_runs_and_passes(monkeypatch):
    """`python quickstart/run_demo.py` must exit 0 with all-passes."""
    env = os.environ.copy()
    env.pop("EVAL_CONFIG_PATH", None)
    # Command is a fixed local script path derived from REPO_ROOT; no external input. shell=True is not used.
    result = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit -- test invokes a fixed local command, no external input  # nosec B603
        [sys.executable, str(REPO_ROOT / "quickstart" / "run_demo.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Quickstart failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Overall: PASS" in result.stdout
