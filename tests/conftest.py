# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared pytest setup: make the example agent importable as ``agent``."""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the example agent importable as the "agent" package. It lives under
# examples/vehicle-auction-agent/ (the reference example), separate from the
# SDK in src/agentic_evaluation/.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_DIR = _REPO_ROOT / "examples" / "vehicle-auction-agent"
_AGENT_DIR = _EXAMPLE_DIR / "agent"
# _EXAMPLE_DIR makes "import agent.app" resolve; _AGENT_DIR makes the agent's
# own intra-package imports (e.g. "from utils.geo import ...") resolve.
for _p in (_REPO_ROOT, _EXAMPLE_DIR, _AGENT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
