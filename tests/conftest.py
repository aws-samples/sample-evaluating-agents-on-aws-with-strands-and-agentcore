# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared pytest setup: make the example agent importable as ``agent``."""

import os
import sys
from functools import lru_cache
from pathlib import Path

import boto3
import pytest

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

from scripts.aws_safety import reverify_identity, verified_session  # noqa: E402


@lru_cache(maxsize=1)
def _deployed_aws_target() -> tuple[boto3.Session, str, str, str]:
    profile = os.environ.get("AWS_PROFILE")
    region = os.environ.get("AWS_REGION")
    expected_account = os.environ.get("AWS_ACCOUNT_ID")
    missing = [
        name
        for name, value in (
            ("AWS_PROFILE", profile),
            ("AWS_REGION", region),
            ("AWS_ACCOUNT_ID", expected_account),
        )
        if not value
    ]
    if missing:
        pytest.fail("Deployed tests require explicit AWS target variables: " + ", ".join(missing))
    session, identity = verified_session(
        profile=profile,
        region=region,
        expected_account=expected_account,
    )
    return session, profile, region, identity["Account"]


@pytest.fixture(autouse=True)
def _guard_deployed_aws_test(request):
    """Pin every deployed test to one verified account/profile/region."""
    if request.node.get_closest_marker("deployed") is None:
        yield
        return

    session, profile, region, expected_account = _deployed_aws_target()
    expected_approval = f"run-deployed-tests {expected_account} {region}"
    if os.environ.get("AWS_DEPLOYED_TEST_APPROVAL") != expected_approval:
        pytest.fail(
            "Deployed tests incur AWS request cost and may mutate sample data. "
            f"After approval, set AWS_DEPLOYED_TEST_APPROVAL={expected_approval!r}."
        )
    reverify_identity(
        session,
        profile=profile,
        region=region,
        expected_account=expected_account,
    )
    previous_session = boto3.DEFAULT_SESSION
    boto3.DEFAULT_SESSION = session
    try:
        yield
    finally:
        boto3.DEFAULT_SESSION = previous_session


@pytest.fixture
def aws_region() -> str:
    return _deployed_aws_target()[2]


@pytest.fixture
def account_id() -> str:
    return _deployed_aws_target()[3]


@pytest.fixture
def environment() -> str:
    value = os.environ.get("ENVIRONMENT")
    if not value:
        pytest.fail("Deployed tests require an explicit ENVIRONMENT")
    return value
