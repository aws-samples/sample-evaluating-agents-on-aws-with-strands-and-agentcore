# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared account, region, and approval guards for repository AWS scripts."""

from __future__ import annotations

import re
import sys
from typing import Any

import boto3

ALLOWED_AWS_REGIONS = frozenset({"eu-west-1"})
_ACCOUNT_ID_PATTERN = re.compile(r"^\d{12}$")


def verified_session(
    *,
    profile: str,
    region: str,
    expected_account: str,
) -> tuple[boto3.Session, dict[str, Any]]:
    """Create an explicit session and fail closed outside the repository allowlist."""
    if not profile.strip():
        raise ValueError("An explicit AWS profile is required")
    if region not in ALLOWED_AWS_REGIONS:
        allowed = ", ".join(sorted(ALLOWED_AWS_REGIONS))
        raise ValueError(f"AWS region {region!r} is not allowed; allowed regions: {allowed}")
    if not _ACCOUNT_ID_PATTERN.fullmatch(expected_account):
        raise ValueError("Expected AWS account must be a 12-digit account ID")

    session = boto3.Session(profile_name=profile, region_name=region)
    identity = session.client("sts", region_name=region).get_caller_identity()
    actual_account = str(identity.get("Account", ""))
    if actual_account != expected_account:
        raise RuntimeError(
            f"AWS account mismatch: profile {profile!r} resolved to "
            f"{actual_account or '<unknown>'}, expected {expected_account}"
        )
    return session, identity


def reverify_identity(
    session: boto3.Session,
    *,
    profile: str,
    region: str,
    expected_account: str,
) -> dict[str, Any]:
    """Re-check STS immediately before a separate mutation phase."""
    identity = session.client("sts", region_name=region).get_caller_identity()
    actual_account = str(identity.get("Account", ""))
    if actual_account != expected_account:
        raise RuntimeError(
            f"AWS account changed before mutation: profile {profile!r} resolved "
            f"to {actual_account or '<unknown>'}, expected {expected_account}"
        )
    return identity


def confirm_mutation(
    *,
    action: str,
    account: str,
    region: str,
    cost: str,
    approved: bool,
) -> None:
    """Display target and cost, then require an exact interactive confirmation."""
    print(f"Target account: {account}")
    print(f"Target region:  {region}")
    print(f"Action:         {action}")
    print(f"Cost estimate:  {cost}")
    if approved:
        print("Approval:      provided by --yes")
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive approval is required; rerun in a TTY or pass --yes")
    phrase = f"{action} {account} {region}"
    response = input(f"Type {phrase!r} to continue: ").strip()
    if response != phrase:
        raise RuntimeError("AWS mutation cancelled: confirmation did not match")
