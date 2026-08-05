# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for AWS account, region, and approval guards."""

from unittest.mock import MagicMock, patch

import pytest

from scripts.aws_safety import confirm_mutation, reverify_identity, verified_session


def test_verified_session_uses_explicit_profile_and_region() -> None:
    session = MagicMock()
    session.client.return_value.get_caller_identity.return_value = {
        "Account": "111122223333",
        "Arn": "arn:aws:iam::111122223333:user/test",
    }

    with patch("scripts.aws_safety.boto3.Session", return_value=session) as session_factory:
        result, identity = verified_session(
            profile="test-profile",
            region="eu-west-1",
            expected_account="111122223333",
        )

    session_factory.assert_called_once_with(
        profile_name="test-profile",
        region_name="eu-west-1",
    )
    session.client.assert_called_once_with("sts", region_name="eu-west-1")
    assert result is session
    assert identity["Account"] == "111122223333"


def test_verified_session_rejects_region_before_loading_credentials() -> None:
    with (
        patch("scripts.aws_safety.boto3.Session") as session_factory,
        pytest.raises(ValueError, match="not allowed"),
    ):
        verified_session(
            profile="test-profile",
            region="us-east-1",
            expected_account="111122223333",
        )
    session_factory.assert_not_called()


def test_verified_session_and_recheck_reject_account_mismatch() -> None:
    session = MagicMock()
    session.client.return_value.get_caller_identity.return_value = {"Account": "999900001111"}
    with (
        patch("scripts.aws_safety.boto3.Session", return_value=session),
        pytest.raises(RuntimeError, match="account mismatch"),
    ):
        verified_session(
            profile="test-profile",
            region="eu-west-1",
            expected_account="111122223333",
        )

    with pytest.raises(RuntimeError, match="account changed"):
        reverify_identity(
            session,
            profile="test-profile",
            region="eu-west-1",
            expected_account="111122223333",
        )


def test_confirmation_requires_exact_phrase(monkeypatch) -> None:
    monkeypatch.setattr("scripts.aws_safety.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda _: "deploy-cdk-stacks 111122223333 eu-west-1",
    )

    confirm_mutation(
        action="deploy-cdk-stacks",
        account="111122223333",
        region="eu-west-1",
        cost="$55-105/month",
        approved=False,
    )
