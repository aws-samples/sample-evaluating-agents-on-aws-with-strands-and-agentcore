"""Tests for guarded CDK command construction."""

from unittest.mock import patch

from scripts.deploy_stack import _cdk_command


def test_deploy_forwards_recorded_approval_to_cdk() -> None:
    with (
        patch("scripts.deploy_stack.shutil.which", return_value="/usr/local/bin/cdk"),
        patch("scripts.deploy_stack._run") as run,
    ):
        _cdk_command(
            "deploy",
            image_uri="111122223333.dkr.ecr.eu-west-1.amazonaws.com/repo@sha256:abc",
            profile="test",
            region="eu-west-1",
            account="111122223333",
            env_name="dev",
        )

    command = run.call_args.args[0]
    assert command[:2] == ["cdk", "deploy"]
    assert "--all" in command
    assert command[-2:] == ["--require-approval", "never"]


def test_diff_does_not_pass_unsupported_all_option() -> None:
    with (
        patch("scripts.deploy_stack.shutil.which", return_value="/usr/local/bin/cdk"),
        patch("scripts.deploy_stack._run") as run,
    ):
        _cdk_command(
            "diff",
            image_uri="111122223333.dkr.ecr.eu-west-1.amazonaws.com/repo@sha256:abc",
            profile="test",
            region="eu-west-1",
            account="111122223333",
            env_name="dev",
        )

    command = run.call_args.args[0]
    assert command[:2] == ["cdk", "diff"]
    assert "--all" not in command
    assert "--require-approval" not in command
