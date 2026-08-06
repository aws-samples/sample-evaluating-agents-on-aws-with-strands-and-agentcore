"""Tests for guarded CDK command construction."""

from unittest.mock import Mock, patch

from scripts.deploy_stack import AwsTarget, _cdk_command

_IMAGE_URI = "111122223333.dkr.ecr.eu-west-1.amazonaws.com/repo@sha256:abc"


def _target() -> AwsTarget:
    """A verified target whose session is never used by _cdk_command."""
    return AwsTarget(
        session=Mock(),
        profile="test",
        region="eu-west-1",
        account="111122223333",
        expected_account="111122223333",
    )


def _command_for(action: str, *, alert_email: str | None = None) -> list[str]:
    """Return the argv ``_cdk_command`` would hand to the CDK CLI."""
    with (
        patch("scripts.deploy_stack.shutil.which", return_value="/usr/local/bin/cdk"),
        patch("scripts.deploy_stack._run") as run,
    ):
        _cdk_command(
            _target(),
            action,
            image_uri=_IMAGE_URI,
            env_name="dev",
            alert_email=alert_email,
        )
    return run.call_args.args[0]


def test_deploy_forwards_recorded_approval_to_cdk() -> None:
    command = _command_for("deploy")
    assert command[:2] == ["cdk", "deploy"]
    assert "--all" in command
    assert command[-2:] == ["--require-approval", "never"]


def test_diff_does_not_pass_unsupported_all_option() -> None:
    command = _command_for("diff")
    assert command[:2] == ["cdk", "diff"]
    assert "--all" not in command
    assert "--require-approval" not in command


def test_alert_email_is_forwarded_as_cdk_context() -> None:
    assert "alert_email=ops@example.com" in _command_for("deploy", alert_email="ops@example.com")


def test_omitted_alert_email_adds_no_context() -> None:
    """No flag must mean no context key, so the default template is unchanged."""
    assert not any(arg.startswith("alert_email=") for arg in _command_for("deploy"))
