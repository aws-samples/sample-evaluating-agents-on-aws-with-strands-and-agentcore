"""Tests for persistent CodeBuild deployment security controls."""

import json
import re
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from scripts.deploy_codebuild import (
    SCAN_EXCEPTIONS,
    _ensure_ecr_repo,
    _ensure_source_bucket,
    _wait_for_image_scan,
)


def test_existing_ecr_repository_is_made_immutable() -> None:
    ecr = MagicMock()
    ecr.describe_repositories.return_value = {
        "repositories": [
            {
                "imageTagMutability": "MUTABLE",
                "imageScanningConfiguration": {"scanOnPush": False},
            }
        ]
    }

    uri = _ensure_ecr_repo(ecr, "agent-eval-runtime", "eu-west-1", "111122223333")

    assert uri == "111122223333.dkr.ecr.eu-west-1.amazonaws.com/agent-eval-runtime"
    ecr.put_image_tag_mutability.assert_called_once_with(
        repositoryName="agent-eval-runtime",
        imageTagMutability="IMMUTABLE",
    )
    ecr.put_image_scanning_configuration.assert_called_once_with(
        repositoryName="agent-eval-runtime",
        imageScanningConfiguration={"scanOnPush": True},
    )


def test_existing_source_bucket_is_reconciled() -> None:
    s3 = MagicMock()
    s3.get_bucket_policy.side_effect = ClientError(
        {"Error": {"Code": "NoSuchBucketPolicy", "Message": "missing"}},
        "GetBucketPolicy",
    )

    _ensure_source_bucket(s3, "source-bucket", "eu-west-1")

    s3.create_bucket.assert_not_called()
    s3.put_bucket_versioning.assert_called_once_with(
        Bucket="source-bucket",
        VersioningConfiguration={"Status": "Enabled"},
    )
    policy = json.loads(s3.put_bucket_policy.call_args.kwargs["Policy"])
    deny = policy["Statement"][0]
    assert deny["Effect"] == "Deny"
    assert deny["Action"] == "s3:*"
    assert deny["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}


def test_image_scan_blocks_critical_or_high_findings() -> None:
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = {
        "imageScanStatus": {"status": "COMPLETE"},
        "imageScanFindings": {"findingSeverityCounts": {"CRITICAL": 1, "HIGH": 2}},
    }

    with pytest.raises(RuntimeError, match="CRITICAL=1, HIGH=2"):
        _wait_for_image_scan(ecr, "agent-eval-runtime", "review")


def test_image_scan_allows_medium_and_lower_findings() -> None:
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = {
        "imageScanStatus": {"status": "COMPLETE"},
        "imageScanFindings": {"findingSeverityCounts": {"MEDIUM": 2, "LOW": 1}},
    }

    assert _wait_for_image_scan(ecr, "agent-eval-runtime", "review") == {
        "MEDIUM": 2,
        "LOW": 1,
    }


def _scan_response(counts: dict[str, int], findings: list[dict[str, object]]) -> dict[str, object]:
    """A completed scan-on-push response, shaped as ECR basic scanning returns it."""
    return {
        "imageScanStatus": {"status": "COMPLETE"},
        "imageScanFindings": {"findingSeverityCounts": counts, "findings": findings},
    }


def _finding(severity: str, name: str, package: str, version: str) -> dict[str, object]:
    return {
        "severity": severity,
        "name": name,
        "attributes": [
            {"key": "package_name", "value": package},
            {"key": "package_version", "value": version},
        ],
    }


def test_blocked_scan_names_the_offending_package_not_just_a_count() -> None:
    # A count alone ("HIGH=1") forces a second manual describe call before
    # anyone can judge whether the finding is reachable, so the CVE and the
    # package version belong in the error that stops the deploy. MEDIUM is
    # present to prove the summary reports only what actually blocked.
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = _scan_response(
        {"HIGH": 1, "MEDIUM": 1},
        [
            _finding("HIGH", "CVE-2026-17564", "zlib1g", "1:1.2.13.dfsg-1"),
            _finding("MEDIUM", "CVE-2026-18477", "tar", "1.34+dfsg-1.2+deb12u1"),
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        _wait_for_image_scan(ecr, "agent-eval-runtime", "review")

    message = str(excinfo.value)
    assert "HIGH CVE-2026-17564 (zlib1g 1:1.2.13.dfsg-1)" in message
    assert "CVE-2026-18477" not in message


def test_documented_exception_lets_the_openssl_quic_finding_through() -> None:
    # The one waiver this gate carries: the base image ships openssl 3.0.20,
    # which has no QUIC server, and no Debian suite offers a fixed version, so
    # the finding is both unreachable and unfixable. Everything about it is
    # asserted from SCAN_EXCEPTIONS rather than restated, so editing the
    # registry cannot leave this test asserting a stale claim.
    (exception,) = SCAN_EXCEPTIONS
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = _scan_response(
        {"HIGH": 1},
        [_finding("HIGH", exception.cve, exception.package, "3.0.20-1~deb12u2")],
    )

    assert _wait_for_image_scan(ecr, "agent-eval-runtime", "review") == {"HIGH": 1}


def test_exception_expires_once_the_package_reaches_the_affected_release() -> None:
    # Positive control for the version bound. trixie ships openssl 3.5.6 --
    # the series that introduced the vulnerable QUIC listener -- so a base
    # image bump must make this same CVE block again instead of inheriting the
    # waiver written for the 3.0 series.
    (exception,) = SCAN_EXCEPTIONS
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = _scan_response(
        {"HIGH": 1},
        [_finding("HIGH", exception.cve, exception.package, "3.5.6-1~deb13u2")],
    )

    with pytest.raises(RuntimeError, match=re.escape(f"{exception.cve} ({exception.package}")):
        _wait_for_image_scan(ecr, "agent-eval-runtime", "review")


@pytest.mark.parametrize(
    ("name", "package", "version"),
    [
        # Same package and version, different CVE: the waiver is per-finding.
        ("CVE-2026-99999", "openssl", "3.0.20-1~deb12u2"),
        # Same CVE, different package: a name collision must not excuse it.
        ("CVE-2026-14456", "openssl-dev", "3.0.20-1~deb12u2"),
        # Unparseable version: fail closed rather than assume it is below 3.5.
        ("CVE-2026-14456", "openssl", "unknown"),
    ],
)
def test_near_misses_of_the_exception_still_block(name: str, package: str, version: str) -> None:
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = _scan_response(
        {"HIGH": 1}, [_finding("HIGH", name, package, version)]
    )

    with pytest.raises(RuntimeError, match="ECR scan blocked"):
        _wait_for_image_scan(ecr, "agent-eval-runtime", "review")


def test_unaccounted_findings_block_even_when_every_listed_one_is_excused() -> None:
    # `findings` is paginated and this gate reads only the first page, so a
    # truncated list must not read as "everything else was excused too". The
    # count is authoritative: two HIGHs reported, one accounted for, still a
    # blocked deploy.
    (exception,) = SCAN_EXCEPTIONS
    ecr = MagicMock()
    ecr.describe_image_scan_findings.return_value = _scan_response(
        {"HIGH": 2},
        [_finding("HIGH", exception.cve, exception.package, "3.0.20-1~deb12u2")],
    )

    with pytest.raises(RuntimeError, match="only 1 accounted for"):
        _wait_for_image_scan(ecr, "agent-eval-runtime", "review")
