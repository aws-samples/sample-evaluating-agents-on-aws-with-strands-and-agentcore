"""Tests for persistent CodeBuild deployment security controls."""

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from scripts.deploy_codebuild import (
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
