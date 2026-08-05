"""Tests for post-deployment resource validation."""

import json
from unittest.mock import MagicMock

from scripts.validate_deployment import (
    validate_lambda_exposure,
    validate_s3_bucket,
)


def test_validate_s3_bucket_reads_standard_encryption_response() -> None:
    s3 = MagicMock()
    s3.get_bucket_encryption.return_value = {
        "ServerSideEncryptionConfiguration": {
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256",
                    }
                }
            ]
        }
    }
    s3.get_bucket_versioning.return_value = {"Status": "Enabled"}
    session = MagicMock()
    session.client.return_value = s3

    assert validate_s3_bucket(session, "bucket", "eu-west-1") is True


def test_validate_lambda_exposure_accepts_scoped_service_permission() -> None:
    lambda_client = MagicMock()
    lambda_client.list_function_url_configs.return_value = {"FunctionUrlConfigs": []}
    lambda_client.get_policy.return_value = {
        "Policy": json.dumps(
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "events.amazonaws.com"},
                        "Action": "lambda:InvokeFunction",
                        "Condition": {
                            "ArnLike": {
                                "AWS:SourceArn": (
                                    "arn:aws:events:eu-west-1:111122223333:rule/expected"
                                )
                            }
                        },
                    }
                ]
            }
        )
    }
    session = MagicMock()
    session.client.return_value = lambda_client

    assert (
        validate_lambda_exposure(
            session,
            "function",
            "eu-west-1",
            allowed_service_principal="events.amazonaws.com",
            allowed_source_arn_patterns=(r"arn:aws:events:eu-west-1:111122223333:rule/expected",),
        )
        is True
    )


def test_validate_lambda_exposure_rejects_function_url() -> None:
    lambda_client = MagicMock()
    lambda_client.list_function_url_configs.return_value = {
        "FunctionUrlConfigs": [{"AuthType": "AWS_IAM"}]
    }
    session = MagicMock()
    session.client.return_value = lambda_client

    assert (
        validate_lambda_exposure(
            session,
            "function",
            "eu-west-1",
            allowed_service_principal="events.amazonaws.com",
            allowed_source_arn_patterns=(r".+",),
        )
        is False
    )


def test_validate_lambda_exposure_rejects_public_permission() -> None:
    lambda_client = MagicMock()
    lambda_client.list_function_url_configs.return_value = {"FunctionUrlConfigs": []}
    lambda_client.get_policy.return_value = {
        "Policy": json.dumps(
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "lambda:InvokeFunction",
                    }
                ]
            }
        )
    }
    session = MagicMock()
    session.client.return_value = lambda_client

    assert (
        validate_lambda_exposure(
            session,
            "function",
            "eu-west-1",
            allowed_service_principal="events.amazonaws.com",
            allowed_source_arn_patterns=(r".+",),
        )
        is False
    )
