#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""End-to-end deploy: CodeBuild builds the agent image, then CDK deploys all stacks.

Pipeline
--------
1. Run ``deploy_codebuild.py`` to build the agent container in AWS CodeBuild
   and push it to ECR. No local Docker required. The script builds only;
   CDK owns the runtime.
2. Compute the resulting image URI.
3. Invoke ``cdk deploy --all`` with ``-c agent_image_uri=<uri>``. The
   ``agent_runtime_stack`` references the prebuilt image via
   ``AgentRuntimeArtifact.from_image_uri`` (no bundling, no Docker).

Usage::

    python scripts/deploy_stack.py --profile PROFILE --region eu-west-1 \
        --expected-account 123456789012

Prereqs: ``aws-cdk`` CLI on PATH (``npm i -g aws-cdk``), AWS creds with
permissions for IAM/ECR/CodeBuild/S3/CloudFormation/Bedrock.

Cleanup
-------
Do not run a bare ``cdk destroy``. Inventory the explicit profile, expected
account, and ``eu-west-1`` first; classify every stateful resource and image
digest; create and verify backups; review the destroy change; and obtain
approval for the exact stack and retained-data deletion sets. Reverify STS
identity before every mutation and inventory billable residuals afterwards.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import boto3

try:
    from scripts.aws_safety import confirm_mutation, reverify_identity, verified_session
except ModuleNotFoundError:
    from aws_safety import confirm_mutation, reverify_identity, verified_session

logger = logging.getLogger("deploy_stack")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
CDK_DIR = REPO_ROOT / "examples" / "vehicle-auction-agent" / "cdk"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    # cmd is built entirely from static/operator-controlled values (sys.executable,
    # repo-internal paths from Path(__file__), and argparse inputs). No external
    # or user-supplied shell strings are involved; shell=True is not used.
    return subprocess.run(
        cmd, check=True, **kw
    )  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit -- args are static/operator-controlled, not external input  # nosec B603


def _git_sha() -> str | None:
    try:
        result = _run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _digest_uri(
    session: boto3.Session,
    *,
    region: str,
    account: str,
    ecr_repo: str,
    image_tag: str,
) -> str:
    ecr = session.client("ecr", region_name=region)
    details = ecr.describe_images(
        repositoryName=ecr_repo,
        imageIds=[{"imageTag": image_tag}],
    )["imageDetails"]
    if len(details) != 1 or not details[0].get("imageDigest"):
        raise RuntimeError(f"Could not resolve ECR digest for {ecr_repo}:{image_tag}")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{ecr_repo}@{details[0]['imageDigest']}"


def _build_image(
    session: boto3.Session,
    *,
    profile: str,
    region: str,
    account: str,
    expected_account: str,
    ecr_repo: str,
    image_tag: str,
) -> str:
    """Build via CodeBuild and return an immutable digest URI."""
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "deploy_codebuild.py"),
            "--region",
            region,
            "--profile",
            profile,
            "--expected-account",
            expected_account,
            "--ecr-repo",
            ecr_repo,
            "--image-tag",
            image_tag,
            "--yes",
            # No --register-runtime: build only. CDK (deployed below) owns the
            # runtime and its role/env wiring.
        ]
    )
    return _digest_uri(
        session,
        region=region,
        account=account,
        ecr_repo=ecr_repo,
        image_tag=image_tag,
    )


def _cdk_command(
    action: str,
    *,
    image_uri: str,
    profile: str,
    region: str,
    account: str,
    env_name: str,
) -> None:
    if shutil.which("cdk") is None:
        raise RuntimeError("cdk CLI not found on PATH. Install with: npm install -g aws-cdk")
    env = {
        **os.environ,
        "AWS_PROFILE": profile,
        "AWS_ACCOUNT_ID": account,
        "AWS_REGION": region,
        "AGENT_IMAGE_URI": image_uri,
        "JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION": "1",
    }
    command = [
        "cdk",
        action,
        "--profile",
        profile,
        "-c",
        f"environment={env_name}",
        "-c",
        f"agent_image_uri={image_uri}",
    ]
    if action == "deploy":
        # The repository confirmation above already verifies the exact account,
        # region, action, and cost. Forward that approval for non-interactive
        # execution; otherwise CDK creates a change set and aborts without a TTY.
        command.extend(["--all", "--require-approval", "never"])
    _run(
        command,
        cwd=CDK_DIR,
        env=env,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, help="Explicit AWS CLI profile")
    p.add_argument("--region", required=True)
    p.add_argument("--expected-account", required=True, help="Expected 12-digit AWS account ID")
    p.add_argument("--ecr-repo", default="agent-eval-runtime")
    p.add_argument("--image-tag", default=None, help="Immutable tag; defaults to the git SHA")
    p.add_argument("--environment", default="dev")
    p.add_argument("--yes", action="store_true", help="Approve displayed AWS mutation and cost")
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing immutable --image-tag; skip CodeBuild and run CDK.",
    )
    args = p.parse_args()

    session, identity = verified_session(
        profile=args.profile,
        region=args.region,
        expected_account=args.expected_account,
    )
    account = identity["Account"]
    if args.image_tag is None:
        args.image_tag = _git_sha()
        if not args.image_tag:
            p.error("--image-tag is required outside a git checkout")

    if args.skip_build:
        image_uri = _digest_uri(
            session,
            region=args.region,
            account=account,
            ecr_repo=args.ecr_repo,
            image_tag=args.image_tag,
        )
        logger.info("Skipping CodeBuild; using existing image: %s", image_uri)
    else:
        confirm_mutation(
            action="build-agent-image",
            account=account,
            region=args.region,
            cost=(
                "about $0.01 per CodeBuild minute, $0.10/GB-month ECR, and "
                "$0.023/GB-month S3 source storage"
            ),
            approved=args.yes,
        )
        image_uri = _build_image(
            session,
            profile=args.profile,
            region=args.region,
            account=account,
            expected_account=args.expected_account,
            ecr_repo=args.ecr_repo,
            image_tag=args.image_tag,
        )
        logger.info("Built image: %s", image_uri)

    reverify_identity(
        session,
        profile=args.profile,
        region=args.region,
        expected_account=args.expected_account,
    )
    logger.info("Reviewing CDK diff before deployment")
    _cdk_command(
        "diff",
        image_uri=image_uri,
        profile=args.profile,
        region=args.region,
        account=account,
        env_name=args.environment,
    )
    confirm_mutation(
        action="deploy-cdk-stacks",
        account=account,
        region=args.region,
        cost=(
            "estimated $55-105/month for the dev reference stack (including "
            "four customer-managed KMS keys), plus Bedrock model and AgentCore "
            "request usage"
        ),
        approved=args.yes,
    )
    reverify_identity(
        session,
        profile=args.profile,
        region=args.region,
        expected_account=args.expected_account,
    )
    _cdk_command(
        "deploy",
        image_uri=image_uri,
        profile=args.profile,
        region=args.region,
        account=account,
        env_name=args.environment,
    )
    logger.info("Deploy complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
