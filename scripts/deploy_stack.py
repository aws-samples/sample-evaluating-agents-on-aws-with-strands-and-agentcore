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

    python scripts/deploy_stack.py --region eu-west-1
    python scripts/deploy_stack.py --region eu-west-1 --skip-build  # CDK-only
    python scripts/deploy_stack.py --region eu-west-1 --image-tag v1.2.3

Prereqs: ``aws-cdk`` CLI on PATH (``npm i -g aws-cdk``), AWS creds with
permissions for IAM/ECR/CodeBuild/S3/CloudFormation/Bedrock.

Cleanup
-------
**Important:** The following command deletes all S3 objects and
DynamoDB tables in the target environment. If you need to preserve any data,
create a backup before running it.
Some resources use ``RemovalPolicy.RETAIN`` in non-dev environments and must
be deleted manually afterwards. AWS costs accrue until cleanup is complete.

To remove all deployed stacks run::

    cdk destroy --all -c environment=<env>
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


def _build_image(region: str, ecr_repo: str, image_tag: str) -> str:
    """Build via CodeBuild, return the resulting image URI."""
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "deploy_codebuild.py"),
            "--region",
            region,
            "--ecr-repo",
            ecr_repo,
            "--image-tag",
            image_tag,
            # No --register-runtime: build only. CDK (deployed below) owns the
            # runtime and its role/env wiring.
        ]
    )
    account = boto3.Session(region_name=region).client("sts").get_caller_identity()["Account"]
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{ecr_repo}:{image_tag}"


def _cdk_deploy(image_uri: str, region: str, env_name: str) -> None:
    if shutil.which("cdk") is None:
        raise RuntimeError("cdk CLI not found on PATH. Install with: npm install -g aws-cdk")
    account = boto3.Session(region_name=region).client("sts").get_caller_identity()["Account"]
    env = {
        **os.environ,
        "AWS_ACCOUNT_ID": account,
        "AWS_REGION": region,
        "AGENT_IMAGE_URI": image_uri,
        "JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION": "1",
    }
    _run(
        [
            "cdk",
            "deploy",
            "--all",
            "--require-approval",
            "never",
            "--concurrency",
            "2",
            "-c",
            f"environment={env_name}",
            "-c",
            f"agent_image_uri={image_uri}",
        ],
        cwd=CDK_DIR,
        env=env,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", required=True)
    p.add_argument("--ecr-repo", default="agent-eval-runtime")
    p.add_argument("--image-tag", default="latest")
    p.add_argument("--environment", default="dev")
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing :latest image; skip CodeBuild and just run CDK.",
    )
    args = p.parse_args()

    if args.skip_build:
        account = (
            boto3.Session(region_name=args.region).client("sts").get_caller_identity()["Account"]
        )
        image_uri = (
            f"{account}.dkr.ecr.{args.region}.amazonaws.com/{args.ecr_repo}:{args.image_tag}"
        )
        logger.info("Skipping CodeBuild; using existing image: %s", image_uri)
    else:
        image_uri = _build_image(args.region, args.ecr_repo, args.image_tag)
        logger.info("Built image: %s", image_uri)

    _cdk_deploy(image_uri, args.region, args.environment)
    logger.info("Deploy complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
