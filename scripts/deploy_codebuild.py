#!/usr/bin/env python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Build the agent container in AWS CodeBuild and push it to Amazon Elastic Container Registry (Amazon ECR).

Avoids local docker entirely (per project policy). Idempotent:
- Amazon ECR repo is created if missing.
- AWS Identity and Access Management (IAM) roles (AWS CodeBuild service role + AgentCore execution role) are
  created/updated as needed.
- AWS CodeBuild project is created or updated.
- Source is uploaded as a zip to Amazon S3 and AWS CodeBuild reads it from there.
- Prints the resulting immutable image URI.

**This script builds the image only.** The AgentCore Runtime is owned by
CDK (``cdk/lib/agent_runtime_stack.py``), which wires the execution-role
Amazon S3/Amazon DynamoDB grants and the ``DATA_BUCKET``/``LANCEDB_PATH``/``DEALERS_TABLE``
environment variables the agent needs. Registering the runtime here as well
created two owners for one resource: an out-of-band ``update_agent_runtime``
silently overwrote CDK's role and nulled its env vars (CloudFormation drift),
so the deployed agent could not read its data. The normal path is
``scripts/deploy_stack.py`` (build here, then ``cdk deploy``).

``--register-runtime`` is an explicit escape hatch for a no-CDK quickstart:
it creates/updates a standalone runtime directly. Do **not** point it at a
CDK-managed runtime name or you reintroduce the drift.

Cost Warning:
    This script creates AWS resources that incur ongoing charges:
    - **ECR repository**: Image storage costs (~$0.10/GB/month)
    - **S3 bucket**: Source artifact storage costs (~$0.023/GB/month)
    - **CodeBuild**: Compute costs per build minute (~$0.01/min for ARM LARGE)
    - **Amazon Bedrock AgentCore runtime** (if --register-runtime): Execution costs per request
    - **IAM roles**: No direct cost, but grant access to billable services

    Delete resources when no longer needed to avoid ongoing charges.
    See the Cleanup section below for deletion instructions.

Usage::

    # Build only (default); feed the printed image URI to `cdk deploy`:
    python scripts/deploy_codebuild.py --ecr-repo agent-eval-runtime --region eu-west-1

    # No-CDK quickstart: build AND register a standalone runtime:
    python scripts/deploy_codebuild.py --register-runtime \\
        --runtime-name agent_eval_runtime_dev --region eu-west-1 \\
        --env DATA_BUCKET=... --env DEALERS_TABLE=... --env LANCEDB_PATH=lancedb/latest.json

Cleanup:
    Important: The following commands delete AWS resources and their data.
    If you need to preserve any data, create a backup before running them.

    This script creates billable, persistent resources. CDK does not manage them,
    so ``cdk destroy`` will not remove them. Delete them manually when finished
    (defaults shown; adjust names/region to match your run):

        # 1. Delete the Amazon Bedrock AgentCore runtime (only if you used --register-runtime)
        aws bedrock-agentcore-control delete-agent-runtime \\
            --agent-runtime-name agent_eval_runtime_dev --region eu-west-1

        # 2. Delete the CodeBuild project
        aws codebuild delete-project --name agent-eval-runtime-build --region eu-west-1

        # 3. Back up source bucket contents (if needed), then empty and delete it
        # Default bucket name is agent-eval-codebuild-src-<ACCOUNT>-<REGION>
        # (or whatever you passed to --source-bucket). Replace ACCOUNT with your
        # AWS account ID (e.g., 123456789012) and the region to match your run.
        aws s3 sync s3://agent-eval-codebuild-src-ACCOUNT-eu-west-1 ./backup/codebuild-src/
        aws s3 rb s3://agent-eval-codebuild-src-ACCOUNT-eu-west-1 --force

        # 4. Delete the Amazon ECR repository
        # Note: --force deletes all container images without confirmation.
        aws ecr delete-repository --repository-name agent-eval-runtime \\
            --region eu-west-1 --force

        # 5. Delete the IAM roles (detach/delete inline policies first)
        for role in agent-eval-codebuild-role agent-eval-agentcore-role; do
          for p in $(aws iam list-role-policies --role-name "$role" \\
              --query 'PolicyNames[]' --output text); do
            aws iam delete-role-policy --role-name "$role" --policy-name "$p"
          done
          aws iam delete-role --role-name "$role"
        done
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("deploy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _git_sha() -> str | None:
    """Short git SHA of HEAD, or None outside a git checkout.

    Used as the default immutable image tag so each build pushes a unique,
    reproducible tag the ECR repo (IMMUTABLE) will accept.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip() or None


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "examples" / "vehicle-auction-agent" / "agent"

# Maximum wall-clock seconds to wait for a CodeBuild build before giving up.
# 30 minutes is generous for a container image build; adjust if your image is
# unusually large.
BUILD_TIMEOUT_SECONDS = 1800

# Models the agent invokes. Kept in sync with agent/app.py (MODEL_ID and the
# Amazon Titan embedding model) and with the Bedrock policy in
# cdk/lib/agent_runtime_stack.py so the runtime role grants least-privilege
# Amazon Bedrock access scoped to exactly these models, not Resource "*".
DEFAULT_MODEL_ID = "eu.anthropic.claude-sonnet-4-6"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Cross-region inference-profile IDs are prefixed with the geo (eu./us./apac.).
# The underlying foundation-model ARN drops that prefix, so a profile invocation
# needs both the inference-profile ARN and the bare foundation-model ARN granted.
_INFERENCE_PROFILE_PREFIXES = ("eu.", "us.", "apac.", "us-gov.")


def _base_model_id(model_id: str) -> str:
    """Strip a cross-region inference-profile geo prefix to the foundation model.

    ``eu.anthropic.claude-sonnet-4-6`` -> ``anthropic.claude-sonnet-4-6``. A
    plain foundation-model ID (no recognised prefix) is returned unchanged.
    """
    for prefix in _INFERENCE_PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


CODEBUILD_TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def _agentcore_trust(account: str, region: str) -> dict:
    """AgentCore Runtime trust policy with confused-deputy guards.

    ``aws:SourceAccount`` and ``aws:SourceArn`` conditions prevent
    cross-account confused-deputy attacks (AWS security best practice
    for service-linked execution roles).
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account},
                    "ArnLike": {
                        "aws:SourceArn": (f"arn:aws:bedrock-agentcore:{region}:{account}:runtime/*")
                    },
                },
            }
        ],
    }


def _ensure_role(iam, name: str, trust: dict, managed_policies: list[str]) -> str:
    try:
        role = iam.get_role(RoleName=name)["Role"]
        logger.info("IAM role exists: %s — refreshing trust policy", name)
        # Refresh trust policy so confused-deputy guards apply on re-deploy.
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust))
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        logger.info("Creating IAM role: %s", name)
        role = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust),
        )["Role"]
        # IAM role propagation
        time.sleep(8)
    for pol in managed_policies:
        iam.attach_role_policy(RoleName=name, PolicyArn=pol)
    return role["Arn"]


def _ensure_inline_policy(iam, role_name: str, policy_name: str, doc: dict) -> None:
    iam.put_role_policy(RoleName=role_name, PolicyName=policy_name, PolicyDocument=json.dumps(doc))


def _data_source_statements(runtime_env: dict[str, str], account: str, region: str) -> list[dict]:
    """Least-privilege grants for the agent's data sources in the no-CDK path.

    The runtime role otherwise has no S3/DynamoDB access, so the agent's
    ``hybrid_search``/``get_dealer_profile`` tools would fail at first call.
    Scope the grants to exactly the ``DATA_BUCKET``/``DEALERS_TABLE`` the caller
    passes via ``--env``; emit nothing when they are absent so the policy stays
    valid. Mirrors the grants in ``cdk/lib/agent_runtime_stack.py``.
    """
    statements: list[dict] = []
    bucket = runtime_env.get("DATA_BUCKET")
    if bucket:
        statements.append(
            {
                "Sid": "DataBucketRead",
                "Effect": "Allow",
                "Action": ["s3:GetObject*", "s3:GetBucket*", "s3:List*"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            }
        )
    table = runtime_env.get("DEALERS_TABLE")
    if table:
        statements.append(
            {
                "Sid": "DealersTableRead",
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:Query"],
                "Resource": f"arn:aws:dynamodb:{region}:{account}:table/{table}",
            }
        )
    return statements


def _ensure_ecr_repo(ecr, name: str, region: str, account: str) -> str:
    try:
        ecr.describe_repositories(repositoryNames=[name])
        logger.info("ECR repo exists: %s", name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise
        logger.info("Creating ECR repo: %s", name)
        ecr.create_repository(
            repositoryName=name,
            imageTagMutability="IMMUTABLE",
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
        )
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{name}"


def _ensure_source_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        logger.info("Source bucket exists: %s", bucket)
        return
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
            raise
    logger.info("Creating source bucket: %s", bucket)
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )


def _zip_agent_dir() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(AGENT_DIR.rglob("*")):
            if path.is_dir():
                continue
            if "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(AGENT_DIR))
    return buf.getvalue()


def _ensure_codebuild_project(
    cb,
    project_name: str,
    service_role_arn: str,
    source_bucket: str,
    source_key: str,
    region: str,
    account: str,
    repo_name: str,
    image_tag: str,
) -> None:
    env_vars = [
        {"name": "AWS_DEFAULT_REGION", "value": region},
        {"name": "AWS_ACCOUNT_ID", "value": account},
        {"name": "IMAGE_REPO_NAME", "value": repo_name},
        {"name": "IMAGE_TAG", "value": image_tag},
    ]
    project_kwargs = {
        "name": project_name,
        "source": {
            "type": "S3",
            "location": f"{source_bucket}/{source_key}",
            "buildspec": "buildspec.yml",
        },
        "artifacts": {"type": "NO_ARTIFACTS"},
        "environment": {
            "type": "ARM_CONTAINER",
            "image": "aws/codebuild/amazonlinux2-aarch64-standard:3.0",
            "computeType": "BUILD_GENERAL1_LARGE",
            "privilegedMode": True,
            "environmentVariables": env_vars,
        },
        "serviceRole": service_role_arn,
    }
    try:
        cb.create_project(**project_kwargs)
        logger.info("Created CodeBuild project: %s", project_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
        cb.update_project(**project_kwargs)
        logger.info("Updated CodeBuild project: %s", project_name)


def _run_build(cb, project_name: str) -> str:
    logger.info("Starting build for %s", project_name)
    build_id = cb.start_build(projectName=project_name)["build"]["id"]
    logger.info("Build started: %s — polling for completion", build_id)
    deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
    while True:
        time.sleep(20)
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"CodeBuild build {build_id} did not complete within "
                f"{BUILD_TIMEOUT_SECONDS}s ({BUILD_TIMEOUT_SECONDS // 60} minutes). "
                "Check the build logs in the AWS Console."
            )
        b = cb.batch_get_builds(ids=[build_id])["builds"][0]
        status = b["buildStatus"]
        phase = b.get("currentPhase")
        logger.info("  status=%s phase=%s", status, phase)
        if status != "IN_PROGRESS":
            if status != "SUCCEEDED":
                raise RuntimeError(f"CodeBuild build {build_id} ended with status {status}")
            return build_id


def _ensure_runtime(
    bac,
    runtime_name: str,
    image_uri: str,
    execution_role_arn: str,
    environment_variables: dict[str, str] | None = None,
) -> str:
    """Create or update the AgentCore Runtime and return its ARN."""
    existing = None
    paginator = bac.get_paginator("list_agent_runtimes")
    for page in paginator.paginate():
        for rt in page.get("agentRuntimes", []):
            if rt.get("agentRuntimeName") == runtime_name:
                existing = rt
                break
        if existing:
            break

    config: dict = {
        "agentRuntimeName": runtime_name,
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": image_uri}},
        "roleArn": execution_role_arn,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "protocolConfiguration": {"serverProtocol": "HTTP"},
    }
    if environment_variables:
        # ``environmentVariables`` is a first-class ``create_agent_runtime``
        # parameter (boto3 service model). Letting the runtime config carry
        # them avoids rebuilding the container image to flip a value.
        config["environmentVariables"] = environment_variables
    if existing:
        runtime_id = existing["agentRuntimeId"]
        logger.info("Updating existing runtime: %s", runtime_id)
        # ``update_agent_runtime`` keys the runtime by ``agentRuntimeId`` and
        # rejects ``agentRuntimeName`` (immutable after create), so drop it.
        update_config = {k: v for k, v in config.items() if k != "agentRuntimeName"}
        resp = bac.update_agent_runtime(agentRuntimeId=runtime_id, **update_config)
    else:
        logger.info("Creating new runtime: %s", runtime_name)
        resp = bac.create_agent_runtime(**config)
    return resp["agentRuntimeArn"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-name", default="agent_eval_runtime_dev")
    p.add_argument("--ecr-repo", default="agent-eval-runtime")
    p.add_argument("--codebuild-project", default="agent-eval-runtime-build")
    p.add_argument(
        "--source-bucket",
        default=None,
        help=(
            "S3 bucket that holds the zipped build source. Defaults to "
            "agent-eval-codebuild-src-<account>-<region> (globally unique). "
            "This is a single source of truth: the same name scopes the "
            "CodeBuild role's S3 permissions and receives the upload."
        ),
    )
    p.add_argument(
        "--image-tag",
        default=None,
        help=(
            "Immutable image tag. Defaults to the short git SHA of HEAD so "
            "each build is uniquely identifiable and CDK can pin an exact "
            "image. The ECR repo is IMMUTABLE, so a reused tag (e.g. 'latest') "
            "is rejected on re-push. Outside a git checkout, pass this "
            "explicitly."
        ),
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Bedrock inference-profile / model ID the agent invokes. Used to "
            "scope the runtime role's bedrock:InvokeModel permission to this "
            "model instead of Resource '*' (least-privilege)."
        ),
    )
    p.add_argument("--region", default=None, help="AWS region (defaults to session region)")
    p.add_argument(
        "--codebuild-role",
        default="agent-eval-codebuild-role",
        help="IAM role name for CodeBuild service role",
    )
    p.add_argument(
        "--runtime-role",
        default="agent-eval-agentcore-role",
        help="IAM role name for AgentCore Runtime execution role",
    )
    p.add_argument(
        "--register-runtime",
        action="store_true",
        help=(
            "Opt in to create/update a standalone AgentCore Runtime after the "
            "build (no-CDK quickstart path). Off by default: the runtime is "
            "owned by CDK (cdk/lib/agent_runtime_stack.py), and registering it "
            "here too causes CloudFormation drift. When set, pass the agent's "
            "data wiring via --env (DATA_BUCKET, DEALERS_TABLE, LANCEDB_PATH) "
            "and a runtime role that can read them."
        ),
    )
    p.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Environment variable to set on the AgentCore runtime (repeatable). "
            "Only used with --register-runtime. Sent via the "
            "``environmentVariables`` field of ``create_agent_runtime`` so "
            "config can be flipped without rebuilding the image."
        ),
    )
    args = p.parse_args()

    if args.image_tag is None:
        args.image_tag = _git_sha()
        if not args.image_tag:
            logger.error(
                "No --image-tag given and not in a git checkout. The ECR repo "
                "is IMMUTABLE, so an explicit unique tag is required."
            )
            return 2
        logger.info("Using git SHA as image tag: %s", args.image_tag)

    runtime_env: dict[str, str] = {}
    for kv in args.env:
        if "=" not in kv:
            logger.error("--env must be KEY=VALUE, got %r", kv)
            return 2
        k, v = kv.split("=", 1)
        runtime_env[k] = v

    session = boto3.Session(region_name=args.region) if args.region else boto3.Session()
    region = session.region_name
    if not region:
        logger.error("No AWS region configured. Pass --region or set AWS_REGION.")
        return 2
    account = session.client("sts").get_caller_identity()["Account"]
    logger.info("Deploying in account=%s region=%s", account, region)

    # Single source of truth for the build-source bucket name. Threaded into the
    # CodeBuild role's S3 grant and the upload below so the name is defined once.
    source_bucket = args.source_bucket or f"agent-eval-codebuild-src-{account}-{region}"

    iam = session.client("iam")
    s3 = session.client("s3")
    ecr = session.client("ecr")
    cb = session.client("codebuild")

    # 1. ECR repo
    repo_uri = _ensure_ecr_repo(ecr, args.ecr_repo, region, account)

    # 2. IAM roles
    # No managed policy: AWSCodeBuildDeveloperAccess grants codebuild:* plus
    # broad S3/CloudWatch/CodeCommit access meant for human operators who start
    # and view builds. The CodeBuild *service* role only needs the scoped
    # ECR/logs/S3 permissions attached inline below (least-privilege).
    cb_role_arn = _ensure_role(
        iam,
        args.codebuild_role,
        CODEBUILD_TRUST,
        [],
    )
    _ensure_inline_policy(
        iam,
        args.codebuild_role,
        "ecr-and-logs",
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    # SECURITY NOTE: ecr:GetAuthorizationToken is a non-resource-level
                    # action — AWS does not support scoping it below Resource "*".
                    # This is the correct implementation given AWS constraints.
                    # See: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonelasticcontainerregistry.html
                    "Sid": "ECRAuthToken",
                    "Effect": "Allow",
                    "Action": "ecr:GetAuthorizationToken",
                    "Resource": "*",
                },
                {
                    # ECR push/pull actions scoped to the specific repository
                    # that this CodeBuild project writes to.
                    "Sid": "ECRRepoPushPull",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "ecr:InitiateLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload",
                        "ecr:PutImage",
                    ],
                    "Resource": (f"arn:aws:ecr:{region}:{account}:repository/{args.ecr_repo}"),
                },
                {
                    # Scoped to the log group for this specific CodeBuild project.
                    "Sid": "CodeBuildLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                    ],
                    "Resource": (
                        f"arn:aws:logs:{region}:{account}:log-group:"
                        f"/aws/codebuild/{args.codebuild_project}:*"
                    ),
                },
                {
                    # S3 source bucket scoped to the bucket and its objects.
                    "Sid": "S3Source",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{source_bucket}",
                        f"arn:aws:s3:::{source_bucket}/*",
                    ],
                },
            ],
        },
    )

    runtime_role_arn = _ensure_role(
        iam,
        args.runtime_role,
        _agentcore_trust(account, region),
        [],
    )
    _ensure_inline_policy(
        iam,
        args.runtime_role,
        "ecr-logs-bedrock",
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    # SECURITY NOTE: ecr:GetAuthorizationToken is a non-resource-level
                    # action — AWS does not support scoping it below Resource "*".
                    # This is the correct implementation given AWS constraints.
                    # See: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonelasticcontainerregistry.html
                    "Sid": "ECRAuthToken",
                    "Effect": "Allow",
                    "Action": "ecr:GetAuthorizationToken",
                    "Resource": "*",
                },
                {
                    # ECR pull actions scoped to the specific repository
                    # that the AgentCore runtime pulls its image from.
                    "Sid": "ECRRepoPull",
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                    ],
                    "Resource": (f"arn:aws:ecr:{region}:{account}:repository/{args.ecr_repo}"),
                },
                {
                    # Scoped to the AgentCore runtime log group for this runtime.
                    "Sid": "RuntimeLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                    ],
                    "Resource": (
                        f"arn:aws:logs:{region}:{account}:log-group:"
                        f"/aws/bedrock-agentcore/runtimes/{args.runtime_name}:*"
                    ),
                },
                {
                    # SECURITY NOTE: X-Ray trace actions are non-resource-level —
                    # AWS does not support scoping xray:PutTraceSegments or
                    # xray:PutTelemetryRecords below Resource "*".
                    # This is the correct implementation given AWS constraints.
                    # See: https://docs.aws.amazon.com/service-authorization/latest/reference/list_awsx-ray.html
                    "Sid": "XRayTracing",
                    "Effect": "Allow",
                    "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
                    "Resource": "*",
                },
                {
                    # Scoped to exactly the models the agent invokes: the
                    # cross-region inference profile for the chat model, its
                    # underlying foundation model (the profile fans out to
                    # foundation-model ARNs at invoke time), and the Titan
                    # embedding model used for hybrid search. Mirrors the
                    # Amazon Bedrock policy in cdk/lib/agent_runtime_stack.py.
                    "Sid": "BedrockInvoke",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                        "bedrock:Converse",
                        "bedrock:ConverseStream",
                    ],
                    # The chat model is a cross-region inference profile
                    # (``eu.`` prefix): at invoke time it fans out to the
                    # underlying foundation-model ARN in whichever EU region it
                    # routes to, so the foundation-model resources are scoped to
                    # the ``eu-*`` partition rather than a single region.
                    # Pinning to one region would deny requests the profile
                    # routes elsewhere in the EU; ``eu-*`` still excludes every
                    # other partition. Mirrors cdk/lib/agent_runtime_stack.py.
                    "Resource": [
                        f"arn:aws:bedrock:{region}:{account}:inference-profile/{args.model_id}",
                        f"arn:aws:bedrock:eu-*::foundation-model/{_base_model_id(args.model_id)}",
                        f"arn:aws:bedrock:eu-*::foundation-model/{EMBEDDING_MODEL_ID}",
                    ],
                },
                *_data_source_statements(runtime_env, account, region),
            ],
        },
    )

    # 3. Source bucket and zip upload (name resolved once above)
    _ensure_source_bucket(s3, source_bucket, region)
    source_key = f"{args.ecr_repo}/source-{int(time.time())}.zip"
    payload = _zip_agent_dir()
    s3.put_object(Bucket=source_bucket, Key=source_key, Body=payload)
    logger.info("Uploaded source to s3://%s/%s (%d bytes)", source_bucket, source_key, len(payload))

    # 4. CodeBuild project
    _ensure_codebuild_project(
        cb,
        args.codebuild_project,
        cb_role_arn,
        source_bucket,
        source_key,
        region,
        account,
        args.ecr_repo,
        args.image_tag,
    )

    # 5. Run the build
    _run_build(cb, args.codebuild_project)
    image_uri = f"{repo_uri}:{args.image_tag}"
    logger.info("Built image: %s", image_uri)

    if not args.register_runtime:
        # Default path: build only. CDK (agent_runtime_stack) owns the runtime
        # and its role/env wiring. Feed this image URI to `cdk deploy`.
        logger.info("Build complete; runtime registration is owned by CDK (use deploy_stack.py)")
        print(json.dumps({"image_uri": image_uri}, indent=2))
        return 0

    # 6. Register runtime (opt-in, no-CDK quickstart only)
    bac = session.client("bedrock-agentcore-control")
    runtime_arn = _ensure_runtime(
        bac,
        args.runtime_name,
        image_uri,
        runtime_role_arn,
        environment_variables=runtime_env or None,
    )
    logger.info("Runtime ready: %s", runtime_arn)
    print(
        json.dumps(
            {
                "image_uri": image_uri,
                "runtime_arn": runtime_arn,
                "runtime_name": args.runtime_name,
                "region": region,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
