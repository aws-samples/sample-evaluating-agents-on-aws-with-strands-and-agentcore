# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda source bundling for the reference deployment.

``Code.from_asset`` zips a directory verbatim, so a developer's local ``.venv``
and ``__pycache__`` would otherwise ship inside the deployment package. They also
make the asset hash depend on the machine that ran ``cdk synth``, which surfaces
as a spurious Lambda update in every ``cdk diff``. Both handlers import only the
standard library and ``boto3`` (already present in the managed runtime), so no
dependencies need to be vendored.

Keeping the exclusion list and the path resolution here gives every function one
home for both, rather than repeating them per stack.
"""

from pathlib import Path

from aws_cdk import aws_lambda as lambda_

_FUNCTIONS_DIR = Path(__file__).resolve().parents[2] / "lambda" / "functions"

# Local development artifacts that must never reach a deployment package.
_EXCLUDED = [
    ".DS_Store",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "*.egg-info",
    "*.pyc",
    "__pycache__",
]


def function_code(function_name: str) -> lambda_.Code:
    """Bundle a Lambda handler directory without local development artifacts.

    Args:
        function_name: Directory name under
            ``examples/vehicle-auction-agent/lambda/functions``.

    Returns:
        Asset code to pass as ``lambda_.Function(code=...)``. The path is resolved
        from this module, so synthesis does not depend on the working directory.
    """
    return lambda_.Code.from_asset(str(_FUNCTIONS_DIR / function_name), exclude=_EXCLUDED)
