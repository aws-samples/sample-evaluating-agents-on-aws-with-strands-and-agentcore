# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Contents contract for the Lambda deployment packages.

``Code.from_asset`` zips a directory verbatim, so without an exclusion list a
developer's local ``.venv`` and ``__pycache__`` ship inside the deployment package
and make the asset hash depend on the machine that synthesized it. These tests
stage the real assets and assert the handler is present and the local development
artifacts are not.
"""

import sys
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk import aws_lambda as lambda_

CDK_DIR = Path(__file__).resolve().parents[2] / "examples" / "vehicle-auction-agent" / "cdk"
sys.path.insert(0, str(CDK_DIR))

from lib.lambda_assets import function_code  # noqa: E402

_FORBIDDEN = {".venv", "__pycache__"}


def _staged_asset(function_name: str, outdir: Path) -> Path:
    """Synthesize a throwaway stack and return the staged asset directory."""
    app = cdk.App(outdir=str(outdir))
    stack = cdk.Stack(app, "asset-stack")
    lambda_.Function(
        stack,
        "Fn",
        runtime=lambda_.Runtime.PYTHON_3_14,
        handler="handler.lambda_handler",
        code=function_code(function_name),
    )
    app.synth()
    staged = [entry for entry in outdir.iterdir() if entry.name.startswith("asset.")]
    assert len(staged) == 1, f"expected exactly one staged asset, got {staged}"
    return staged[0]


@pytest.mark.parametrize("function_name", ["data_ingestion", "dealer_api"])
def test_deployment_package_carries_the_handler(function_name: str, tmp_path: Path) -> None:
    assert (_staged_asset(function_name, tmp_path) / "handler.py").is_file()


@pytest.mark.parametrize("function_name", ["data_ingestion", "dealer_api"])
def test_deployment_package_excludes_local_dev_artifacts(
    function_name: str, tmp_path: Path
) -> None:
    """A local venv or bytecode cache in the source tree must not be deployed."""
    asset = _staged_asset(function_name, tmp_path)
    leaked = sorted(
        str(path.relative_to(asset))
        for path in asset.rglob("*")
        if path.name in _FORBIDDEN or path.suffix == ".pyc"
    )
    assert not leaked, f"local development artifacts leaked into the package: {leaked}"


def test_asset_path_is_independent_of_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved from the module, so synthesis works from any working directory."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    outdir = tmp_path / "out"
    outdir.mkdir()
    assert (_staged_asset("dealer_api", outdir) / "handler.py").is_file()
