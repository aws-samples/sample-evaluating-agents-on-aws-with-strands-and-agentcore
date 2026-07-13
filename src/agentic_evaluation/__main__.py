# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Entry point for ``python -m agentic_evaluation``."""

import sys

from agentic_evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
