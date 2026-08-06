# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Generate test cases using strands-agents-evals ExperimentGenerator.

Bootstraps evaluation cases from a high-level description of the agent's
capabilities. Useful for broad coverage early on — supplement with
hand-crafted cases targeting known edge cases as your evaluation matures.

Usage:
    python -m agentic_evaluation.generate_cases --num-cases 20 --output cases.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from strands_evals.evaluators import OutputEvaluator
from strands_evals.generators import ExperimentGenerator

from agentic_evaluation.config import load_config

logger = logging.getLogger(__name__)

AGENT_CONTEXT = (
    "A dealer stock search agent for an online car marketplace. "
    "Dealers search vehicle inventory using natural language queries. "
    "The agent has 8 tools: get_schema (data schema discovery), "
    "search_vehicles (structured filtering across 89+ attributes), "
    "hybrid_search (semantic + structured), get_embedding (vector generation), "
    "filter_by_distance (location-based filtering), get_bids (bid history), "
    "get_dealer_profile (dealer preferences and location), and "
    "run_sql (complex compound query fallback). "
    "Vehicles have attributes like make, model, year, mileage, fuel_type, "
    "body_type, transmission, colour, price, and location coordinates."
)

TASK_DESCRIPTION = (
    "Handle dealer queries about vehicle inventory including: "
    "structured searches (e.g. 'BMW 3 Series under 50k miles'), "
    "semantic searches (e.g. 'something sporty for a family'), "
    "location-based filtering (e.g. 'within 30 miles of my dealership'), "
    "bid history lookups, dealer profile queries, and multi-turn "
    "refinements. The agent should refuse bidding actions and "
    "winning probability claims."
)


async def generate_cases_async(
    num_cases: int = 20,
    model: str | None = None,
) -> list[dict]:
    """Generate test cases using ExperimentGenerator.

    Args:
        num_cases: Number of test cases to generate.
        model: Model ID for generation. Falls back to config then default.

    Returns:
        List of generated case dicts with name, input, and expected_output.
    """
    if model is None:
        cfg = load_config()
        model = cfg.judge_model or os.environ.get(
            "JUDGE_MODEL_ID", "eu.anthropic.claude-sonnet-4-6"
        )

    generator = ExperimentGenerator(
        input_type=str,
        output_type=str,
        include_expected_output=True,
        include_expected_trajectory=True,
        model=model,
    )

    experiment = await generator.from_context_async(
        context=AGENT_CONTEXT,
        task_description=TASK_DESCRIPTION,
        num_cases=num_cases,
        # The class, not an instance: ExperimentGenerator looks the argument up
        # in its `_default_evaluators` table (keyed by class) and then calls it
        # to build the instance with the generated rubric. Its annotation says
        # `Evaluator`, which contradicts that usage — an upstream bug, so the
        # correct call has to be silenced here.
        evaluator=OutputEvaluator,  # pyright: ignore[reportArgumentType]
    )

    return [
        {
            "name": case.name,
            "input": case.input,
            "expected_output": case.expected_output,
            "expected_trajectory": case.expected_trajectory,
        }
        for case in experiment.cases
    ]


def main() -> None:
    """Generate cases from the CLI, writing JSON to ``--output`` or stdout."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation test cases with ExperimentGenerator"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=20,
        help="Number of test cases to generate (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for generation (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cases = asyncio.run(generate_cases_async(args.num_cases, args.model))

    output = json.dumps(cases, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        logger.info("Wrote %d cases to %s", len(cases), args.output)
    else:
        # Without --output the generated JSON *is* this command's result, so it
        # goes to stdout to stay pipeable. Diagnostics use `logger` above.
        print(output)  # noqa: T201


if __name__ == "__main__":
    main()
