# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Test case registry for agent evaluation.

Organizes test cases by category:
- Happy path: Common, expected queries
- Edge cases: Unusual but valid queries
- Safety/Guardrails: Queries the agent should refuse
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TestCategory(Enum):
    """Test case categories."""

    __test__ = False

    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    SAFETY = "safety"
    MULTI_TURN = "multi_turn"
    PERFORMANCE = "performance"


class EvaluationLayer(Enum):
    """Evaluation layers that apply to this test.

    The layer names below are canonical (they match the accompanying blog
    post). They map onto standard industry vocabulary as follows:

    - LAYER_1_TOOL_USAGE  -> "tool correctness" / "tool-call accuracy"
      (DeepEval ``ToolCorrectnessMetric``, Ragas ``ToolCallAccuracy``)
    - LAYER_2_REASONING   -> "process evaluation" (trajectory-level
      LLM-as-judge scoring of the agent's decision-making)
    - LAYER_3_OUTPUT_QUALITY -> "outcome evaluation" / "goal success"
      (Ragas ``AgentGoalAccuracy``, DeepEval ``TaskCompletion``)

    eval_config.yaml accepts either the canonical name or the standard
    alias (see ``agentic_evaluation.config._LAYER_MAP``).
    """

    LAYER_1_TOOL_USAGE = "layer_1_tool_usage"
    LAYER_2_REASONING = "layer_2_reasoning"
    LAYER_3_OUTPUT_QUALITY = "layer_3_output_quality"


@dataclass
class TestCase:
    """A single test case for agent evaluation."""

    id: str
    query: str
    category: TestCategory
    expected_tools: list[str]
    expected_behavior: str
    evaluation_layers: list[EvaluationLayer]
    tags: list[str]
    reference_solution: dict[str, Any] | None = None
    expected_tool_parameters: dict[str, dict[str, Any]] | None = None
    # Human-authored success criteria for GoalSuccessRateEvaluator. When set,
    # the judge scores against these explicit assertions (assertion mode)
    # instead of inferring goals from the conversation (basic mode), which on
    # multi-turn transcripts can hallucinate goals and fail correct answers.
    expected_assertion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert test case to dictionary for serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category.value,
            "expected_tools": self.expected_tools,
            "expected_behavior": self.expected_behavior,
            "evaluation_layers": [layer.value for layer in self.evaluation_layers],
            "tags": self.tags,
            "reference_solution": self.reference_solution,
            "expected_tool_parameters": self.expected_tool_parameters,
            "expected_assertion": self.expected_assertion,
        }


class TestCaseRegistry:
    """Registry of test cases for agent evaluation.

    Create a registry from eval_config.yaml (recommended) or build one
    programmatically by calling add_test_case().

    Example::

        # From YAML config (recommended for SDK users)
        cfg = load_config("eval_config.yaml")
        registry = TestCaseRegistry.from_config(cfg.test_cases)

        # Programmatic
        registry = TestCaseRegistry()
        registry.add_test_case(TestCase(id="tc_001", ...))
    """

    def __init__(self) -> None:
        """Initialize an empty test case registry.

        Test cases are loaded from eval_config.yaml via from_config(),
        or added programmatically via add_test_case().
        """
        self.test_cases: list[TestCase] = []

    @classmethod
    def from_config(cls, test_cases: list[TestCase]) -> "TestCaseRegistry":
        """Create a registry from a pre-loaded list of test cases.

        Use this with agentic_evaluation.config.load_config() to load test cases
        from eval_config.yaml instead of the hardcoded defaults.

        Example:
            cfg = load_config("eval_config.yaml")
            registry = TestCaseRegistry.from_config(cfg.test_cases)
        """
        instance = cls.__new__(cls)
        instance.test_cases = list(test_cases)
        return instance

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the registry."""
        self.test_cases.append(test_case)

    def get_by_category(self, category: TestCategory) -> list[TestCase]:
        """Get all test cases in a category."""
        return [tc for tc in self.test_cases if tc.category == category]

    def get_by_tag(self, tag: str) -> list[TestCase]:
        """Get all test cases with a specific tag."""
        return [tc for tc in self.test_cases if tag in tc.tags]

    def get_by_id(self, test_id: str) -> TestCase | None:
        """Get a specific test case by ID."""
        for tc in self.test_cases:
            if tc.id == test_id:
                return tc
        return None

    def to_json(self) -> list[dict[str, Any]]:
        """Export all test cases as JSON."""
        return [tc.to_dict() for tc in self.test_cases]

    def __len__(self) -> int:
        """Return number of test cases in registry."""
        return len(self.test_cases)
