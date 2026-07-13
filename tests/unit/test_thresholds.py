# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for evaluation thresholds."""

from agentic_evaluation.thresholds import (
    DEV_THRESHOLDS,
    EVALUATION_THRESHOLDS,
    PRODUCTION_THRESHOLDS,
    STAGING_THRESHOLDS,
    EvaluationThresholds,
)


class TestEvaluationThresholds:
    """Tests for EvaluationThresholds."""

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        thresholds = EVALUATION_THRESHOLDS

        # Layer 1
        assert thresholds.tool_selection_accuracy == 0.95
        assert thresholds.tool_parameter_accuracy == 0.95

        # Layer 2
        assert thresholds.helpfulness_score == 0.83
        assert thresholds.reasoning_coherence == 0.85

        # Layer 3
        assert thresholds.goal_success_rate == 0.90
        assert thresholds.output_quality_score == 0.90

    def test_validate_layer_1_pass(self) -> None:
        """Test Layer 1 validation passes with good scores."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_layer_1(
            tool_accuracy=0.96,
            param_accuracy=0.97,
        )
        assert result is True

    def test_validate_layer_1_fail(self) -> None:
        """Test Layer 1 validation fails with low scores."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_layer_1(
            tool_accuracy=0.94,  # Below 0.95
            param_accuracy=0.97,
        )
        assert result is False

    def test_validate_layer_2_pass(self) -> None:
        """Test Layer 2 validation passes."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_layer_2(
            helpfulness=0.92,
            coherence=0.90,
        )
        assert result is True

    def test_validate_layer_2_fail(self) -> None:
        """Test Layer 2 validation fails."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_layer_2(
            helpfulness=0.50,  # Below 0.83
            coherence=0.90,
        )
        assert result is False

    def test_validate_layer_3_pass(self) -> None:
        """Test Layer 3 validation passes."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_layer_3(
            goal_success=0.95,
            output_quality=0.92,
        )
        assert result is True

    def test_validate_layer_3_fail(self) -> None:
        """Test Layer 3 validation fails."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_layer_3(
            goal_success=0.88,  # Below 0.90
            output_quality=0.92,
        )
        assert result is False

    def test_validate_all_layers_pass(self) -> None:
        """Test all layers pass with good scores."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_all_layers(
            tool_accuracy=0.96,
            param_accuracy=0.97,
            helpfulness=0.92,
            coherence=0.90,
            goal_success=0.95,
            output_quality=0.92,
        )

        assert result["layer_1_passed"] is True
        assert result["layer_2_passed"] is True
        assert result["layer_3_passed"] is True
        assert result["all_passed"] is True

    def test_validate_all_layers_fail_one(self) -> None:
        """Test that failing one layer fails overall."""
        thresholds = EVALUATION_THRESHOLDS
        result = thresholds.validate_all_layers(
            tool_accuracy=0.96,
            param_accuracy=0.97,
            helpfulness=0.50,  # Fails Layer 2 (below 0.83)
            coherence=0.90,
            goal_success=0.95,
            output_quality=0.92,
        )

        assert result["layer_1_passed"] is True
        assert result["layer_2_passed"] is False
        assert result["layer_3_passed"] is True
        assert result["all_passed"] is False

    def test_to_dict_serialization(self) -> None:
        """Test threshold serialization to dictionary."""
        thresholds = EVALUATION_THRESHOLDS
        as_dict = thresholds.to_dict()

        assert as_dict["tool_selection_accuracy"] == 0.95
        assert as_dict["helpfulness_score"] == 0.83
        assert as_dict["goal_success_rate"] == 0.90
        assert "hallucination_rate" in as_dict
        assert as_dict["domain_aggregate_score"] == 0.90

    def test_custom_thresholds(self) -> None:
        """Test creating custom thresholds."""
        custom = EvaluationThresholds(
            tool_selection_accuracy=0.98,
            helpfulness_score=0.95,
        )

        assert custom.tool_selection_accuracy == 0.98
        assert custom.helpfulness_score == 0.95
        # Other fields use defaults
        assert custom.goal_success_rate == 0.90

    def test_environment_specific_thresholds(self) -> None:
        """Test environment-specific threshold configurations."""
        # Dev has lower thresholds
        assert DEV_THRESHOLDS.tool_selection_accuracy == 0.90
        assert (
            DEV_THRESHOLDS.tool_selection_accuracy < PRODUCTION_THRESHOLDS.tool_selection_accuracy
        )

        # Staging is between dev and prod
        assert (
            DEV_THRESHOLDS.tool_selection_accuracy
            < STAGING_THRESHOLDS.tool_selection_accuracy
            < PRODUCTION_THRESHOLDS.tool_selection_accuracy
        )

        # Production uses strict defaults
        assert PRODUCTION_THRESHOLDS.tool_selection_accuracy == 0.95
