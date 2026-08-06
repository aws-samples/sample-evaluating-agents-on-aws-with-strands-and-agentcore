# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Metric definitions for the agent-evaluation dashboard and alarms.

Held apart from ``monitoring_stack`` so that "what is measured, in which
namespace, under which dimensions" has one home. The dashboard layout and the
alarm thresholds both read the groups returned here instead of restating the
namespace/dimension pairs they need, which is what previously let a widget and
an alarm drift onto two different definitions of the same metric.
"""

from dataclasses import dataclass

import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)

# Amazon Bedrock AgentCore service metrics; see
# https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html
#
# The lowercase "bedrock-agentcore" namespace in the observability docs is for
# OTEL/EMF metrics emitted by instrumented agent code, not for these service
# metrics, and carries no data for them.
_AGENTCORE_NAMESPACE = "AWS/Bedrock-AgentCore"

# Bound once at import rather than per call: a ``Duration`` in a default
# argument is a function call in a default (ruff B008), and every metric below
# samples on one of these three intervals.
_ONE_MINUTE = cdk.Duration.minutes(1)
_FIVE_MINUTES = cdk.Duration.minutes(5)
_ONE_HOUR = cdk.Duration.hours(1)


def _metric(
    namespace: str,
    metric_name: str,
    dimensions: dict[str, str],
    *,
    statistic: str = "Sum",
    period: cdk.Duration = _FIVE_MINUTES,
) -> cloudwatch.Metric:
    """Build one CloudWatch metric.

    Args:
        namespace: The metric's AWS namespace.
        metric_name: Name of the metric within that namespace.
        dimensions: The complete dimension set that keys the metric. CloudWatch
            treats a partial set as a different metric, which reports no data.
        statistic: Aggregation to graph or alarm on.
        period: Sampling interval.

    Returns:
        The metric, ready to hand to a widget or an alarm.
    """
    return cloudwatch.Metric(
        namespace=namespace,
        metric_name=metric_name,
        dimensions_map=dimensions,
        statistic=statistic,
        period=period,
    )


@dataclass(frozen=True, slots=True)
class AgentRuntimeMetrics:
    """AgentCore Runtime service metrics for one runtime ARN."""

    invocations: cloudwatch.Metric
    errors: cloudwatch.Metric
    latency_p50: cloudwatch.Metric
    latency_p95: cloudwatch.Metric
    latency_p99: cloudwatch.Metric


@dataclass(frozen=True, slots=True)
class MemoryMetrics:
    """AgentCore Memory extraction metrics for one memory ARN."""

    extractions: cloudwatch.Metric
    extraction_errors: cloudwatch.Metric


@dataclass(frozen=True, slots=True)
class DealerApiMetrics:
    """Request, error, latency, and table metrics for the dealer API."""

    requests: cloudwatch.Metric
    errors_4xx: cloudwatch.Metric
    errors_5xx: cloudwatch.Metric
    integration_latency: cloudwatch.Metric
    total_latency: cloudwatch.Metric
    read_capacity: cloudwatch.Metric
    throttles: cloudwatch.IMetric


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    """Metrics for the ingestion function and the schedule that triggers it."""

    ingestion_invocations: cloudwatch.Metric
    ingestion_errors: cloudwatch.Metric
    schedule_invocations: cloudwatch.Metric
    schedule_failures: cloudwatch.Metric


def agent_runtime_metrics(runtime_arn: str, env_name: str) -> AgentRuntimeMetrics:
    """Read the AgentCore Runtime service metrics for one deployed runtime.

    Args:
        runtime_arn: ARN of the deployed AgentCore Runtime.
        env_name: Environment suffix, part of the runtime's metric ``Name``.

    Returns:
        The runtime's invocation, error, and latency-percentile metrics.
    """
    # Runtime service metrics are keyed on all three of Resource + Operation +
    # Name; dropping Name yields no datapoints.
    dimensions = {
        "Resource": runtime_arn,
        "Operation": "InvokeAgentRuntime",
        "Name": f"agent_eval_runtime_{env_name}::DEFAULT",
    }
    return AgentRuntimeMetrics(
        invocations=_metric(_AGENTCORE_NAMESPACE, "Invocations", dimensions),
        errors=_metric(_AGENTCORE_NAMESPACE, "Errors", dimensions),
        latency_p50=_metric(_AGENTCORE_NAMESPACE, "Duration", dimensions, statistic="p50"),
        latency_p95=_metric(_AGENTCORE_NAMESPACE, "Duration", dimensions, statistic="p95"),
        latency_p99=_metric(_AGENTCORE_NAMESPACE, "Duration", dimensions, statistic="p99"),
    )


def memory_metrics(memory_arn: str) -> MemoryMetrics:
    """Read the AgentCore Memory extraction metrics for one memory store.

    Args:
        memory_arn: ARN of the deployed AgentCore Memory resource.

    Returns:
        The store's extraction-volume and extraction-error metrics.
    """
    return MemoryMetrics(
        # Extraction volume comes from CreationCount/MemoryRecordsExtracted, not
        # Invocations: AgentCore only emits Invocations for Extraction alongside
        # per-strategy StrategyId/StrategyType dimensions, so an Invocations
        # metric keyed on Resource+Operation alone reports no data at all.
        extractions=_metric(
            _AGENTCORE_NAMESPACE,
            "CreationCount",
            {"Resource": memory_arn, "ItemType": "MemoryRecordsExtracted"},
        ),
        extraction_errors=_metric(
            _AGENTCORE_NAMESPACE,
            "Errors",
            {"Resource": memory_arn, "Operation": "Extraction"},
        ),
    )


def _table_throttles(table_dimensions: dict[str, str]) -> cloudwatch.IMetric:
    """Sum a table's read and write throttling into one series.

    UserErrors is not a throttle proxy — it also counts validation and other
    caller faults — so throttling is read from the two dedicated metrics.

    Args:
        table_dimensions: Dimension set naming the DynamoDB table.

    Returns:
        A math expression totalling read and write throttle events.
    """
    return cloudwatch.MathExpression(
        expression="read + write",
        using_metrics={
            "read": _metric("AWS/DynamoDB", "ReadThrottleEvents", table_dimensions),
            "write": _metric("AWS/DynamoDB", "WriteThrottleEvents", table_dimensions),
        },
        label="Throttle Events",
        period=_FIVE_MINUTES,
    )


def dealer_api_metrics(env_name: str) -> DealerApiMetrics:
    """Read the dealer API's gateway metrics and its table's capacity metrics.

    Args:
        env_name: Environment suffix in the API, stage, and table names.

    Returns:
        The API's request, error, and latency metrics plus the dealer table's
        consumed read capacity and throttling.
    """
    stage = {"ApiName": f"agent-eval-dealer-api-{env_name}", "Stage": env_name}
    table = {"TableName": f"agent-eval-dealers-{env_name}"}
    return DealerApiMetrics(
        requests=_metric("AWS/ApiGateway", "Count", stage, period=_ONE_MINUTE),
        errors_4xx=_metric("AWS/ApiGateway", "4XXError", stage),
        errors_5xx=_metric("AWS/ApiGateway", "5XXError", stage),
        integration_latency=_metric(
            "AWS/ApiGateway", "IntegrationLatency", stage, statistic="Average"
        ),
        total_latency=_metric("AWS/ApiGateway", "Latency", stage, statistic="Average"),
        read_capacity=_metric("AWS/DynamoDB", "ConsumedReadCapacityUnits", table),
        throttles=_table_throttles(table),
    )


def pipeline_metrics(env_name: str) -> PipelineMetrics:
    """Read the data-ingestion function and daily-refresh schedule metrics.

    Args:
        env_name: Environment suffix in the function and rule names.

    Returns:
        The ingestion function's invocation and error metrics alongside the
        EventBridge rule's invocation and failure counts.
    """
    function = {"FunctionName": f"agent-eval-data-ingestion-{env_name}"}
    rule = {"RuleName": f"agent-eval-daily-refresh-{env_name}"}
    return PipelineMetrics(
        ingestion_invocations=_metric("AWS/Lambda", "Invocations", function),
        ingestion_errors=_metric("AWS/Lambda", "Errors", function),
        schedule_invocations=_metric("AWS/Events", "Invocations", rule, period=_ONE_HOUR),
        schedule_failures=_metric("AWS/Events", "FailedInvocations", rule, period=_ONE_HOUR),
    )
