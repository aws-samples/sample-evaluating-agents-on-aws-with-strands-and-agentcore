# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Monitoring Stack - Amazon CloudWatch Dashboards + Alarms.

Creates operational monitoring for agent evaluation:
- Dashboard with metrics for agent runtime, dealer API, and data pipeline
- AWS service metrics from AWS Lambda, Amazon API Gateway, Amazon DynamoDB, Amazon EventBridge, and Amazon Bedrock AgentCore
- Alarms with SNS notifications for critical thresholds

The metrics themselves are defined in ``lib.monitoring_metrics``; this module
decides how they are laid out and which thresholds page an operator.

Cost considerations:
This stack creates billable Amazon CloudWatch dashboards and alarms. It does
not create unpublished custom evaluation metrics. Estimated default cost is
roughly $1-5/month, depending on the account's free-tier usage.

Cleanup:
Important: Destroying this stack deletes alarm history and dashboard
configuration. If you need to preserve any dashboard JSON or alarm
configurations, create a backup first. To remove: ``cdk destroy MonitoringStack``
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import aws_cdk as cdk
from aws_cdk import (
    Stack,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_cloudwatch_actions as cw_actions,
)
from aws_cdk import (
    aws_sns as sns,
)
from constructs import Construct

from lib.monitoring_metrics import (
    AgentRuntimeMetrics,
    DealerApiMetrics,
    MemoryMetrics,
    PipelineMetrics,
    agent_runtime_metrics,
    dealer_api_metrics,
    memory_metrics,
    pipeline_metrics,
)

# A dashboard row is 24 units wide: one full-width graph or two half-width ones.
_FULL_WIDTH = 24
_HALF_WIDTH = 12


class _Axis(NamedTuple):
    """One side of a two-axis graph widget: a series and the axis label."""

    metric: cloudwatch.IMetric
    label: str


@dataclass(frozen=True, slots=True)
class _Metrics:
    """Every metric group this stack graphs and alarms on.

    ``agent`` and ``memory`` are ``None`` when the corresponding AgentCore
    resource was not deployed in this environment.
    """

    agent: AgentRuntimeMetrics | None
    memory: MemoryMetrics | None
    dealer_api: DealerApiMetrics
    pipeline: PipelineMetrics


@dataclass(frozen=True, slots=True)
class _AlarmSpec:
    """One alarm to create. Every alarm notifies the same evaluation topic."""

    construct_id: str
    name_suffix: str
    metric: cloudwatch.IMetric
    threshold: float
    description: str
    evaluation_periods: int = 1
    comparison_operator: cloudwatch.ComparisonOperator = (
        cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD
    )


def _dual_axis_graph(title: str, left: _Axis, right: _Axis, width: int) -> cloudwatch.GraphWidget:
    """Graph two related series against their own axes.

    Args:
        title: Widget title.
        left: Series and label for the left axis.
        right: Series and label for the right axis.
        width: Widget width in dashboard grid units.

    Returns:
        The widget, with both axes anchored at zero so a flat line at zero reads
        as "no errors" rather than as an autoscaled baseline.
    """
    return cloudwatch.GraphWidget(
        title=title,
        left=[left.metric],
        right=[right.metric],
        width=width,
        left_y_axis=cloudwatch.YAxisProps(min=0, label=left.label),
        right_y_axis=cloudwatch.YAxisProps(min=0, label=right.label),
        legend_position=cloudwatch.LegendPosition.BOTTOM,
    )


def _runtime_absent_row() -> list[cloudwatch.IWidget]:
    """Explain the missing agent panels when no runtime was deployed.

    Returns:
        A single text row, so the dashboard states why the agent graphs are
        absent instead of rendering panels that can never carry data.
    """
    return [
        cloudwatch.TextWidget(
            markdown=(
                "## Agent Runtime Not Deployed\n\n"
                "The AgentCore Runtime was not deployed in this environment. "
                "Re-deploy with `agent_image_uri` context to enable agent runtime metrics."
            ),
            width=_FULL_WIDTH,
            height=3,
        )
    ]


def _agent_rows(metrics: _Metrics) -> list[list[cloudwatch.IWidget]]:
    """Lay out the AgentCore runtime and memory rows.

    Args:
        metrics: The metric groups for this environment.

    Returns:
        One row per graph, or the placeholder row when no runtime was deployed.
    """
    agent = metrics.agent
    if agent is None:
        return [_runtime_absent_row()]

    rows: list[list[cloudwatch.IWidget]] = [
        [
            _dual_axis_graph(
                "Agent Runtime - Invocations vs Errors",
                _Axis(agent.invocations, "Invocations"),
                _Axis(agent.errors, "Errors"),
                _FULL_WIDTH,
            )
        ],
        [
            cloudwatch.GraphWidget(
                title="Agent Runtime - Latency Percentiles (ms)",
                left=[agent.latency_p50, agent.latency_p95, agent.latency_p99],
                width=_FULL_WIDTH,
                left_y_axis=cloudwatch.YAxisProps(min=0, label="Latency (ms)"),
                legend_position=cloudwatch.LegendPosition.BOTTOM,
            )
        ],
    ]
    if metrics.memory is not None:
        rows.append(
            [
                _dual_axis_graph(
                    "AgentCore Memory - Extractions vs Errors",
                    _Axis(metrics.memory.extractions, "Extractions"),
                    _Axis(metrics.memory.extraction_errors, "Errors"),
                    _FULL_WIDTH,
                )
            ]
        )
    return rows


def _service_rows(metrics: _Metrics) -> list[list[cloudwatch.IWidget]]:
    """Lay out the dealer API and data pipeline rows.

    Args:
        metrics: The metric groups for this environment.

    Returns:
        Three rows of two half-width graphs: API traffic and errors, API latency
        against table pressure, then ingestion against its schedule.
    """
    api = metrics.dealer_api
    pipeline = metrics.pipeline
    return [
        [
            cloudwatch.GraphWidget(
                title="Dealer API - Requests per Second",
                left=[api.requests],
                width=_HALF_WIDTH,
                left_y_axis=cloudwatch.YAxisProps(min=0, label="Requests/sec"),
            ),
            _dual_axis_graph(
                "Dealer API - HTTP Error Rates",
                _Axis(api.errors_4xx, "4xx Errors"),
                _Axis(api.errors_5xx, "5xx Errors"),
                _HALF_WIDTH,
            ),
        ],
        [
            _dual_axis_graph(
                "Dealer API - Latency (ms)",
                _Axis(api.integration_latency, "Integration Latency (ms)"),
                _Axis(api.total_latency, "Total Latency (ms)"),
                _HALF_WIDTH,
            ),
            _dual_axis_graph(
                "DynamoDB - Read Capacity & Throttles",
                _Axis(api.read_capacity, "Read Capacity Units"),
                _Axis(api.throttles, "Throttle Events"),
                _HALF_WIDTH,
            ),
        ],
        [
            _dual_axis_graph(
                "Data Ingestion Lambda - Invocations & Errors",
                _Axis(pipeline.ingestion_invocations, "Invocations"),
                _Axis(pipeline.ingestion_errors, "Errors"),
                _HALF_WIDTH,
            ),
            _dual_axis_graph(
                "EventBridge - Daily Refresh Invocations",
                _Axis(pipeline.schedule_invocations, "Invocations"),
                _Axis(pipeline.schedule_failures, "Failed"),
                _HALF_WIDTH,
            ),
        ],
    ]


def _agent_alarm_specs(metrics: _Metrics) -> list[_AlarmSpec]:
    """Describe the AgentCore alarms, skipping resources that were not deployed.

    Args:
        metrics: The metric groups for this environment.

    Returns:
        The runtime error-rate and latency alarms, plus the memory extraction
        alarm, for whichever of the two resources exists.
    """
    specs: list[_AlarmSpec] = []
    if metrics.agent is not None:
        specs += [
            _AlarmSpec(
                construct_id="AgentErrorRateAlarm",
                name_suffix="agent-error-rate",
                metric=metrics.agent.errors,
                threshold=5,
                evaluation_periods=2,
                description="Agent runtime errors exceeded threshold of 5 per 5 minutes",
            ),
            _AlarmSpec(
                construct_id="AgentLatencyP99Alarm",
                name_suffix="agent-latency-p99",
                metric=metrics.agent.latency_p99,
                # Keep this threshold aligned with alert_latency_p99_ms in
                # eval_config.yaml.
                threshold=30000,
                evaluation_periods=2,
                description="Agent runtime P99 latency exceeded 30 seconds",
            ),
        ]
    if metrics.memory is not None:
        specs.append(
            _AlarmSpec(
                construct_id="MemoryExtractionErrorAlarm",
                name_suffix="memory-extraction-error",
                metric=metrics.memory.extraction_errors,
                threshold=0,
                description="AgentCore Memory extraction encountered errors",
            )
        )
    return specs


def _service_alarm_specs(metrics: _Metrics) -> list[_AlarmSpec]:
    """Describe the dealer API and data pipeline alarms.

    Args:
        metrics: The metric groups for this environment.

    Returns:
        The four alarms that always exist, since their resources are deployed in
        every environment.
    """
    api = metrics.dealer_api
    pipeline = metrics.pipeline
    return [
        _AlarmSpec(
            construct_id="ApiErrorRateAlarm",
            name_suffix="api-error-rate",
            metric=api.errors_5xx,
            threshold=2,
            evaluation_periods=2,
            description=("Dealer API 5xx error count exceeded threshold of 2 errors per 5 minutes"),
        ),
        _AlarmSpec(
            construct_id="DynamoDBThrottleAlarm",
            name_suffix="dynamodb-throttle",
            metric=api.throttles,
            threshold=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            description="DynamoDB throttling events detected",
        ),
        _AlarmSpec(
            construct_id="IngestionErrorAlarm",
            name_suffix="ingestion-error",
            metric=pipeline.ingestion_errors,
            threshold=0,
            description="Data ingestion Lambda encountered errors",
        ),
        _AlarmSpec(
            construct_id="EventBridgeFailedAlarm",
            name_suffix="eventbridge-failed",
            metric=pipeline.schedule_failures,
            threshold=0,
            description="EventBridge rule failed to invoke data ingestion",
        ),
    ]


class MonitoringStack(Stack):
    """Stack for CloudWatch dashboards and alarms."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        evaluation_topic: sns.ITopic,
        agent_runtime_arn: str | None = None,
        agent_memory_arn: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Provision the evaluation dashboard and its alarms.

        Args:
            scope: The parent construct, normally the CDK ``App``.
            construct_id: Logical id of the stack.
            evaluation_topic: Topic every alarm in this stack notifies.
            agent_runtime_arn: ARN of the deployed AgentCore Runtime. When
                omitted, the agent panels are replaced by a note explaining
                their absence and the agent alarms are not created.
            agent_memory_arn: ARN of the deployed AgentCore Memory resource,
                omitted on the same terms.
            **kwargs: Passed through to ``Stack``, notably ``env``. The target
                environment is read from the ``environment`` context value.
        """
        super().__init__(scope, construct_id, **kwargs)

        # Get environment from context
        env_name = self.node.try_get_context("environment") or "dev"

        dashboard = cloudwatch.Dashboard(
            self,
            "EvaluationDashboard",
            dashboard_name=f"agent-eval-{env_name}",
        )

        metrics = _Metrics(
            agent=agent_runtime_metrics(agent_runtime_arn, env_name) if agent_runtime_arn else None,
            memory=memory_metrics(agent_memory_arn) if agent_memory_arn else None,
            dealer_api=dealer_api_metrics(env_name),
            pipeline=pipeline_metrics(env_name),
        )

        for row in [*_agent_rows(metrics), *_service_rows(metrics)]:
            dashboard.add_widgets(*row)

        self._add_alarms(env_name, evaluation_topic, metrics)

        # Outputs
        cdk.CfnOutput(
            self,
            "DashboardURL",
            value=f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard.dashboard_name}",
            description="CloudWatch Dashboard URL",
        )

    def _add_alarms(self, env_name: str, topic: sns.ITopic, metrics: _Metrics) -> None:
        """Create every alarm and route it to the evaluation alert topic.

        Args:
            env_name: Environment suffix in each alarm name.
            topic: Topic each alarm publishes to.
            metrics: The metric groups for this environment.
        """
        for spec in [*_agent_alarm_specs(metrics), *_service_alarm_specs(metrics)]:
            alarm = cloudwatch.Alarm(
                self,
                spec.construct_id,
                alarm_name=f"agent-eval-{spec.name_suffix}-{env_name}",
                metric=spec.metric,
                threshold=spec.threshold,
                evaluation_periods=spec.evaluation_periods,
                comparison_operator=spec.comparison_operator,
                # Absent data means the workload was idle, not unhealthy: every
                # metric alarmed on here is a fault count or a latency
                # percentile over sparse traffic.
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description=spec.description,
            )
            alarm.add_alarm_action(cw_actions.SnsAction(topic))
