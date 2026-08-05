# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Monitoring Stack - Amazon CloudWatch Dashboards + Alarms.

Creates operational monitoring for agent evaluation:
- Dashboard with metrics for agent runtime, dealer API, and data pipeline
- AWS service metrics from AWS Lambda, Amazon API Gateway, Amazon DynamoDB, Amazon EventBridge, and Amazon Bedrock AgentCore
- Alarms with SNS notifications for critical thresholds

Cost considerations:
This stack creates billable Amazon CloudWatch dashboards and alarms. It does
not create unpublished custom evaluation metrics. Estimated default cost is
roughly $1-5/month, depending on the account's free-tier usage.

Cleanup:
Important: Destroying this stack deletes alarm history and dashboard
configuration. If you need to preserve any dashboard JSON or alarm
configurations, create a backup first. To remove: ``cdk destroy MonitoringStack``
"""

from typing import Any

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
        super().__init__(scope, construct_id, **kwargs)

        # Get environment from context
        env_name = self.node.try_get_context("environment") or "dev"

        # CloudWatch Dashboard
        dashboard = cloudwatch.Dashboard(
            self,
            "EvaluationDashboard",
            dashboard_name=f"agent-eval-{env_name}",
        )

        # =====================================================================
        # AGENT RUNTIME METRICS (AgentCore Runtime)
        # =====================================================================

        # Amazon Bedrock AgentCore Runtime service metrics; see
        # https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html
        #
        # Namespace is "AWS/Bedrock-AgentCore". The lowercase "bedrock-agentcore"
        # namespace in the observability docs is for OTEL/EMF metrics emitted by
        # instrumented agent code, not for these service metrics, and carries no
        # data for them. Runtime service metrics are keyed on all three of
        # Resource + Operation + Name; dropping Name yields no datapoints.
        if agent_runtime_arn:
            _ac_dims: dict[str, str] = {
                "Resource": agent_runtime_arn,
                "Operation": "InvokeAgentRuntime",
                "Name": f"agent_eval_runtime_{env_name}::DEFAULT",
            }

            agent_invocations_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="Invocations",
                dimensions_map=_ac_dims,
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )

            agent_errors_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="Errors",
                dimensions_map=_ac_dims,
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )

            agent_latency_p50_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="Duration",
                dimensions_map=_ac_dims,
                statistic="p50",
                period=cdk.Duration.minutes(5),
            )

            agent_latency_p95_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="Duration",
                dimensions_map=_ac_dims,
                statistic="p95",
                period=cdk.Duration.minutes(5),
            )

            agent_latency_p99_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="Duration",
                dimensions_map=_ac_dims,
                statistic="p99",
                period=cdk.Duration.minutes(5),
            )

        if agent_memory_arn:
            _memory_dims = {
                "Resource": agent_memory_arn,
                "Operation": "Extraction",
            }
            # Extraction volume comes from CreationCount/MemoryRecordsExtracted, not
            # Invocations: AgentCore only emits Invocations for Extraction alongside
            # per-strategy StrategyId/StrategyType dimensions, so an Invocations
            # metric keyed on Resource+Operation alone reports no data at all.
            memory_extractions_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="CreationCount",
                dimensions_map={
                    "Resource": agent_memory_arn,
                    "ItemType": "MemoryRecordsExtracted",
                },
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )
            memory_extraction_errors_metric = cloudwatch.Metric(
                namespace="AWS/Bedrock-AgentCore",
                metric_name="Errors",
                dimensions_map=_memory_dims,
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )

        # =====================================================================
        # DEALER API METRICS (API Gateway + Lambda + DynamoDB)
        # =====================================================================

        # API Gateway Requests per Second
        api_requests_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="Count",
            dimensions_map={
                "ApiName": f"agent-eval-dealer-api-{env_name}",
                "Stage": env_name,
            },
            statistic="Sum",
            period=cdk.Duration.minutes(1),
        )

        # API Gateway 4xx Errors
        api_4xx_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="4XXError",
            dimensions_map={
                "ApiName": f"agent-eval-dealer-api-{env_name}",
                "Stage": env_name,
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )

        # API Gateway 5xx Errors
        api_5xx_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="5XXError",
            dimensions_map={
                "ApiName": f"agent-eval-dealer-api-{env_name}",
                "Stage": env_name,
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )

        # API Gateway Integration Latency
        api_integration_latency_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="IntegrationLatency",
            dimensions_map={
                "ApiName": f"agent-eval-dealer-api-{env_name}",
                "Stage": env_name,
            },
            statistic="Average",
            period=cdk.Duration.minutes(5),
        )

        # API Gateway Total Latency
        api_total_latency_metric = cloudwatch.Metric(
            namespace="AWS/ApiGateway",
            metric_name="Latency",
            dimensions_map={
                "ApiName": f"agent-eval-dealer-api-{env_name}",
                "Stage": env_name,
            },
            statistic="Average",
            period=cdk.Duration.minutes(5),
        )

        # DynamoDB Read Capacity
        dynamodb_read_capacity_metric = cloudwatch.Metric(
            namespace="AWS/DynamoDB",
            metric_name="ConsumedReadCapacityUnits",
            dimensions_map={
                "TableName": f"agent-eval-dealers-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )

        # DynamoDB throttling metrics. UserErrors includes validation and other
        # caller faults, so it must not be used as a throttle proxy.
        dynamodb_read_throttles_metric = cloudwatch.Metric(
            namespace="AWS/DynamoDB",
            metric_name="ReadThrottleEvents",
            dimensions_map={
                "TableName": f"agent-eval-dealers-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )
        dynamodb_write_throttles_metric = cloudwatch.Metric(
            namespace="AWS/DynamoDB",
            metric_name="WriteThrottleEvents",
            dimensions_map={
                "TableName": f"agent-eval-dealers-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )
        dynamodb_throttles_metric = cloudwatch.MathExpression(
            expression="read + write",
            using_metrics={
                "read": dynamodb_read_throttles_metric,
                "write": dynamodb_write_throttles_metric,
            },
            label="Throttle Events",
            period=cdk.Duration.minutes(5),
        )

        # =====================================================================
        # DATA PIPELINE METRICS (Lambda + EventBridge)
        # =====================================================================

        # Data Ingestion Lambda Invocations
        ingestion_invocations_metric = cloudwatch.Metric(
            namespace="AWS/Lambda",
            metric_name="Invocations",
            dimensions_map={
                "FunctionName": f"agent-eval-data-ingestion-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )

        # Data Ingestion Lambda Errors
        ingestion_errors_metric = cloudwatch.Metric(
            namespace="AWS/Lambda",
            metric_name="Errors",
            dimensions_map={
                "FunctionName": f"agent-eval-data-ingestion-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )

        # EventBridge Rule Invocations
        eventbridge_invocations_metric = cloudwatch.Metric(
            namespace="AWS/Events",
            metric_name="Invocations",
            dimensions_map={
                "RuleName": f"agent-eval-daily-refresh-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.hours(1),
        )

        # EventBridge Failed Invocations
        eventbridge_failed_metric = cloudwatch.Metric(
            namespace="AWS/Events",
            metric_name="FailedInvocations",
            dimensions_map={
                "RuleName": f"agent-eval-daily-refresh-{env_name}",
            },
            statistic="Sum",
            period=cdk.Duration.hours(1),
        )

        # =====================================================================
        # DASHBOARD LAYOUT
        # =====================================================================

        # ROW 1: AGENT RUNTIME METRICS - OVERVIEW
        if agent_runtime_arn:
            dashboard.add_widgets(
                cloudwatch.GraphWidget(
                    title="Agent Runtime - Invocations vs Errors",
                    left=[agent_invocations_metric],
                    right=[agent_errors_metric],
                    width=24,
                    left_y_axis=cloudwatch.YAxisProps(
                        min=0,
                        label="Invocations",
                    ),
                    right_y_axis=cloudwatch.YAxisProps(
                        min=0,
                        label="Errors",
                    ),
                    legend_position=cloudwatch.LegendPosition.BOTTOM,
                ),
            )

            # ROW 2: AGENT RUNTIME METRICS - LATENCY
            dashboard.add_widgets(
                cloudwatch.GraphWidget(
                    title="Agent Runtime - Latency Percentiles (ms)",
                    left=[
                        agent_latency_p50_metric,
                        agent_latency_p95_metric,
                        agent_latency_p99_metric,
                    ],
                    width=24,
                    left_y_axis=cloudwatch.YAxisProps(
                        min=0,
                        label="Latency (ms)",
                    ),
                    legend_position=cloudwatch.LegendPosition.BOTTOM,
                ),
            )
            if agent_memory_arn:
                dashboard.add_widgets(
                    cloudwatch.GraphWidget(
                        title="AgentCore Memory - Extractions vs Errors",
                        left=[memory_extractions_metric],
                        right=[memory_extraction_errors_metric],
                        width=24,
                        left_y_axis=cloudwatch.YAxisProps(
                            min=0,
                            label="Extractions",
                        ),
                        right_y_axis=cloudwatch.YAxisProps(
                            min=0,
                            label="Errors",
                        ),
                        legend_position=cloudwatch.LegendPosition.BOTTOM,
                    ),
                )
        else:
            dashboard.add_widgets(
                cloudwatch.TextWidget(
                    markdown=(
                        "## Agent Runtime Not Deployed\n\n"
                        "The AgentCore Runtime was not deployed in this environment. "
                        "Re-deploy with `agent_image_uri` context to enable agent runtime metrics."
                    ),
                    width=24,
                    height=3,
                ),
            )

        # ROW 3: DEALER API METRICS - REQUEST AND ERROR RATES
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Dealer API - Requests per Second",
                left=[api_requests_metric],
                width=12,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Requests/sec",
                ),
            ),
            cloudwatch.GraphWidget(
                title="Dealer API - HTTP Error Rates",
                left=[api_4xx_metric],
                right=[api_5xx_metric],
                width=12,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="4xx Errors",
                ),
                right_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="5xx Errors",
                ),
                legend_position=cloudwatch.LegendPosition.BOTTOM,
            ),
        )

        # ROW 4: DEALER API METRICS - LATENCY AND DYNAMODB
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Dealer API - Latency (ms)",
                left=[api_integration_latency_metric],
                right=[api_total_latency_metric],
                width=12,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Integration Latency (ms)",
                ),
                right_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Total Latency (ms)",
                ),
                legend_position=cloudwatch.LegendPosition.BOTTOM,
            ),
            cloudwatch.GraphWidget(
                title="DynamoDB - Read Capacity & Throttles",
                left=[dynamodb_read_capacity_metric],
                right=[dynamodb_throttles_metric],
                width=12,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Read Capacity Units",
                ),
                right_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Throttle Events",
                ),
                legend_position=cloudwatch.LegendPosition.BOTTOM,
            ),
        )

        # ROW 5: DATA PIPELINE METRICS
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Data Ingestion Lambda - Invocations & Errors",
                left=[ingestion_invocations_metric],
                right=[ingestion_errors_metric],
                width=12,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Invocations",
                ),
                right_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Errors",
                ),
                legend_position=cloudwatch.LegendPosition.BOTTOM,
            ),
            cloudwatch.GraphWidget(
                title="EventBridge - Daily Refresh Invocations",
                left=[eventbridge_invocations_metric],
                right=[eventbridge_failed_metric],
                width=12,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Invocations",
                ),
                right_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    label="Failed",
                ),
                legend_position=cloudwatch.LegendPosition.BOTTOM,
            ),
        )

        # =====================================================================
        # ALARMS - AGENT RUNTIME METRICS
        # =====================================================================

        if agent_runtime_arn:
            # Alarm: Agent runtime errors > 5 per 5 minutes
            agent_error_rate_alarm = cloudwatch.Alarm(
                self,
                "AgentErrorRateAlarm",
                alarm_name=f"agent-eval-agent-error-rate-{env_name}",
                metric=agent_errors_metric,
                threshold=5,
                evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description="Agent runtime errors exceeded threshold of 5 per 5 minutes",
            )
            agent_error_rate_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

            # Keep the operational alarm aligned with alert_latency_p99_ms in
            # eval_config.yaml.
            agent_latency_p99_alarm = cloudwatch.Alarm(
                self,
                "AgentLatencyP99Alarm",
                alarm_name=f"agent-eval-agent-latency-p99-{env_name}",
                metric=agent_latency_p99_metric,
                threshold=30000,
                evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description="Agent runtime P99 latency exceeded 30 seconds",
            )
            agent_latency_p99_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        if agent_memory_arn:
            memory_extraction_error_alarm = cloudwatch.Alarm(
                self,
                "MemoryExtractionErrorAlarm",
                alarm_name=f"agent-eval-memory-extraction-error-{env_name}",
                metric=memory_extraction_errors_metric,
                threshold=0,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description="AgentCore Memory extraction encountered errors",
            )
            memory_extraction_error_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # =====================================================================
        # ALARMS - DEALER API METRICS
        # =====================================================================

        # Alarm: Dealer API error rate > 2%
        api_error_rate_alarm = cloudwatch.Alarm(
            self,
            "ApiErrorRateAlarm",
            alarm_name=f"agent-eval-api-error-rate-{env_name}",
            metric=cloudwatch.Metric(
                namespace="AWS/ApiGateway",
                metric_name="5XXError",
                dimensions_map={
                    "ApiName": f"agent-eval-dealer-api-{env_name}",
                    "Stage": env_name,
                },
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            ),
            threshold=2,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Dealer API 5xx error count exceeded threshold of 2 errors per 5 minutes",
        )
        api_error_rate_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Alarm: DynamoDB throttling events
        dynamodb_throttle_alarm = cloudwatch.Alarm(
            self,
            "DynamoDBThrottleAlarm",
            alarm_name=f"agent-eval-dynamodb-throttle-{env_name}",
            metric=dynamodb_throttles_metric,
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="DynamoDB throttling events detected",
        )
        dynamodb_throttle_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # =====================================================================
        # ALARMS - DATA PIPELINE METRICS
        # =====================================================================

        # Alarm: Data ingestion Lambda errors > 0
        ingestion_error_alarm = cloudwatch.Alarm(
            self,
            "IngestionErrorAlarm",
            alarm_name=f"agent-eval-ingestion-error-{env_name}",
            metric=ingestion_errors_metric,
            threshold=0,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Data ingestion Lambda encountered errors",
        )
        ingestion_error_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Alarm: EventBridge failed invocations
        eventbridge_failed_alarm = cloudwatch.Alarm(
            self,
            "EventBridgeFailedAlarm",
            alarm_name=f"agent-eval-eventbridge-failed-{env_name}",
            metric=eventbridge_failed_metric,
            threshold=0,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="EventBridge rule failed to invoke data ingestion",
        )
        eventbridge_failed_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Outputs
        cdk.CfnOutput(
            self,
            "DashboardURL",
            value=f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard.dashboard_name}",
            description="CloudWatch Dashboard URL",
        )
