# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Monitoring Stack - Amazon CloudWatch Dashboards + Alarms.

Creates operational monitoring for agent evaluation:
- Dashboard with metrics for agent runtime, dealer API, and data pipeline
- AWS service metrics from AWS Lambda, Amazon API Gateway, Amazon DynamoDB, Amazon EventBridge, and Amazon Bedrock AgentCore
- Alarms with SNS notifications for critical thresholds

Cost considerations:
This stack creates billable Amazon CloudWatch resources: custom metrics
(about $0.30 per metric per month), dashboards (first 3 free, then about
$3 per dashboard per month), and alarms (first 10 free, then about $0.10
per alarm per month). Estimated default cost is roughly $5-15/month depending
on metric cardinality.

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

        # Amazon Bedrock AgentCore Runtime metrics (namespace "bedrock-agentcore"); see https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html
        if agent_runtime_arn:
            _ac_dims: dict[str, str] = {"ResourceArn": agent_runtime_arn}

            agent_invocations_metric = cloudwatch.Metric(
                namespace="bedrock-agentcore",
                metric_name="Invocations",
                dimensions_map=_ac_dims,
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )

            agent_system_errors_metric = cloudwatch.Metric(
                namespace="bedrock-agentcore",
                metric_name="SystemErrors",
                dimensions_map=_ac_dims,
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )

            agent_user_errors_metric = cloudwatch.Metric(
                namespace="bedrock-agentcore",
                metric_name="UserErrors",
                dimensions_map=_ac_dims,
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            )

            agent_latency_p50_metric = cloudwatch.Metric(
                namespace="bedrock-agentcore",
                metric_name="Latency",
                dimensions_map=_ac_dims,
                statistic="p50",
                period=cdk.Duration.minutes(5),
            )

            agent_latency_p95_metric = cloudwatch.Metric(
                namespace="bedrock-agentcore",
                metric_name="Latency",
                dimensions_map=_ac_dims,
                statistic="p95",
                period=cdk.Duration.minutes(5),
            )

            agent_latency_p99_metric = cloudwatch.Metric(
                namespace="bedrock-agentcore",
                metric_name="Latency",
                dimensions_map=_ac_dims,
                statistic="p99",
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

        # DynamoDB User Errors (Throttling)
        dynamodb_user_errors_metric = cloudwatch.Metric(
            namespace="AWS/DynamoDB",
            metric_name="UserErrors",
            dimensions_map={
                "TableName": f"agent-eval-dealers-{env_name}",
            },
            statistic="Sum",
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
                    title="Agent Runtime - Invocations vs System Errors",
                    left=[agent_invocations_metric],
                    right=[agent_system_errors_metric],
                    width=12,
                    left_y_axis=cloudwatch.YAxisProps(
                        min=0,
                        label="Invocations",
                    ),
                    right_y_axis=cloudwatch.YAxisProps(
                        min=0,
                        label="System Errors",
                    ),
                    legend_position=cloudwatch.LegendPosition.BOTTOM,
                ),
                cloudwatch.GraphWidget(
                    title="Agent Runtime - User Errors",
                    left=[agent_user_errors_metric],
                    width=12,
                    left_y_axis=cloudwatch.YAxisProps(
                        min=0,
                        label="User Errors",
                    ),
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
                right=[dynamodb_user_errors_metric],
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
            # Alarm: Agent runtime system errors > 5 per 5 minutes
            agent_error_rate_alarm = cloudwatch.Alarm(
                self,
                "AgentErrorRateAlarm",
                alarm_name=f"agent-eval-agent-error-rate-{env_name}",
                metric=agent_system_errors_metric,
                threshold=5,
                evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description=(
                    "Agent runtime SystemErrors exceeded threshold of 5 errors per 5 minutes"
                ),
            )
            agent_error_rate_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

            # Alarm: Agent runtime latency P99 > 5 seconds
            agent_latency_p99_alarm = cloudwatch.Alarm(
                self,
                "AgentLatencyP99Alarm",
                alarm_name=f"agent-eval-agent-latency-p99-{env_name}",
                metric=agent_latency_p99_metric,
                threshold=5000,  # 5 seconds in milliseconds
                evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
                alarm_description="Agent runtime P99 latency exceeded 5 seconds",
            )
            agent_latency_p99_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

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
            metric=dynamodb_user_errors_metric,
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

        # =====================================================================
        # ALARMS - EVALUATION QUALITY METRICS
        # =====================================================================
        # The evaluation pipeline publishes per-run quality scores to the
        # "AgentEvaluation/{env}" namespace (the evaluation role grants
        # PutMetricData scoped to exactly this namespace). Alert thresholds
        # mirror the ``alert_*`` values in evaluation/thresholds.py. Metrics
        # may be sparse between runs, so missing data is NOT_BREACHING and the
        # alarms evaluate against the latest reported datapoint.
        eval_namespace = f"AgentEvaluation/{env_name}"

        def _eval_metric(metric_name: str) -> cloudwatch.Metric:
            return cloudwatch.Metric(
                namespace=eval_namespace,
                metric_name=metric_name,
                statistic="Average",
                period=cdk.Duration.hours(1),
            )

        # Task completion rate fell below the acceptable floor.
        task_completion_alarm = cloudwatch.Alarm(
            self,
            "TaskCompletionAlarm",
            alarm_name=f"agent-eval-task-completion-{env_name}",
            metric=_eval_metric("TaskCompletionRate"),
            threshold=0.80,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Agent task-completion rate dropped below 0.80",
        )
        task_completion_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Tool-selection accuracy fell below the acceptable floor.
        tool_selection_alarm = cloudwatch.Alarm(
            self,
            "ToolSelectionAlarm",
            alarm_name=f"agent-eval-tool-selection-{env_name}",
            metric=_eval_metric("ToolSelectionAccuracy"),
            threshold=0.90,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Agent tool-selection accuracy dropped below 0.90",
        )
        tool_selection_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Helpfulness score fell below the acceptable floor.
        helpfulness_alarm = cloudwatch.Alarm(
            self,
            "HelpfulnessAlarm",
            alarm_name=f"agent-eval-helpfulness-{env_name}",
            metric=_eval_metric("HelpfulnessScore"),
            threshold=0.58,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Agent helpfulness score dropped below 0.58",
        )
        helpfulness_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Hallucination rate rose above the acceptable ceiling.
        hallucination_alarm = cloudwatch.Alarm(
            self,
            "HallucinationAlarm",
            alarm_name=f"agent-eval-hallucination-{env_name}",
            metric=_eval_metric("HallucinationRate"),
            threshold=0.05,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            alarm_description="Agent hallucination rate exceeded 0.05",
        )
        hallucination_alarm.add_alarm_action(cw_actions.SnsAction(evaluation_topic))

        # Outputs
        cdk.CfnOutput(
            self,
            "DashboardURL",
            value=f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard.dashboard_name}",
            description="CloudWatch Dashboard URL",
        )
