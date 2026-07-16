# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Dealer API Stack - DynamoDB + Lambda + Amazon API Gateway.

Exposes dealer data via REST API for AgentCore Gateway integration.

Cleanup:
    Important: Destroying this stack deletes resources and their data.
    If you need to preserve any data, create a backup first.
    These resources are billable: Amazon DynamoDB table, AWS Lambda
    function, Amazon API Gateway.
    Remove them with:
        cdk destroy DealerApiStack -c environment=<env>
    In non-dev environments the DynamoDB table uses RemovalPolicy.RETAIN
    and survives ``cdk destroy``. Delete it manually to stop charges.
"""

from typing import Any

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    Stack,
)
from aws_cdk import (
    aws_apigateway as apigw,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_wafv2 as wafv2,
)
from aws_cdk.aws_bedrockagentcore import (
    ApiGatewayHttpMethod,
    ApiGatewayToolConfiguration,
    ApiGatewayToolFilter,
    Gateway,
    GatewayAuthorizer,
    GatewayCredentialProvider,
    GatewayTarget,
)
from constructs import Construct


class DealerApiStack(Stack):
    """Stack for Dealer API with DynamoDB backend."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Get environment from context
        env_name = self.node.try_get_context("environment") or "dev"

        # DynamoDB Table for Dealers
        self.dealers_table = dynamodb.Table(
            self,
            "DealersTable",
            table_name=f"agent-eval-dealers-{env_name}",
            partition_key=dynamodb.Attribute(name="dealer_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=(
                cdk.RemovalPolicy.DESTROY if env_name == "dev" else cdk.RemovalPolicy.RETAIN
            ),
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True,
            ),
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=kms.Key(
                self,
                "DealersTableKey",
                alias=f"agent-eval/dealers-table-{env_name}",
                enable_key_rotation=True,
                removal_policy=(
                    cdk.RemovalPolicy.DESTROY if env_name == "dev" else cdk.RemovalPolicy.RETAIN
                ),
            ),
        )

        # Lambda Function for Dealer API
        dealer_api_function = lambda_.Function(
            self,
            "DealerApiFunction",
            function_name=f"agent-eval-dealer-api-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_14,
            code=lambda_.Code.from_asset("../lambda/functions/dealer_api"),
            handler="handler.lambda_handler",
            timeout=Duration.seconds(30),
            memory_size=512,
            environment={
                "DEALERS_TABLE": self.dealers_table.table_name,
                "ENVIRONMENT": env_name,
                "ALLOWED_ORIGIN": (
                    "http://localhost:3000"
                    if env_name == "dev"
                    else f"https://agent-eval.{env_name}.example.com"
                ),
            },
            log_group=logs.LogGroup(
                self,
                "DealerApiFunctionLogGroup",
                retention=logs.RetentionDays.ONE_WEEK,
                removal_policy=(
                    cdk.RemovalPolicy.DESTROY if env_name == "dev" else cdk.RemovalPolicy.RETAIN
                ),
            ),
        )

        # Grant DynamoDB permissions to Lambda
        self.dealers_table.grant_read_data(dealer_api_function)

        # Access Logs for API Gateway (Security best practice)
        api_log_group = logs.LogGroup(
            self,
            "ApiAccessLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=(
                cdk.RemovalPolicy.DESTROY if env_name == "dev" else cdk.RemovalPolicy.RETAIN
            ),
        )

        # Amazon API Gateway REST API with security best practices
        api = apigw.RestApi(
            self,
            "DealerApi",
            rest_api_name=f"agent-eval-dealer-api-{env_name}",
            description="Dealer API for AgentCore Gateway integration",
            deploy_options=apigw.StageOptions(
                stage_name=env_name,
                throttling_rate_limit=100,  # Rate limiting (security)
                throttling_burst_limit=200,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=False,  # Disabled: prevents logging sensitive request/response bodies
                metrics_enabled=True,  # CloudWatch metrics
                tracing_enabled=True,  # X-Ray distributed tracing
                access_log_destination=apigw.LogGroupLogDestination(api_log_group),
                access_log_format=apigw.AccessLogFormat.clf(),  # Common Log Format
            ),
            # Dev allows localhost; prod uses your real domain (replace the example.com placeholder).
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=(
                    ["http://localhost:3000"]
                    if env_name == "dev"
                    else [f"https://agent-eval.{env_name}.example.com"]
                ),
                allow_methods=["GET", "OPTIONS"],
                allow_headers=["Content-Type", "X-Api-Key", "Authorization"],
            ),
            endpoint_types=[apigw.EndpointType.REGIONAL],  # Regional endpoint
        )

        # Authentication is IAM (SigV4) on every method (see the GET methods
        # below). The AgentCore Gateway signs its outbound calls with the runtime
        # execution role's credentials, so no API key or shared secret is needed.
        # AWS does not support AgentCore Gateway targets whose methods require
        # BOTH AWS_IAM auth and an API key, so this API is IAM-only. Rate limiting
        # is still enforced by stage-level throttling (deploy_options above) and
        # the AWS WAF WebACL below.

        # Lambda integration (proxy mode for full event passthrough)
        dealer_integration = apigw.LambdaIntegration(
            dealer_api_function,
            proxy=True,  # Pass full API Gateway event to Lambda
        )

        # Declared method responses. AgentCore Gateway builds the MCP tool schema
        # from API Gateway's exported OpenAPI spec, which requires every GET
        # operation to declare a `responses` block — without this the Gateway
        # target fails to stabilize ("responses is missing").
        json_ok = apigw.MethodResponse(
            status_code="200",
            response_models={"application/json": apigw.Model.EMPTY_MODEL},
        )

        # /dealers resource
        dealers = api.root.add_resource("dealers")

        # GET /dealers - List all dealers (IAM SigV4 auth). operation_name sets
        # the OpenAPI operationId, which AgentCore Gateway requires to name the
        # MCP tool it derives from this operation.
        dealers.add_method(
            "GET",
            dealer_integration,
            authorization_type=apigw.AuthorizationType.IAM,
            operation_name="listDealers",
            method_responses=[json_ok],
        )

        # /dealers/{dealer_id} resource
        dealer = dealers.add_resource("{dealer_id}")

        # GET /dealers/{dealer_id} - Get specific dealer (IAM SigV4 auth).
        dealer.add_method(
            "GET",
            dealer_integration,
            authorization_type=apigw.AuthorizationType.IAM,
            operation_name="getDealerProfile",
            method_responses=[json_ok],
        )

        # AWS WAF WebACL — edge protection for the only internet-facing HTTP
        # surface in this project. Layers on top of the usage-plan throttling
        # (per-key quota) with volumetric + signature defenses:
        #   - rate-based rule: block IPs exceeding 2000 requests / 5 min
        #   - AWS managed common rule set (OWASP-style signatures)
        #   - AWS managed known-bad-inputs rule set
        web_acl = wafv2.CfnWebACL(
            self,
            "DealerApiWebAcl",
            name=f"agent-eval-dealer-api-waf-{env_name}",
            scope="REGIONAL",  # REGIONAL for API Gateway (CLOUDFRONT is for CF only)
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=f"agent-eval-dealer-api-waf-{env_name}",
                sampled_requests_enabled=True,
            ),
            rules=[
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimitPerIp",
                    priority=0,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            limit=2000,
                            aggregate_key_type="IP",
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimitPerIp",
                        sampled_requests_enabled=True,
                    ),
                ),
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSCommonRuleSet",
                    priority=1,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=(
                            wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                                vendor_name="AWS",
                                name="AWSManagedRulesCommonRuleSet",
                            )
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="AWSCommonRuleSet",
                        sampled_requests_enabled=True,
                    ),
                ),
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSKnownBadInputs",
                    priority=2,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=(
                            wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                                vendor_name="AWS",
                                name="AWSManagedRulesKnownBadInputsRuleSet",
                            )
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="AWSKnownBadInputs",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )
        wafv2.CfnWebACLAssociation(
            self,
            "DealerApiWebAclAssociation",
            resource_arn=api.deployment_stage.stage_arn,
            web_acl_arn=web_acl.attr_arn,
        )

        # ── Amazon Bedrock AgentCore Gateway ────────────────────────────────
        # The Gateway fronts the Dealer REST API as an MCP tool so the agent
        # reaches dealer profiles through AgentCore Gateway rather than calling
        # DynamoDB directly. Inbound auth to the Gateway is IAM (the runtime's
        # execution role signs its MCP calls); outbound auth to the Dealer API is
        # also IAM — the Gateway's service role signs SigV4 requests to the REST
        # API. No API key or shared secret is involved (AWS excludes methods that
        # require both AWS_IAM and an API key from AgentCore Gateway processing).
        gateway = Gateway(
            self,
            "DealerGateway",
            gateway_name=f"agent-eval-dealer-gw-{env_name}",
            authorizer_configuration=GatewayAuthorizer.using_aws_iam(),
        )

        # Expose the REST API's GET operations as MCP tools. The Gateway derives
        # the tool schema from API Gateway's exported OpenAPI 3.0 spec, which is
        # why every method declares method_responses above.
        GatewayTarget.for_api_gateway(
            self,
            "DealerApiTarget",
            gateway=gateway,
            rest_api=api,
            api_gateway_tool_configuration=ApiGatewayToolConfiguration(
                tool_filters=[
                    ApiGatewayToolFilter(
                        filter_path="/dealers/*",
                        methods=[ApiGatewayHttpMethod.GET],
                    )
                ],
            ),
            credential_provider_configurations=[
                GatewayCredentialProvider.from_iam_role(),
            ],
        )

        self.gateway = gateway
        self.gateway_url = gateway.gateway_url

        # Outputs
        cdk.CfnOutput(
            self,
            "DealerApiUrl",
            value=api.url,
            description="Dealer API Gateway URL",
        )

        cdk.CfnOutput(
            self,
            "DealerGatewayUrl",
            value=gateway.gateway_url or "",
            description="AgentCore Gateway MCP endpoint for the dealer-profile tool",
        )

        cdk.CfnOutput(
            self,
            "DealerApiWebAclArn",
            value=web_acl.attr_arn,
            description="WAF WebACL protecting the Dealer API",
        )

        cdk.CfnOutput(
            self,
            "DealersTableName",
            value=self.dealers_table.table_name,
            description="DynamoDB table for dealers",
        )

        # Export for use in other stacks
        self.api_url = api.url
        self.api = api
        self.dealers_table_name = self.dealers_table.table_name
