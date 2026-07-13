# Agent Evaluation Pipeline Security

## Introduction

This document describes the security model for the Agent Evaluation Pipeline reference implementation. It covers how to report vulnerabilities, the security controls applied across the stack, and a checklist to complete before deploying to production. Review it before deployment and adapt the controls to align with your organization's requirements.

## Prerequisites

- An AWS account with permissions to review and modify security controls
- Understanding of the AWS Shared Responsibility Model
- Familiarity with AWS IAM roles, policies, and AWS KMS
- AWS CLI installed and configured
- Access to review AWS CloudTrail logs and the deployed infrastructure

## Reporting Vulnerabilities

If you discover a potential security issue, report it through
[AWS Vulnerability Reporting](https://aws.amazon.com/security/vulnerability-reporting/).
Do **not** create a public GitHub/GitLab issue.

## Shared Responsibility

This repository is a **reference implementation** (MIT-0 license) intended as sample content, not a turnkey solution. Treat the pre-deployment checklist in this document as a starting point, and adapt the security controls to your organization's requirements before any real-world use.

The AWS services used in this implementation operate under the [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/). These services include Amazon S3, Amazon DynamoDB, AWS Lambda, Amazon API Gateway, Amazon Bedrock, and Amazon Bedrock AgentCore.

## Threat Model

### Trust Boundaries

| Boundary | Description | Controls |
|----------|-------------|----------|
| User -> Agent | Dealer queries via AgentCore Runtime | API key auth, rate limiting, input validation |
| Agent -> Bedrock | LLM reasoning and embedding calls | IAM roles, inference profile scoping |
| Agent -> DynamoDB | Dealer profile lookups | IAM least privilege (`GetItem`, `Query` only) |
| Agent -> S3 | Vehicle data reads | IAM scoped to specific bucket, SSL enforced |
| Amazon EventBridge -> Lambda | Daily data ingestion trigger | IAM execution role, 2 retry attempts |
| Client -> API Gateway | Dealer API REST calls | API key + usage plan, CORS scoping, rate limiting |

### Attack Surface

| Vector | Mitigation |
|--------|------------|
| Pandas query injection (`df.query()`) | Multi-layer defense: NFKC normalization (Unicode compatibility normalization that converts characters to canonical forms to prevent injection via visually similar characters), regex allow list, method call detection, token blocklist (`_UNSAFE_TOKENS` and `_is_safe_query()` in `examples/vehicle-auction-agent/agent/app.py`) |
| DynamoDB injection | Parameterized `get_item(Key=...)` calls via boto3 SDK |
| Information disclosure | Generic error responses; no stack traces, internal paths, or user input reflected to clients |
| Log injection | Lazy `%s` formatting for user-supplied values in logger calls |
| Cross-dealer data leakage | Session-scoped dealer ID; `DealerDataScopingEvaluator` validates scope at evaluation time |
| Stale auction data | `DataFreshnessEvaluator` enforces a 24-hour refresh threshold |
| Unauthorized bidding | `SafetyGuardrailEvaluator` blocks forbidden tool calls and phrases |
| S3 public access | `BlockPublicAccess.BLOCK_ALL` enforces no public access; SSL enforcement and server-side encryption apply |
| CORS bypass | Production origins explicitly allowlisted; `*` restricted to dev |
| Brute-force API access | API Gateway: 100 rps / 200 burst / 10K daily quota |

### Out of Scope

- **Amazon Bedrock model security**: Amazon Bedrock manages model access controls and content filtering.
- **AgentCore Runtime isolation**: Amazon Bedrock AgentCore manages request-level compute isolation.
- **Network-level DDoS**: Handled by AWS Shield Standard (included with API Gateway).

## Security Controls

### Authentication and Authorization
- API Gateway endpoints require API keys with usage plans
- Lambda functions use IAM execution roles with least-privilege grants
- AgentCore Runtime uses scoped IAM for Bedrock, S3, DynamoDB, and SSM access
- The implementation uses AWS Systems Manager (SSM) parameters with ADVANCED tier (which provides AWS KMS encryption at rest and higher throughput limits)

### Encryption
- **At rest**: S3 (SSE-S3), DynamoDB (AWS-managed), SSM (KMS via ADVANCED tier)
- **In transit**: S3 SSL enforcement, API Gateway HTTPS, Bedrock API TLS 1.2+

### Input Validation
- `_is_safe_query()` validates all pandas query expressions before execution; allowlist expected operators and identifiers
- `search_vehicles()` uses typed parameters with structured filters (no raw SQL)
- `run_sql()` is a fallback for complex compound queries only; validate and sanitize expressions before passing to the engine
- To prevent prompt-injection attacks, sanitize natural-language dealer queries before forwarding to LLM calls
- DynamoDB keys are parameterized via boto3 SDK; validate key format and length before every lookup
- Validate S3 key paths against an allowlist of expected prefixes (`raw/`, `lancedb/`) before any `GetObject` or `PutObject` call
- Apply schema validation to all structured inputs (JSON payloads, API parameters) and reject inputs that fail type or format checks

### Monitoring
- Amazon CloudWatch alarms for error rates, latency P99, DynamoDB throttling
- Amazon SNS alerts for threshold breaches
- API Gateway access logs in Common Log Format
- Lambda function logs with configurable retention

## Dependency Management

Pin dependencies and audit regularly. Use uv (the uv Python package and project manager) to run the following security scanning tools:

```bash
uv run pip-audit                                                          # Check for known CVEs
uv run bandit -r src/agentic_evaluation/ examples/vehicle-auction-agent/ -ll  # Static security analysis
pre-commit run --all-files                                                # git-secrets + linting
```

### Known Advisories

| Package | CVE | Status | Impact |
|---------|-----|--------|--------|
| PyJWT | CVE-2026-32597 | Resolved: project pins 2.13.0 (fix landed in 2.12.0) | None: patched version in use; transitive dependency not used for JWT validation |

## Pre-Deployment Checklist

Before deploying to production:

- [ ] Replace `example.com` CORS origin with your actual domain
- [ ] Set `METRICS_SOURCE=live` in evaluation trigger Lambda
- [ ] Verify API key rotation process is documented in your runbook
- [ ] Run `uv run pip-audit` and resolve any HIGH/CRITICAL CVEs
- [ ] Subscribe to the SNS alert topic for evaluation notifications
- [ ] Review IAM roles and remove any unused permissions
- [ ] Set `ENVIRONMENT=prod` (enables stricter removal policies and CORS)

## Cleaning Up

When you no longer need the deployed resources, remove them to avoid ongoing charges and reduce your attack surface. The main [README Cleaning Up](../README.md#cleaning-up) section has detailed teardown commands. Those commands cover CDK stacks, Amazon S3 buckets, AWS Lambda functions, Amazon DynamoDB tables, and Amazon CloudWatch log groups.

Security-specific resources to remove manually if they were created outside the CDK stacks:

- **AWS Identity and Access Management (IAM) roles**: `aws iam list-roles --query 'Roles[?contains(RoleName,\`agent-eval\`)]'`
- **AWS Key Management Service (AWS KMS) keys**: Disable and schedule deletion via the AWS KMS console.
- **AWS CloudTrail trails**: `aws cloudtrail delete-trail --name <trail-name>`

## Conclusion

Security is a shared responsibility. This guide provides a secure deployment foundation. Adapt these controls to your organization's requirements using the pre-deployment checklist.
