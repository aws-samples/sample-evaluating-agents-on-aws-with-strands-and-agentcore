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

## Security Controls

### Authentication and Authorization
- Dealer API Gateway endpoints require AWS IAM (SigV4) authorization; stage-level throttling and an AWS WAF WebACL provide rate limiting
- The AgentCore Gateway fronts the Dealer API with IAM authorization inbound (the runtime execution role) and outbound (the Gateway service role signs SigV4 to the REST API); no API key or shared secret is used
- Lambda functions use IAM execution roles with least-privilege grants
- AgentCore Runtime uses scoped IAM for Amazon Bedrock and Amazon S3, scoped read/write to its AgentCore Memory, and Gateway invoke; it reaches dealer profiles through the AgentCore Gateway rather than Amazon DynamoDB directly
- When you wire in real secrets, store them in AWS Secrets Manager or AWS Systems Manager Parameter Store (the sample mocks its data source and creates no such parameter). See `.env.example` for the recommended pattern.

### Encryption
- **At rest**: Amazon S3 (SSE-S3), Amazon DynamoDB (AWS-managed encryption)
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
- [ ] Set `METRICS_SOURCE=live` in your deployment environment (replaces the `.env.example` placeholder; wire it to your real metrics source)
- [ ] Confirm the Dealer API and AgentCore Gateway use IAM (SigV4) authorization end to end (no API keys to rotate in this design)
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
