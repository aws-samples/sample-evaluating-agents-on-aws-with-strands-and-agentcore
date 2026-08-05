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
- Lambda Function URLs are prohibited. An app-wide CDK aspect fails synthesis
  if any stack creates one or grants public/account-principal invocation.
- Cross-service Lambda invocation permissions require a specific AWS service
  principal and scoped source ARN. The dealer Lambda accepts only this API
  Gateway; the private ingestion Lambda accepts its named EventBridge rule.
- The AgentCore Gateway fronts the Dealer API with IAM authorization inbound (the runtime execution role) and outbound (the Gateway service role signs SigV4 to the REST API); no API key or shared secret is used
- IAM Runtime deployments are single-tenant by default. Cognito deployments derive dealer identity from a verified JWT claim. The runtime exposes only a zero-argument dealer-profile wrapper and injects the trusted dealer ID server-side; the model cannot list dealers or select a different ID.
- Lambda functions use IAM execution roles with least-privilege grants
- AgentCore Runtime uses scoped IAM for Amazon Bedrock and Amazon S3, scoped read/write to its AgentCore Memory, and Gateway invoke; it reaches dealer profiles through the AgentCore Gateway rather than Amazon DynamoDB directly
- When you wire in real secrets, store them in AWS Secrets Manager or AWS Systems Manager Parameter Store (the sample mocks its data source and creates no such parameter). See `.env.example` for the recommended pattern.

### Encryption
- **At rest**: Amazon S3 uses SSE-S3. Customer-managed, rotating KMS keys
  protect DynamoDB, Lambda environment variables, application-managed CloudWatch
  logs, SNS alerts, and the Secrets Manager evaluation token. AgentCore's
  service-created runtime logs use CloudWatch Logs encryption at rest and a
  seven-day retention policy. The ingestion DLQ uses the AWS-managed SQS KMS key.
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
# Root SDK project (all extras).
uv export --frozen --no-dev --all-extras --no-emit-project \
  --format requirements-txt |
  uvx --from pip-audit pip-audit -r /dev/stdin --no-deps --disable-pip
# CDK app and both Lambda functions, which also deploy to AWS.
for project in examples/vehicle-auction-agent/cdk \
  examples/vehicle-auction-agent/lambda/functions/data_ingestion \
  examples/vehicle-auction-agent/lambda/functions/dealer_api; do
  uv export --frozen --no-dev --no-emit-project \
    --format requirements-txt --project "$project" |
    uvx --from pip-audit pip-audit -r /dev/stdin --no-deps --disable-pip
done
# Hash-pinned agent runtime image.
uvx --from pip-audit pip-audit --no-deps --disable-pip \
  -r examples/vehicle-auction-agent/agent/requirements.lock
uvx --from bandit bandit -r src scripts examples/vehicle-auction-agent/agent \
  examples/vehicle-auction-agent/cdk/lib -ll
uv run ruff check .
```

Regenerate the agent runtime lock after changing `requirements.txt` or `constraints.txt`:

```bash
cd examples/vehicle-auction-agent/agent
uv pip compile requirements.txt --constraint constraints.txt \
  --python-platform aarch64-manylinux_2_28 --python-version 3.14 \
  --generate-hashes --only-binary :all: --output-file requirements.lock
```

### Known Advisories

| Package | CVE | Status | Impact |
|---------|-----|--------|--------|
| PyJWT | CVE-2026-32597 | Resolved: project pins 2.13.0 (fix landed in 2.12.0) | None: patched version in use; transitive dependency not used for JWT validation |
| aiohttp | CVE-2026-69244, CVE-2026-69243, CVE-2026-59881 | Resolved: constrained to >=3.14.3 | None: patched version in use. Reached only transitively via `strands-agents-evals` -> `strands-agents-tools`; no first-party code imports aiohttp, and it is absent from the deployed agent image |
| bedrock-agentcore | CVE-2026-16796 | Resolved: floor raised to >=1.18.1 | None: patched version in use. The affected Code Interpreter `install_packages()` path is not used by this project |
| cryptography | CVE-2026-69247 | Resolved: constrained to >=50.0.0 | None: patched version in use. Reached transitively via authlib, joserfc, `pyjwt[crypto]` and secretstorage; no PKCS#7 `EnvelopedData` decryption anywhere in the project |
| Pillow | 13 advisories, including CVE-2026-59197, CVE-2026-59199, CVE-2026-59205 | Resolved: constrained to >=12.3.0 | None: patched version in use. Reached only transitively via `strands-agents-evals` -> `strands-agents-tools`; no first-party code imports Pillow, and it is absent from the deployed agent image |
| mcp | CVE-2026-59950 | Resolved: floor raised to >=1.28.1 | None: patched version in use. The affected WebSocket server transport is unused: the agent consumes MCP only as a client, over SigV4-signed streamable HTTP to the AgentCore Gateway |

Transitive floors live in `[tool.uv] constraint-dependencies` (root `pyproject.toml`) and
`examples/vehicle-auction-agent/agent/constraints.txt`, so they survive re-resolution.

## Pre-Deployment Checklist

Before deploying to production:

- [ ] Replace `example.com` CORS origin with your actual domain
- [ ] Confirm the Dealer API and AgentCore Gateway use IAM (SigV4) authorization end to end (no API keys to rotate in this design)
- [ ] Select an explicit AWS profile, `eu-west-1`, and expected 12-digit account; verify them with STS
- [ ] Run the frozen-lock dependency audit above and resolve any known advisories
- [ ] Review `cdk diff`, the immutable ECR digest, and the approximately $55-105/month dev estimate before approving deployment
- [ ] Subscribe to the SNS alert topic for evaluation notifications
- [ ] Review IAM roles and remove any unused permissions
- [ ] Set `ENVIRONMENT=prod` (enables stricter removal policies and CORS)

## Cleaning Up

Before teardown, create a retention and backup manifest covering every bucket
version, table, log group, secret, and ECR image digest. Verify the explicit
profile/account/region again and obtain separate approval for stack deletion
and retained-data deletion. CDK retains the S3 data and access-log buckets; a
deleted stack is not evidence that billable data was removed.

## Conclusion

Security is a shared responsibility. This guide provides a secure deployment foundation. Adapt these controls to your organization's requirements using the pre-deployment checklist.
