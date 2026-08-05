# Recommended Security Practices

## Introduction

This document covers the security controls, IAM policies, data-protection measures, and operational monitoring for the Agent Evaluation Pipeline. It serves as a pre-deployment reference and a guide for adapting security controls to your organization's requirements.

## Overview

This document outlines the security measures implemented in the Agent Evaluation Pipeline and provides guidance for maintaining a secure deployment. The pipeline runs on AWS Lambda, Amazon S3, Amazon DynamoDB, and Amazon Bedrock AgentCore. Its security controls focus on least-privilege IAM and encryption in transit and at rest. These controls also address secrets isolation and pipeline-integrated vulnerability scanning. The scope covers infrastructure-as-code configuration, runtime IAM policies, data-protection controls, and operational monitoring. Use this guide as a pre-deployment reference and adapt each control to your organization's security requirements.

## Table of Contents

- [Prerequisites](#prerequisites)
- [IAM Policies and Permissions](#iam-policies-and-permissions)
- [Data Encryption](#data-encryption)
- [Network Security](#network-security)
- [Secrets Management](#secrets-management)
- [Audit and Compliance](#audit-and-compliance)
- [Security Scanning](#security-scanning)
- [Incident Response](#incident-response)

---

## Prerequisites

- An AWS account with permissions to review and modify IAM roles and security controls
- Familiarity with the AWS Shared Responsibility Model
- Working knowledge of AWS IAM roles, policies, and AWS Key Management Service (AWS KMS)
- AWS CLI installed and configured with appropriate credentials
- Access to AWS CloudTrail logs and the deployed infrastructure stack

---

## IAM Policies and Permissions

### Principle of Least Privilege

All IAM roles follow the principle of least privilege, granting only the minimum permissions required for operation.

### Data Ingestion Lambda Role

**Role Name**: `DataIngestionRole`

This role grants the data-ingestion AWS Lambda function access to Amazon Bedrock for model invocation, along with Amazon S3 and Amazon CloudWatch Logs permissions.

**Permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::agent-eval-data-{env}-{account}-{region}/raw/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::agent-eval-data-{env}-{account}-{region}/lancedb/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:{region}:{account}:log-group:/aws/lambda/agent-eval-data-ingestion-{env}:*"
    }
  ]
}
```

The ingestion path addresses known object keys directly, so it does not need
`s3:ListBucket`, object deletion, or wildcard action names.

### Agent Runtime Role

**Role Name**: `agent-eval-runtime-role-{env}`

**Permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::agent-eval-data-{env}-{account}-{region}/lancedb/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:{region}:{account}:inference-profile/eu.anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-*::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-*::foundation-model/amazon.titan-embed-text-v2:0"
      ]
    }
  ]
}
```

The chat model is invoked through a cross-region inference profile (the `eu.`
prefix), which fans out to the underlying foundation model in whichever EU
region it routes to. The foundation-model resources are therefore scoped to the
`eu-*` partition rather than a single region; pinning to one region would deny
requests the profile routes elsewhere in the EU, while `eu-*` still excludes
every other partition (us/ap/etc.).

### Online Evaluation Role

**Role Name**: `agent-eval-agentcore-{env}`

**Permissions**:
- Amazon CloudWatch Logs: Read evaluation results
- Amazon S3 Read: Access evaluation data
- Amazon SNS Publish: Send alerts on threshold breaches

Custom online-evaluation metrics and their alarms are adopter-owned and are not
provisioned by this repository.

### Build-Time Evaluation Role

**Role Name**: `agent-eval-buildtime-{env}`

**Permissions**:
- Amazon CloudWatch Logs: Write evaluation results
- Amazon S3 Read: Access evaluation data
- Amazon Bedrock: `InvokeModel` for LLM-as-judge evaluators

### AWS Service Limitations (Resource: "*")

The following IAM actions are **non-resource-level**, so AWS does not support scoping them below `Resource: "*"`. This is an AWS service constraint, not a security oversight. Condition keys or scoped companion statements mitigate each one:

| Action | Reason | Mitigation |
|--------|--------|------------|
| `ecr:GetAuthorizationToken` | Returns a token for any ECR registry; cannot target a specific repo | Companion statements scope push/pull to specific repositories |
| `xray:PutTraceSegments` | Trace ingestion is account-wide | No sensitive data exposed; traces are write-only |
| `xray:PutTelemetryRecords` | Telemetry ingestion is account-wide | Same as above |
| `cloudwatch:PutMetricData` | Non-resource-level action | Scoped by `cloudwatch:namespace` condition key |

References:
- [Amazon ECR service authorization](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonelasticcontainerregistry.html)
- [AWS X-Ray service authorization](https://docs.aws.amazon.com/service-authorization/latest/reference/list_awsx-ray.html)
- [Amazon CloudWatch service authorization](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazoncloudwatch.html)

---

## Data Encryption

### S3 Bucket Encryption

**Encryption Method**: S3 Server-Side Encryption (SSE-S3 / AES-256)

**Encryption at Rest**:
- All S3 buckets use S3-managed encryption (SSE-S3)
- Customer-managed, rotating KMS keys protect DynamoDB, Lambda environment
  variables, CloudWatch logs, SNS alerts, and the Secrets Manager evaluation
  token
- The ingestion dead-letter queue uses the AWS-managed SQS KMS key

**Encryption in Transit**:
- All S3 operations enforce SSL/TLS (deny HTTP requests)
- Bucket policy enforces `aws:SecureTransport: true`

> **Note:** The action list in this deny policy is illustrative. Scope it to only the actions your role legitimately needs (for example, `["s3:GetObject", "s3:PutObject"]`) rather than listing every S3 action. Applying least privilege to deny policies prevents accidental over-restriction of future roles.

```json
{
  "Effect": "Deny",
  "Principal": "*",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:ListBucket",
    "s3:GetBucketLocation",
    "s3:ListBucketMultipartUploads",
    "s3:AbortMultipartUpload",
    "s3:ListMultipartUploadParts"
  ],
  "Resource": [
    "arn:aws:s3:::agent-eval-data-{env}-{account}-{region}",
    "arn:aws:s3:::agent-eval-data-{env}-{account}-{region}/*"
  ],
  "Condition": {
    "Bool": {
      "aws:SecureTransport": "false"
    }
  }
}
```

### Lambda Environment Variables

**Sensitive Data**: Never store secrets in environment variables.

**Use AWS Secrets Manager or Parameter Store for**:
- API keys
- Database credentials
- Third-party service tokens

---

## Network Security

### Lambda Invocation Boundary

Lambda Function URLs are prohibited, including URLs configured for IAM
authorization. The CDK application applies a global synthesis guard that also
rejects public or account-principal Lambda invocation permissions. AWS
service-to-service invocation remains allowed only with a specific service
principal and scoped source ARN. The dealer Lambda is invoked through the
IAM-authorized API Gateway; the ingestion Lambda is invoked by its named
EventBridge rule.

### VPC Configuration

**Current Setup**: Lambda functions are not attached to a customer VPC. They
have outbound service access but no directly routable inbound endpoint. API
Gateway is the authenticated, WAF-protected ingress to the dealer Lambda.
AgentCore Runtime uses its managed public network mode and IAM or Cognito
authorization; it is not an anonymous endpoint.

**Recommended for Production**:
1. Deploy Lambda functions in private subnets
2. Use VPC endpoints for Amazon S3, Amazon Bedrock, and other AWS services
3. Configure Security Groups with minimum required ingress/egress

### S3 Access Control

**Public Access**: Blocked at the bucket level
- `BlockPublicAcls: true`
- `BlockPublicPolicy: true`
- `IgnorePublicAcls: true`
- `RestrictPublicBuckets: true`

---

## Secrets Management

### AWS Secrets Manager Integration

For production deployments, integrate AWS Secrets Manager for sensitive configuration:

```python
import boto3
import json

def get_secret(secret_name, region):
    client = boto3.client('secretsmanager', region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage for a real third-party data source
data_source_credentials = get_secret('data-source-credentials', 'eu-west-1')
```

### Secret Rotation

**Recommended**:
- Enable automatic rotation for all secrets
- Rotate secrets every 90 days
- Use AWS Lambda for custom rotation logic

---

## Audit and Compliance

### CloudTrail Logging

CloudTrail is an account-level prerequisite, not provisioned by these stacks.
Before production, verify that organization/account trails record S3 data
events where required, Lambda and Bedrock control-plane activity, IAM role
assumptions, and CloudFormation operations.

### CloudWatch Logs Retention

**Log Groups**:
- `/aws/lambda/agent-eval-data-ingestion-{env}`: 7 days (dev), 30 days (prod)
- `/aws/lambda/agent-eval-runtime-{env}`: 7 days (dev), 30 days (prod)
- `/aws/agent-evaluation/{env}`: 7 days (dev), 30 days (prod)

### Compliance Standards

**Payment Card Industry Data Security Standard (PCI DSS)**: If handling payment data, verify that the following controls are in place:
- Encryption at rest and in transit
- Access logging and monitoring
- Regular vulnerability scans

**General Data Protection Regulation (GDPR)**: If handling EU user data:
- Data minimization
- Right to erasure (S3 lifecycle policies)
- Audit trails

---

## Security Scanning

### Pre-Deployment Scans

The verified local checks are:

1. **Ruff**: Python lint and format checks.
2. **Bandit**: Python static security analysis.
3. **pip-audit**: Frozen `uv.lock` and deployable lock vulnerability checks.
4. **cfn-lint**: Synthesized CloudFormation schema/IAM validation.
5. **Draw.io XML validation**: Diagram source integrity and connector policy.

### Regular Vulnerability Scans

**Recommended**:
- Run `pip-audit` weekly on production dependencies
- Enable Amazon Inspector for EC2/Lambda runtime security
- Use Snyk or similar tools for container image scanning

---

## Incident Response

### Security Incident Workflow

1. **Detection**: CloudWatch Alarms, GuardDuty, or manual report
2. **Containment**: Revoke IAM credentials, isolate affected resources
3. **Investigation**: Review CloudTrail logs, Lambda logs
4. **Remediation**: Apply patches, rotate secrets, update IAM policies
5. **Post-Mortem**: Document lessons learned, update runbooks

### Emergency Contacts

- **Security Team**: security@example.com
- **AWS Support**: 1-877-742-2275 (US)

### Incident Response Plan

Follow your organization's incident-response runbook. For vulnerabilities in
this sample, see the reporting process in the top-level `SECURITY.md`.

---

## Security Checklist

- [ ] Enable MFA for all AWS accounts
- [ ] Use IAM roles instead of access keys for Lambda
- [ ] Enable CloudTrail in all regions
- [ ] Encrypt all S3 buckets with SSE-S3 or KMS
- [ ] Enable S3 versioning for data recovery
- [ ] Use AWS Secrets Manager for sensitive data
- [ ] Regularly rotate IAM credentials and secrets
- [ ] Enable Amazon GuardDuty for threat detection
- [ ] Review IAM policies quarterly
- [ ] Run security scans in CI/CD pipeline
- [ ] Enable AWS Config for compliance monitoring
- [ ] Set up CloudWatch Alarms for anomalous activity
- [ ] Document security procedures in runbooks

---

## Cleaning Up Security Resources

Before decommissioning, create a retention and backup manifest for S3 object
versions, DynamoDB data, logs, Secrets Manager values, and ECR image digests.
Reverify the explicit profile, expected account, and `eu-west-1`; review a
destroy change set and obtain approval. S3 data and access-log buckets are
retained by design. After stack deletion, independently inventory
CloudFormation and every billable service.

---

## Conclusion

Security is a shared responsibility. This guide provides the foundation for a secure deployment, but you must adapt these controls to your organization's requirements. Complete the pre-deployment checklist before going to production, enable monitoring and alerting, and review your security posture regularly.

---

## References

- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [AWS Well-Architected Framework - Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html)
- [OWASP Top 10](https://owasp.org/Top10)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
