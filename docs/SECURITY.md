# Recommended Security Practices

## Introduction

This document covers the security controls, IAM policies, data-protection measures, and operational monitoring for the Agent Evaluation Pipeline. It serves as a pre-deployment reference and a guide for adapting security controls to your organization's requirements.

## Overview

This document outlines the security measures implemented in the Agent Evaluation Pipeline and provides guidance for maintaining a secure deployment. The pipeline runs on AWS Lambda, Amazon S3, Amazon DynamoDB, and Amazon Bedrock AgentCore. Its threat model focuses on least-privilege IAM and encryption in transit and at rest. The model also addresses secrets isolation and pipeline-integrated vulnerability scanning. The scope covers infrastructure-as-code configuration, runtime IAM policies, data-protection controls, and operational monitoring. Use this guide as a pre-deployment reference and adapt each control to your organization's security requirements.

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
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::amzn-s3-demo-bucket-agent-eval",
        "arn:aws:s3:::amzn-s3-demo-bucket-agent-eval/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0",
        "arn:aws:bedrock:{region}::foundation-model/amazon.nova-lite-v1:0"
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

> **Tip:** Scope `s3:ListBucket` to specific prefixes (such as `raw/*`, `lancedb/*`) with an `s3:prefix` condition key for least privilege.

### Agent Runtime Role

**Role Name**: `agent-eval-runtime-{env}`

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
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:Converse",
        "bedrock:ConverseStream"
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

### Evaluation Role

**Role Name**: `agent-eval-agentcore-{env}`

**Permissions**:
- CloudWatch Metrics: Put custom metrics
- CloudWatch Logs: Write evaluation results
- S3 Read: Access evaluation data
- Amazon SNS Publish: Send alerts on threshold breaches

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
- Optional: Upgrade to KMS encryption for dev/staging/prod

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

### VPC Configuration

**Current Setup**: Lambda functions run in default VPC (internet-accessible).

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

# Usage
dealer_api_key = get_secret('dealer-api-key', 'eu-west-1')
```

### Secret Rotation

**Recommended**:
- Enable automatic rotation for all secrets
- Rotate secrets every 90 days
- Use AWS Lambda for custom rotation logic

---

## Audit and Compliance

### CloudTrail Logging

**Enabled for**:
- All API calls to S3, Lambda, Amazon Bedrock
- IAM role assumptions
- AWS CloudFormation stack operations

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

**GitLab CI/CD Pipeline**:

1. **Bandit**: Static analysis for Python security issues
2. **pip-audit**: Check for known vulnerabilities in dependencies
3. **safety**: Additional vulnerability scanning
4. **detect-secrets**: Scan for accidentally committed secrets
5. **cfn-nag**: CloudFormation template security analysis

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

When decommissioning a deployment, remove security-related resources to stop ongoing charges from CloudWatch log storage and Amazon SNS topics. IAM roles themselves are free, but the associated logging and monitoring resources incur storage costs while they remain.

> **Important:** Deleting an IAM role disrupts any active resource that still depends on it. Before removing a role, confirm that the dependent services (AWS Lambda functions, AWS CodeBuild projects, and AgentCore runtimes) are already deleted. If a deletion fails with a conflict error, list and detach the role's policies first, then remove the dependent resources before retrying:
>
> ```bash
> aws iam list-attached-role-policies --role-name agent-eval-agentcore-<env>
> aws iam list-role-policies --role-name agent-eval-agentcore-<env>
> ```

1. Delete the AgentCore execution IAM role:
   ```bash
   aws iam delete-role --role-name agent-eval-agentcore-<env>
   ```
2. Delete the CodeBuild service IAM role:
   ```bash
   aws iam delete-role --role-name agent-eval-codebuild-<env>
   ```
3. Delete Amazon CloudWatch log groups:
   ```bash
   aws logs delete-log-group --log-group-name /aws/agent-evaluation/<env>
   ```
4. Delete Amazon SNS alert topics:
   ```bash
   aws sns delete-topic --topic-arn arn:aws:sns:<region>:<account>:agent-eval-alerts-<env>
   ```
5. Verify removal with the main README cleanup section for full infrastructure teardown.

---

## Conclusion

Security is a shared responsibility. This guide provides the foundation for a secure deployment, but you must adapt these controls to your organization's requirements. Complete the pre-deployment checklist before going to production, enable monitoring and alerting, and review your security posture regularly.

---

## References

- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [AWS Well-Architected Framework - Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html)
- [OWASP Top 10](https://owasp.org/Top10)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
