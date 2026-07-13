# Troubleshooting the Agent Evaluation Pipeline

## Introduction

This guide helps you diagnose and resolve common issues when deploying and operating the Agent Evaluation Pipeline. It covers deployment failures, IAM and permissions errors, Amazon Bedrock model access, and runtime debugging. Work through the sections in order, or jump to the section matching your symptom. If an issue persists after following these steps, escalate to AWS Support with the relevant logs and stack events.

## Prerequisites

- AWS Command Line Interface (AWS CLI) installed and configured
- AWS Identity and Access Management (IAM) permissions to view AWS Lambda functions, Amazon CloudWatch Logs, and AWS CloudFormation stacks
- Access to the deployed infrastructure
- Familiarity with AWS Cloud Development Kit (AWS CDK) deployment commands

## Common Issues and Solutions

This guide groups common problems by area, with diagnosis steps and fixes.

## Table of Contents

- [CDK Deployment Issues](#cdk-deployment-issues)
- [Lambda Function Errors](#lambda-function-errors)
- [Data Ingestion Problems](#data-ingestion-problems)
- [CloudWatch and Monitoring](#cloudwatch-and-monitoring)
- [Docker and AWS CodeBuild Issues](#docker-and-aws-codebuild-issues)
- [Performance Problems](#performance-problems)
- [Debugging Tips](#debugging-tips)

---

## CDK Deployment Issues

### Issue: "You must bootstrap your environment before you can deploy"

**Error**:
```
The stack 'agent-eval-dev-data-pipeline' requires bootstrap stack version '21', but this environment has version '0'
```

**Solution**:
```bash
# To prepare your environment for deployment, bootstrap AWS:
AWS_REGION=eu-west-1 cdk bootstrap aws://<ACCOUNT_ID>/eu-west-1

# Then redeploy
AWS_REGION=eu-west-1 uv run cdk deploy --all
```

---

### Issue: "Resource already exists" during deployment

**Error**:
```
CREATE_FAILED: /aws/lambda/agent-eval-data-ingestion-dev already exists in stack
```

**Solution**:
CloudFormation fails here because a resource with the same name already exists outside the stack.

**Option 1**: Import existing resource into CDK

1. Delete the manually created resource:
   ```bash
   aws logs delete-log-group --log-group-name /aws/lambda/agent-eval-data-ingestion-dev --region eu-west-1
   ```

2. Redeploy the stack:
   ```bash
   AWS_REGION=eu-west-1 uv run cdk deploy --all
   ```

**Option 2**: Use `cdk import` to import existing resources

---

### Issue: "Insufficient permissions to deploy"

**Error**:
```
User is not authorized to perform: iam:CreateRole
```

**Solution**:
Verify that your AWS credentials have sufficient permissions:

> **Warning:** The following is an illustrative minimal-permission example. Never use wildcard actions or `Resource: "*"` in production. Scope actions and resource ARNs to exactly what your deployment needs. Constrain `iam:PassRole` with the `iam:PassedToService` condition key.

> **Note:** `cloudwatch:PutMetricData` requires `Resource: "*"` because it is a non-resource-level action. The `cloudwatch:namespace` condition provides the actual scope. See: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazoncloudwatch.html

> **Warning:** This policy is for diagnostics only. Scope actions and resources to least privilege in production.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:DescribeStacks",
        "cloudformation:CreateStack",
        "cloudformation:UpdateStack",
        "lambda:GetFunction",
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode",
        "s3:GetObject",
        "s3:PutObject",
        "logs:CreateLogGroup",
        "logs:PutLogEvents",
        "events:PutRule",
        "sns:Publish"
      ],
      "Resource": [
        "arn:aws:cloudformation:eu-west-1:ACCOUNT_ID:stack/agent-eval-*/*",
        "arn:aws:lambda:eu-west-1:ACCOUNT_ID:function:agent-eval-*",
        "arn:aws:s3:::agent-eval-*/*",
        "arn:aws:logs:eu-west-1:ACCOUNT_ID:log-group:/aws/lambda/agent-eval-*:*"
      ]
    },
    {
      "Sid": "CloudWatchMetrics",
      "Effect": "Allow",
      "Action": "cloudwatch:PutMetricData",
      "Resource": "*",
      "Condition": {
        "StringEquals": { "cloudwatch:namespace": "AgentEvaluation" }
      }
    },
    {
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::ACCOUNT_ID:role/agent-eval-*",
      "Condition": {
        "StringEquals": { "iam:PassedToService": "lambda.amazonaws.com" }
      }
    }
  ]
}
```

---

## Lambda Function Errors

### Issue: Lambda function times out

**Error**:
```
Task timed out after 600.00 seconds
```

**Causes**:
1. Large data processing (too many vehicles causes the function to exceed its time limit)
2. Slow Amazon Bedrock API responses
3. Amazon S3 read/write operations taking longer than expected

**Solutions**:

1. **Increase timeout**:
```python
# In examples/vehicle-auction-agent/cdk/lib/data_pipeline_stack.py
timeout=Duration.minutes(15),  # Increase from 10 to 15
```

2. **Increase memory** (more memory = more CPU):
```python
memory_size=4096,  # Increase from 2048
```

3. **Optimize code**:
- Process in batches
- Use async/await for Amazon Bedrock calls
- Cache embeddings

---

### Issue: "Permission denied" in Lambda logs

**Error**:
```
botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the PutObject operation
```

**Solution**:

1. Get the Lambda role ARN:
   ```bash
   aws lambda get-function --function-name agent-eval-data-ingestion-dev \
     --query 'Configuration.Role' --region eu-west-1
   ```

2. Check role policies:
   ```bash
   aws iam list-attached-role-policies --role-name DataIngestionRole
   ```

3. Test S3 access:
   ```bash
   aws s3 ls s3://amzn-s3-demo-bucket-agent-eval-dev/ --region eu-west-1
   ```

---

### Issue: Amazon Bedrock model not available

**Error**:
```
ValidationException: The specified model 'anthropic.claude-opus-4-6' is not available in this region
```

**Solution**:
1. Check model availability in your region:
```bash
aws bedrock list-foundation-models --region eu-west-1 \
  --query 'modelSummaries[?contains(modelId, `claude`)]'
```

2. Update model IDs in code:
```python
# Use available model
BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-6"
```

3. If the model is not enabled, you must request model access in the Amazon Bedrock console.

---

## Data Ingestion Problems

### Issue: Sample data not found in S3

**Error**:
```
NoSuchKey: The specified key does not exist: raw/sample_vehicles.json
```

**Solution**:
Upload sample data:
```bash
aws s3 cp examples/vehicle-auction-agent/lambda/functions/data_ingestion/sample_vehicles.json \
  s3://amzn-s3-demo-bucket-agent-eval-dev/raw/sample_vehicles.json \
  --region eu-west-1
```

---

### Issue: Embeddings generation fails

**Error**:
```
ValueError: Input text exceeds maximum length (8192 tokens)
```

**Solution**:
Truncate or split long text:
```python
def truncate_text(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to maximum token length."""
    # Simple character-based approximation (1 token ≈ 4 chars)
    max_chars = max_tokens * 4
    return text[:max_chars] if len(text) > max_chars else text
```

---

## CloudWatch and Monitoring

### Issue: No metrics showing in dashboard

**Causes**:
1. Metrics not being published
2. Incorrect namespace or dimensions
3. Wrong time range selected

**Solution**:

1. **Confirm that CloudWatch is receiving published metrics**:
```bash
aws cloudwatch list-metrics --namespace AgentEvaluation/dev --region eu-west-1
```

2. **Manually publish test metric**:
```bash
aws cloudwatch put-metric-data \
  --namespace AgentEvaluation/dev \
  --metric-name TaskCompletionRate \
  --value 95.5 \
  --region eu-west-1
```

3. **Check dashboard time range** (default is last 3 hours)

---

### Issue: CloudWatch alarms not triggering

**Solution**:

1. **Check alarm state**:
```bash
aws cloudwatch describe-alarms \
  --alarm-names "agent-eval-task-completion-dev" \
  --region eu-west-1
```

2. **Verify Amazon SNS subscriptions**:
```bash
aws sns list-subscriptions-by-topic \
  --topic-arn arn:aws:sns:eu-west-1:{account}:agent-eval-alerts-dev \
  --region eu-west-1
```

3. **Subscribe to SNS topic** (if not already):
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:eu-west-1:{account}:agent-eval-alerts-dev \
  --protocol email \
  --notification-endpoint your-email@example.com \
  --region eu-west-1
```

---

## Docker and AWS CodeBuild Issues

### Issue: CodeBuild fails with "Docker login failed"

**Error**:
```
Error response from daemon: Get https://registry-1.docker.io/v2/: unauthorized
```

**Solution**:
Amazon ECR authentication has likely failed:

1. **Check CodeBuild IAM role** has `ecr:GetAuthorizationToken` permission
2. **Verify buildspec.yml** has correct ECR login command:
```yaml
pre_build:
  commands:
    - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
```

---

### Issue: Docker build fails on Mac (platform mismatch)

**Error**:
```
exec /usr/local/bin/python: exec format error
```

**Solution**:
Use CodeBuild instead of local Docker build:
```bash
# Trigger CodeBuild
aws codebuild start-build \
  --project-name agent-eval-runtime-build-dev \
  --region eu-west-1

# Monitor build
aws codebuild batch-get-builds \
  --ids <build-id> \
  --region eu-west-1
```

---

### Issue: Lambda function fails to pull ECR image

**Error**:
```
CannotPullContainerError: Error response from daemon: pull access denied
```

**Solution**:
1. **Grant Lambda permission to pull from ECR**:
```bash
# Check Lambda execution role has AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy \
  --role-name agent-eval-runtime-dev \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
```

2. **Verify image exists**:
```bash
aws ecr describe-images \
  --repository-name agent-eval-runtime-dev \
  --region eu-west-1
```

---

## Performance Problems

### Issue: High Lambda costs

**Diagnosis**:
```bash
# Check Lambda invocations
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=agent-eval-data-ingestion-dev \
  --start-time 2026-02-01T00:00:00Z \
  --end-time 2026-02-17T23:59:59Z \
  --period 86400 \
  --statistics Sum \
  --region eu-west-1
```

**Solutions**:
1. Reduce Amazon EventBridge frequency (daily to weekly)
2. Lower Lambda memory (if not CPU-bound)
3. Enable S3 lifecycle policies to archive old data

---

### Issue: Slow vector search

**Causes**:
1. Large dataset (>100K vectors)
2. In-memory loading on every invocation
3. No caching

**Solutions**:
1. **Use LanceDB persistence** (not in-memory)
2. **Enable Lambda provisioned concurrency** for warm starts
3. **Implement caching layer** (ElastiCache/Redis)

---

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Tail Lambda Logs in Real-Time

```bash
aws logs tail /aws/lambda/agent-eval-data-ingestion-dev --follow --region eu-west-1
```

### Test Agent Locally

```bash
# Run the agent locally using bedrock-agentcore
cd examples/vehicle-auction-agent/agent
DATA_BUCKET=amzn-s3-demo-bucket-agent-eval-dev \
  ENVIRONMENT=dev \
  AWS_REGION=eu-west-1 \
  python app.py
```

### Check Stack Outputs

```bash
aws cloudformation describe-stacks \
  --stack-name agent-eval-dev-data-pipeline \
  --region eu-west-1 \
  --query 'Stacks[0].Outputs'
```

---

## Getting Help

### AWS Support

- **Documentation**: https://docs.aws.amazon.com
- **Forums**: [AWS re:Post community forums](https://repost.aws)
- **Support Cases**: https://console.aws.amazon.com/support

### Community

- **GitHub Issues**: [Report an issue on GitHub](https://github.com/aws-samples/evaluating-agents-on-aws-with-strands-and-agentcore/issues)
- **Slack**: #agent-eval-support

---

## Common Commands Reference

### CDK Operations

All CDK commands run from the example's CDK directory:
```bash
cd examples/vehicle-auction-agent/cdk
```

**Validate**: synthesize every stack and check for errors:
```bash
uv run cdk synth --strict
```

**Diff**: preview changes before deploying:
```bash
AWS_REGION=eu-west-1 uv run cdk diff
```

**Deploy**: deploy all stacks:
```bash
AWS_REGION=eu-west-1 uv run cdk deploy --all --require-approval never
```

**Destroy**: tear down all stacks:
```bash
AWS_REGION=eu-west-1 uv run cdk destroy --all
```

### Lambda Operations
```bash
# Invoke function
aws lambda invoke --function-name agent-eval-data-ingestion-dev \
  --region eu-west-1 --payload '{}' response.json

# Get function config
aws lambda get-function-configuration \
  --function-name agent-eval-data-ingestion-dev \
  --region eu-west-1

# Update function code
# The ECR repo is IMMUTABLE, so reference an exact image tag (the git SHA the
# image was built from) rather than a mutable "latest" tag.
aws lambda update-function-code \
  --function-name agent-eval-runtime-dev \
  --image-uri {account}.dkr.ecr.eu-west-1.amazonaws.com/agent-eval-runtime-dev:{git-sha} \
  --region eu-west-1
```

### S3 Operations
```bash
# List objects
aws s3 ls s3://amzn-s3-demo-bucket-agent-eval-dev/ --recursive --region eu-west-1

# Download file
aws s3 cp s3://amzn-s3-demo-bucket-agent-eval-dev/lancedb/latest.json . --region eu-west-1

# Upload file
aws s3 cp local-file.json s3://amzn-s3-demo-bucket-agent-eval-dev/raw/ --region eu-west-1
```

---

## Cleaning Up

After troubleshooting is complete, remove AWS resources to avoid ongoing charges.

> **Warning:** The following commands delete data and deployed infrastructure. Verify you have backed up any results or logs you want to keep before running them.

**CDK stacks** (removes Lambda functions, EventBridge rules, CloudWatch dashboards, and SNS topics):
```bash
cd examples/vehicle-auction-agent/cdk
AWS_REGION=eu-west-1 uv run cdk destroy --all
```

**Amazon S3 objects** (empty the bucket before stack deletion if versioning is enabled):
```bash
# Back up first if you need the data:
aws s3 sync s3://amzn-s3-demo-bucket-agent-eval-dev/ ./backup/ --region eu-west-1
# Then delete:
aws s3 rm s3://amzn-s3-demo-bucket-agent-eval-dev/ --recursive --region eu-west-1
```

**Amazon ECR repository** (if not removed by CDK destroy):
```bash
aws ecr delete-repository \
  --repository-name agent-eval-runtime-dev \
  --force \
  --region eu-west-1
```

**Amazon CloudWatch Log groups** (if not removed by CDK destroy):
```bash
aws logs delete-log-group --log-group-name /aws/lambda/agent-eval-data-ingestion-dev --region eu-west-1
aws logs delete-log-group --log-group-name /aws/lambda/agent-eval-runtime-dev --region eu-west-1
```

## Conclusion

This guide covers the most common issues encountered when deploying and running the Agent Evaluation Pipeline. For problems not covered here, consult the preceding Getting Help section or open a GitHub issue with detailed logs and reproduction steps.
