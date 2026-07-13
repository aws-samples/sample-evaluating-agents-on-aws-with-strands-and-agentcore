# Performance Optimization Guide

## Introduction

This guide describes performance optimization strategies for the Agent Evaluation Pipeline. It covers AWS Lambda tuning, batch processing, and Amazon Bedrock API usage. The guide also addresses vector-search performance, target latency, and cost metrics. Use it to identify bottlenecks and apply high-impact optimizations, balancing performance against cost. Benchmarks provide a baseline; actual results vary with data size, query complexity, and traffic patterns.

## Prerequisites

- An AWS account with the reference infrastructure deployed
- AWS CLI installed and configured
- Access to Amazon CloudWatch Logs and metrics
- Familiarity with AWS Lambda configuration
- IAM permissions to modify Lambda function settings

## Table of Contents

- [Lambda Performance](#lambda-performance)
- [Data Processing Optimization](#data-processing-optimization)
- [Cost Optimization](#cost-optimization)
- [Monitoring and Profiling](#monitoring-and-profiling)
- [Scaling Strategies](#scaling-strategies)

---

## Lambda Performance

### Memory and CPU Allocation

Lambda CPU is proportional to memory allocation:

| Memory (MB) | vCPUs | Use Case |
|-------------|-------|----------|
| 512         | 0.33  | Lightweight tasks |
| 1024        | 0.67  | Standard processing |
| 2048        | 1.33  | Data ingestion (current) |
| 4096        | 2.67  | Agent runtime with LanceDB |
| 10240       | 6.00  | Heavy ML workloads |

**Current Configuration**:
- Data Ingestion: 2048 MB
- Agent Runtime: 4096 MB

**Optimization**:
1. To determine optimal memory allocation, profile actual memory usage:
```bash
aws logs filter-log-events \
  --log-group-name /aws/lambda/agent-eval-data-ingestion-dev \
  --filter-pattern "Max Memory Used" \
  --region eu-west-1
```

2. Right-size based on usage:
```python
# If max memory used < 1500 MB, reduce to 1536 MB
memory_size=1536,
```

---

### Cold Start Optimization

**Cold Start Times**:
- Zip deployment: 100-500ms
- Container image: 500-1500ms
- With large dependencies: 2-5 seconds

**Strategies to Reduce Cold Starts**:

1. **Provisioned Concurrency** (for critical paths):
```python
# In CDK
self.agent_function.add_alias(
    "live",
    provisioned_concurrent_executions=2  # Always warm
)
```

2. **Lambda SnapStart** (Java only, not Python)

3. **Reduce package size**:
```bash
# Use Lambda layers for shared dependencies
# Exclude unnecessary files
zip -r function.zip . -x "*.pyc" "**/__pycache__/*" "tests/*"
```

4. **Keep connections warm**:
```python
# Reuse boto3 clients outside handler
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    # Use existing clients
    pass
```

---

### Batch Processing

Process multiple items per invocation:

**Before** (1 item per invocation):
```python
def lambda_handler(event, context):
    vehicle = event['vehicle']
    process_vehicle(vehicle)
```

**After** (batch of items):
```python
def lambda_handler(event, context):
    vehicles = event['vehicles']  # Batch of 10-100
    results = [process_vehicle(v) for v in vehicles]
    return results
```

**Benefits**:
- Reduced invocation overhead
- Lower cost (fewer invocations)
- Better throughput

---

## Data Processing Optimization

### Amazon Bedrock API Optimization

**Current**: Sequential API calls
```python
for vehicle in vehicles:
    embedding = get_embedding(vehicle['description'])
```

**Optimized**: Batch requests
```python
# Batch embeddings (Amazon Titan supports up to 128KB input)
batch_size = 10
for i in range(0, len(vehicles), batch_size):
    batch = vehicles[i:i+batch_size]
    texts = [v['description'] for v in batch]
    embeddings = get_embeddings_batch(texts)
```

**Amazon Bedrock Quotas** (eu-west-1):
- Text generation models (such as Claude from Anthropic) are throttled by tokens per minute (TPM).
- Embedding models (such as Amazon Titan Text Embeddings V2) are throttled by requests per minute (RPM), not TPM.
- Claude Sonnet 4.6 from Anthropic: default quotas of 10K input TPM and 2K output TPM. Quota values vary by account and region.
- Check the Service Quotas console for the current limits on your account, since AWS updates default quotas over time.

**Strategies**:
1. **Use batch APIs** where available
2. **Implement exponential backoff** for rate limits
3. **Cache embeddings** for repeated text

---

### LanceDB Query Optimization

**Vector Search Performance**: Actual results vary by workload and configuration; the following figures are illustrative benchmarks.
- 1K vectors: <10ms
- 10K vectors: <50ms
- 100K vectors: <200ms
- 1M vectors: <1s (with proper indexing)

**Optimization Tips**:

1. **Create indexes**:
```python
# Create IVF-PQ index (Inverted File with Product Quantization) for large datasets
table.create_index(
    metric="cosine",
    num_partitions=256,
    num_sub_vectors=96
)
```

2. **Limit result sets**:
```python
# Only fetch what you need
results = table.search(query_vector).limit(10)
```

3. **Use column projections**:
```python
# Don't fetch all columns
results = table.search(query_vector).select(["id", "make", "model"]).limit(10)
```

---

### Amazon S3 Transfer Optimization

**Large File Uploads**:
```python
# Use multipart upload for files >5MB
s3_client.upload_file(
    'large_file.json',
    'amzn-s3-demo-bucket',
    'key',
    Config=boto3.s3.transfer.TransferConfig(
        multipart_threshold=5 * 1024 * 1024,
        multipart_chunksize=5 * 1024 * 1024
    )
)
```

**S3 Select** (for large JSON files):
```python
# Query JSON without downloading entire file
response = s3_client.select_object_content(
    Bucket='amzn-s3-demo-bucket',
    Key='data.json',
    Expression='SELECT * FROM S3Object[*] WHERE make = "Toyota"',
    ExpressionType='SQL',
    InputSerialization={'JSON': {'Type': 'DOCUMENT'}},
    OutputSerialization={'JSON': {}}
)
```

---

## Cost Optimization

### Current Costs (Dev Environment)

| Resource | Monthly Cost |
|----------|--------------|
| Lambda (ingestion) | $2-5 |
| Lambda (runtime) | $5-10 |
| S3 (storage <1GB) | $0.02-0.05 |
| CloudWatch Logs | $0.50-1.00 |
| CloudWatch Dashboard | $3.00 |
| Amazon Bedrock (embeddings) | $5-20 |
| **Total** | **~$15-40/month** |

### Cost Reduction Strategies

#### 1. Amazon EventBridge Frequency

**Current**: Daily refresh (1 AM UTC)
```python
schedule=events.Schedule.cron(minute="0", hour="1")
```

**Optimized**: Weekly refresh (for stable datasets)
```python
schedule=events.Schedule.cron(minute="0", hour="1", week_day="SUN")
```

**Estimated savings**: Actual results vary by workload and configuration; the following figure is an illustrative benchmark. In some scenarios, this may reduce Lambda invocations by up to 85% compared to daily refresh, depending on your refresh schedule and workload patterns.

#### 2. S3 Lifecycle Policies

```python
s3.LifecycleRule(
    id="ArchiveOldData",
    transitions=[
        s3.Transition(
            storage_class=s3.StorageClass.GLACIER_INSTANT_RETRIEVAL,
            transition_after=Duration.days(30)
        ),
        s3.Transition(
            storage_class=s3.StorageClass.GLACIER_DEEP_ARCHIVE,
            transition_after=Duration.days(90)
        )
    ]
)
```

**Estimated savings**: Actual results vary by workload and configuration; the following figure is an illustrative benchmark. Glacier storage classes typically cost 50-90% less than S3 Standard for archived data. Retrieval costs apply, and actual savings depend on access patterns and usage.

#### 3. CloudWatch Logs Retention

**Current**: 7 days (dev), 30 days (prod)

**Optimized**: Use S3 for long-term log storage
```bash
# Export logs to S3 after 7 days
aws logs create-export-task \
  --log-group-name /aws/lambda/agent-eval-data-ingestion-dev \
  --from 1707264000000 \
  --to 1707868800000 \
  --destination amzn-s3-demo-bucket-logs-archive \
  --destination-prefix lambda-logs/
```

#### 4. Amazon Bedrock Model Selection

*Prices are illustrative examples only. See the official AWS pricing pages for current rates.*

| Model | Input Cost (per 1K tokens) | Output Cost (per 1K tokens) |
|-------|----------------------------|------------------------------|
| Amazon Titan Text Embeddings V2 | $0.0001 | N/A |
| Amazon Nova Lite v1 | $0.00006 | $0.00024 |
| Claude Sonnet 4.6 from Anthropic | $0.003 | $0.015 |

**Strategy**: Use cheaper models for simple tasks

---

## Monitoring and Profiling

### Lambda Insights

Enable Lambda Insights for detailed metrics:

```python
# In CDK
self.ingestion_function.add_environment(
    "AWS_LAMBDA_EXEC_WRAPPER",
    "/opt/otel-instrument"
)

# Add Lambda Insights layer
lambda_insights_layer = lambda_.LayerVersion.from_layer_version_arn(
    self,
    "LambdaInsightsLayer",
    # Documentation example. Replace this with the actual AWS-provided Lambda Insights layer Amazon Resource Name (ARN) for your region.
    f"arn:aws:lambda:{self.region}:123456789012:layer:LambdaInsightsExtension:21"
)
self.ingestion_function.add_layers(lambda_insights_layer)
```

**Metrics Provided**:
- Memory utilization
- CPU time
- Network I/O
- Disk I/O

### X-Ray Tracing

Enable distributed tracing:

```python
# In CDK
self.agent_function.enable_tracing(lambda_.Tracing.ACTIVE)

# In function code
from aws_xray_sdk.core import xray_recorder

@xray_recorder.capture('process_vehicle')
def process_vehicle(vehicle):
    # Function logic
    pass
```

### Custom Metrics

Publish detailed performance metrics:

```python
def publish_performance_metrics(duration_ms, memory_mb):
    cloudwatch.put_metric_data(
        Namespace='AgentEvaluation/Performance',
        MetricData=[
            {
                'MetricName': 'ProcessingDuration',
                'Value': duration_ms,
                'Unit': 'Milliseconds'
            },
            {
                'MetricName': 'MemoryUsage',
                'Value': memory_mb,
                'Unit': 'Megabytes'
            }
        ]
    )
```

---

## Scaling Strategies

### Concurrent Execution Limits

**Default**: 1000 concurrent executions per region

**Per-function limits**:
```bash
# Set reserved concurrency
aws lambda put-function-concurrency \
  --function-name agent-eval-runtime-dev \
  --reserved-concurrent-executions 100 \
  --region eu-west-1
```

### Auto-scaling Strategy

**Lambda**: Automatically scales (no configuration needed)

**Amazon Bedrock**: Request quota increases via AWS Support

**S3**: No scaling needed (auto-scales)

### Multi-Region Deployment

For global availability:

```python
# Deploy to multiple regions
regions = ["eu-west-1", "us-east-1", "ap-southeast-1"]

for region in regions:
    DataPipelineStack(
        app,
        f"agent-eval-dev-data-pipeline-{region}",
        env=cdk.Environment(account=account, region=region)
    )
```

---

## Performance Benchmarks

### Data Ingestion (10,000 vehicles)

Actual results vary by workload and configuration; the following figures are illustrative benchmarks from testing with 10,000 vehicles. Actual performance and costs may vary based on data characteristics, network conditions, and concurrent load.

| Configuration | Duration | Cost |
|---------------|----------|------|
| 1024 MB | 180s | $0.03 |
| 2048 MB (current) | 90s | $0.035 |
| 4096 MB | 50s | $0.042 |

**Recommendation**: Testing showed that 2048 MB typically offers a favorable price/performance balance for this workload.

### Agent Runtime (1 query)

Actual results vary by workload and configuration; the following figures are illustrative benchmarks. Actual latency may vary based on query complexity, data size, and system load.

| Configuration | P50 Latency | P99 Latency | Cost per 1K queries |
|---------------|-------------|-------------|---------------------|
| 2048 MB | 800ms | 2.5s | $1.20 |
| 4096 MB (current) | 500ms | 1.8s | $1.80 |
| 8192 MB | 400ms | 1.5s | $2.40 |

**Recommendation**: 4096 MB for production (in testing, this configuration typically met the <2s target under normal load conditions; actual performance may vary)

---

## Performance Testing

### Load Testing with Artillery

```yaml
# artillery.yml
config:
  target: https://api.example.com
  phases:
    - duration: 60
      arrivalRate: 10
      rampTo: 100

scenarios:
  - name: "Agent Query"
    flow:
      - post:
          url: "/query"
          json:
            query: "Find Toyota Camry"
            dealer_id: "D123"
```

```bash
artillery run artillery.yml
```

### Stress Testing Lambda

```bash
# Invoke Lambda 100 times concurrently
for i in {1..100}; do
  aws lambda invoke \
    --function-name agent-eval-runtime-dev \
    --region eu-west-1 \
    --invocation-type Event \
    --payload '{"query":"test"}' \
    /dev/null &
done
wait
```

---

## Cleaning Up Performance Resources

> **Warning:** Deleting CloudWatch dashboards and alarms removes monitoring configuration permanently. Export them before deletion if you need to preserve dashboard layouts or alarm thresholds: `aws cloudwatch get-dashboard --dashboard-name <name> > dashboard-backup.json`

If performance-testing resources are no longer needed:

1. Remove Lambda Insights layers from functions (CDK redeploy without the layer).
2. Disable Provisioned Concurrency: `aws lambda delete-provisioned-concurrency-config --function-name FUNCTION_NAME --qualifier ALIAS`
3. Delete custom CloudWatch dashboards: `aws cloudwatch delete-dashboards --dashboard-names DASHBOARD_NAME`
4. Delete CloudWatch alarms created for benchmarking: `aws cloudwatch delete-alarms --alarm-names ALARM_NAME`
5. Stop publishing custom CloudWatch metrics to the `AgentEvaluation/Performance` namespace. Custom metrics cannot be manually deleted; they expire automatically after 15 months of inactivity. To avoid ongoing metric storage costs, confirm you have stopped all calls to `put_metric_data` for that namespace.

Confirm deletion by listing each resource and checking it returns empty or not-found:
```bash
# Verify Provisioned Concurrency removed
aws lambda get-provisioned-concurrency-config --function-name FUNCTION_NAME --qualifier ALIAS
# Verify dashboards removed
aws cloudwatch list-dashboards --dashboard-name-prefix DASHBOARD_NAME
# Verify alarms removed
aws cloudwatch describe-alarms --alarm-names ALARM_NAME
# Verify custom metrics are no longer actively published
aws cloudwatch list-metrics --namespace AgentEvaluation/Performance
```

---

## Conclusion

Performance optimization is iterative. Start with the AWS Lambda memory and batch-processing changes for immediate gains, then profile your workload to find bottlenecks. Track P50/P99 latency, cost per query, and throughput, and tune to your requirements. The benchmarks here are a baseline; results will vary with data size, query complexity, and traffic.

---

## References

- [AWS Lambda Performance Optimization](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [Amazon Bedrock Quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html)
- [LanceDB Performance Guide](https://lancedb.github.io/lancedb/)
