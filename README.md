# agentic-evaluation

[![License: MIT-0](https://img.shields.io/badge/License-MIT--0-yellow.svg)](https://opensource.org/licenses/MIT-0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![AWS CDK](https://img.shields.io/badge/AWS--CDK-2.100+-orange.svg)](https://aws.amazon.com/cdk/)

> **Note:** AWS code samples are example code that demonstrates practical implementations of AWS services for specific use cases and scenarios.
>
> These application solutions are not supported products in their own right, but educational examples to help our customers use our products for their applications. As our customer, any applications you integrate these examples into should be thoroughly tested, secured, and optimized according to your business's security standards and policies before deploying to production or handling production workloads.
>
> Before any real-world use, review the security controls in [docs/SECURITY.md](docs/SECURITY.md) and the [Production security recommendations](#production-security-recommendations) below, and adapt the code to your own requirements.

A framework-agnostic software development kit (SDK) for evaluating artificial intelligence (AI) agents. It supports Strands Agents, Amazon Bedrock AgentCore, LangChain, CrewAI, and OpenAI. The SDK also works with custom HTTP endpoints and most agents that can be wrapped in a callable.
Run a 3-layer evaluation (tool usage, reasoning, output quality) plus a
domain-specific layer for safety, scoping, latency, and cost.

This repository also contains a **reference deployment** on AWS using Strands +
Amazon Bedrock AgentCore (a car-auction marketplace dealer stock-search agent). See the
[Reference deployment](#reference-deployment-car-auction-dealer-search) section following.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager (v0.4+)
- Git (to clone the repository)

Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

> **Note:** The no-AWS demo (`quickstart/run_demo.py`) does not require AWS credentials or Amazon Bedrock access.

## SDK quickstart (no AWS, <60s)

```bash
git clone https://github.com/aws-samples/sample-evaluating-agents-on-aws-with-strands-and-agentcore.git
cd sample-evaluating-agents-on-aws-with-strands-and-agentcore
uv sync
uv run python quickstart/run_demo.py
```

You'll see Layer 1 (tool usage), Layer 2/3 (judge defaults to no-op so the
demo needs no LLM), and the domain layer evaluate a mock agent in seconds.

## Use it for *your* agent

```python
from agentic_evaluation import run_all_layers, TaskFnResult
from strands_evals.types.evaluation import EnvironmentState

def task_fn(case) -> TaskFnResult:
    # Replace with your agent: Strands, AgentCore, LangChain, OpenAI, ...
    metrics = {"latency_ms": 420}
    return {
        "output": "the answer",
        "trajectory": ["search", "answer"],
        "environment_state": [EnvironmentState(name="metrics", state=metrics)],
        "metadata": metrics,  # Optional fallback for direct/static fixtures.
    }

results = run_all_layers(task_fn=task_fn)
print("All passed:", results["all_passed"])
```

Or via the CLI:

```bash
agentic-eval init --name my-agent --tools "search,answer"
agentic-eval validate --config eval_config.yaml
agentic-eval run --config eval_config.yaml --task-fn my_pkg.tasks:run
```

Your adapter must return `output`, `trajectory`, and an `EnvironmentState`
named `metrics` for enabled operational evaluators; see
[docs/SDK_GUIDE.md](docs/SDK_GUIDE.md) for framework adapters.

For per-framework recipes (Strands, Amazon Bedrock AgentCore, LangChain, CrewAI, OpenAI, HTTP)
see [docs/SDK_GUIDE.md](docs/SDK_GUIDE.md).

---

## Reference deployment: car-auction dealer search

The rest of this repo is a complete reference implementation of the SDK
on AWS with Strands and Amazon Bedrock AgentCore. The deployment-specific docs are in the following sections.

This repository accompanies the AWS blog post: **[Evaluating production AI agents on AWS with Strands and AgentCore](https://aws.amazon.com/blogs/machine-learning/evaluating-production-ai-agents-on-aws-with-strands-and-agentcore/)**. The blog post provides conceptual understanding, architecture patterns, and key decision points. This repository provides a complete reference implementation of those patterns. Read the blog post first for context, then return here to explore the code.

## What You'll Build

A complete evaluation pipeline for AI agents that:

- **Catches issues before deployment** with build-time *offline* evaluation (strands-agents-evals)
- **Monitors production behavior** with continuous *online* evaluation by sampling live traffic (Amazon Bedrock AgentCore Evaluations)
- **Integrates with CI/CD** as quality gates blocking bad deploys
- **Provides observability** through OpenTelemetry instrumentation
- **Demonstrates patterns for scaling** that were tested with 1,500+ concurrent users

**Use Case**: Based on a real-world car auction marketplace dealer stock search agent. This conversational AI searches 1,500-2,500 vehicles daily across dozens of vehicle attributes. It uses natural language queries, hybrid search with LanceDB, and Amazon Bedrock models. LanceDB is a vector database for storing and searching embeddings, the numerical vector representations of text.

**Estimated Cost**: Approximately $55-105/month for dev/test environments,
including four customer-managed KMS keys, plus usage-dependent model and
AgentCore requests.

## Architecture

The diagrams below are rendered from editable source in [`docs/diagrams/`](docs/diagrams/) (draw.io `.drawio` files, one per PNG). Open a `.drawio` file at [app.diagrams.net](https://app.diagrams.net) to edit, then re-export the matching PNG in `docs/images/`.

### Data Ingestion Pipeline

![Architecture diagram showing a daily Amazon EventBridge trigger invoking an AWS Lambda ingestion function that reads sample vehicle data from Amazon S3, normalizes records and builds descriptions locally, generates vectors with Amazon Titan Text Embeddings V2, and persists rows and vectors in Amazon S3. Amazon Bedrock AgentCore Runtime checksum-verifies, materializes, and periodically refreshes a bounded local LanceDB cache. A production BigQuery or marketplace adapter is shown outside the AWS boundary and is not implemented in the sample.](docs/images/data-ingestion-pipeline.png)

Daily automated refresh of vehicle inventory:
- **Amazon EventBridge**: Scheduled daily trigger before auction starts
- **AWS Lambda**: Ingestion function that reads vehicle data from the source (BigQuery is mocked in this sample; it reads sample data from Amazon S3)
- **AWS Lambda preprocessing**: Deterministic schema normalization and searchable description construction
- **Amazon Titan Text Embeddings V2**: 1,024-dimension vector generation
- **Amazon S3 + LanceDB**: Amazon S3 stores immutable, SHA-256-versioned snapshots and a manifest promoted last; each warm Runtime process verifies and materializes the active snapshot as a bounded local LanceDB cache and polls for newer versions

### Agent Runtime Architecture

![Architecture diagram showing an IAM- or Cognito-authenticated Amazon Bedrock AgentCore Runtime orchestrating 7 local tools plus a dealer-profile MCP tool served through Amazon Bedrock AgentCore Gateway. The runtime injects the trusted dealer identity so the model cannot select another dealer. Claude Sonnet 4.6 uses an Amazon Bedrock Guardrail, Titan Text Embeddings V2 embeds search queries, AgentCore Memory stores dealer preferences and facts, and a local LanceDB table is materialized from Amazon S3 for vector search.](docs/images/agent-runtime.png)

AgentCore Runtime orchestrates **8 tools** — 7 local plus 1 served through the Gateway (see `examples/vehicle-auction-agent/agent/app.py`):
- **Search and retrieval**: `search_vehicles` (structured filters), `run_sql` (validated pandas query), `hybrid_search` (semantic), `filter_by_distance` (geo)
- **Supporting**: `get_schema`, `get_embedding`, `get_bids`
- **Dealer profile**: a zero-argument, dealer-scoped wrapper calls the **Amazon Bedrock AgentCore Gateway**; the runtime injects the authenticated dealer ID and never exposes the Gateway's list operation or path parameter to the model
- **Amazon Bedrock AgentCore Memory**: cross-session dealer memory (preferences and facts) via built-in user-preference and semantic strategies
- **Claude Sonnet 4.6 from Anthropic (available through Amazon Bedrock)**: Reasoning and orchestration
- **LanceDB**: Amazon S3 persists rows and vectors; the runtime uses native cosine search and scalar prefilters, with a bounded pandas fallback only for geospatial distance calculations

### Three-Layer Evaluation Framework

![Diagram showing three-layer evaluation framework: Layer 1 Tool Usage (>95% threshold for tool correctness), Layer 2 Reasoning (>85% threshold for process evaluation), Layer 3 Output Quality (>90% threshold for outcome evaluation), plus a domain layer for operational metrics (latency, cost, safety, freshness).](docs/images/three-layer-framework.png)

Build-time (offline) evaluation across three layers. Each layer maps onto the
vocabulary used by the wider evaluation landscape (DeepEval, Ragas, Anthropic),
so the concepts transfer if you already use another tool:

1. **Tool Usage (>95% threshold)**: correct tool selection and parameter validation.
   *Standard term:* **tool correctness / tool-call accuracy** (DeepEval `ToolCorrectnessMetric`, Ragas `ToolCallAccuracy`).
2. **Reasoning (>85% threshold)**: coherent decision-making scored by an LLM judge over the trajectory.
   *Standard term:* **process evaluation** (judging *how* the agent decided).
3. **Output Quality (>90% threshold)**: helpful, accurate, actionable final responses.
   *Standard term:* **outcome evaluation / goal success** (Ragas `AgentGoalAccuracy`, DeepEval `TaskCompletion`).

All three layers must pass before deployment proceeds.

> **Terminology note.** The layer names in the following table are canonical (they match the
> accompanying blog post). The SDK also accepts and emits the standard aliases:
> `run_all_layers()` results are keyed by both `layer_1`/`layer_2`/`layer_3`/`domain`
> *and* `tool_correctness`/`process_evaluation`/`outcome_evaluation`/`operational_metrics`
> (the alias points at the same object). `eval_config.yaml` accepts either name in
> `evaluation_layers`.

### How each layer runs at build time (offline) vs. in production (online)

The same evaluation *concepts* run in two places, with different mechanics. The
field calls these **offline evaluation** and **online evaluation** (the terms
used by LangSmith, Braintrust, and Anthropic), which this project also describes
as "build time" and "production":

- **Offline / build time** is the SDK (`run_all_layers`) running in CI against a
  fixed set of curated test cases with ground-truth expectations, a hard gate
  that blocks the deploy.
- **Online / production** is Amazon Bedrock AgentCore Evaluations, an AWS-managed service that
  samples a small fraction of *real* live traffic and scores it continuously,
  alerting instead of blocking.

The following table compares how each evaluation layer runs in two contexts: offline during build time (SDK in CI) versus online in production (Amazon Bedrock AgentCore). Each of the four layers (Tool Usage, Reasoning, Output Quality, and Domain) uses different mechanics depending on the context.

| Layer | Offline: build time (SDK, CI gate) | Online: production (Amazon Bedrock AgentCore, continuous) |
|-------|---------------------------|------------------------------------|
| Layer 1: Tool Usage | `ToolSelectionGrader` + `TrajectoryOrderGrader` (deterministic, no LLM) compare each test case's trajectory to its `expected_tools`. Fast, runs on every commit. | `Builtin.ToolSelection` scores tool choice on sampled live sessions. |
| Layer 2: Reasoning | `HelpfulnessEvaluator` + `TrajectoryEvaluator` (LLM-as-judge) score decision quality against a rubric. Needs Amazon Bedrock access. | `Builtin.Helpfulness` on sampled traffic. |
| Layer 3: Output Quality | `OutputEvaluator` + `GoalSuccessRateEvaluator` (LLM-as-judge) score the final answer. | `Builtin.GoalSuccessRate` on sampled traffic. |
| Domain | Deterministic evaluators (latency, cost, safety, freshness, scoping) read each case's `metadata`. No LLM. | Operational telemetry in Amazon CloudWatch; optional custom AgentCore evaluators can be created separately. |

**Key differences:**
- **Coverage**: build time runs *every* curated test case; production samples 1–5% of live traffic.
- **Failure action**: build time **blocks the deploy** (`results["all_passed"]` is the gate); production **alerts** (Amazon SNS / CloudWatch) without blocking users.
- **Data**: build time uses fixed expected trajectories/outputs; production has no ground truth, so it leans on LLM judges and deterministic telemetry checks.
- **Feedback loop**: a production failure becomes a new build-time test case (see [Production feedback loop](docs/EVALUATION_GUIDE.md#production-feedback-loop)), so the gate gets stricter over time.

### Production Evaluation Architecture

![Architecture diagram showing authenticated requests to Amazon Bedrock AgentCore Runtime, OpenTelemetry spans in Amazon CloudWatch Logs, and an optional, manually configured Amazon Bedrock AgentCore online evaluation. The README CLI example selects ToolSelection, Helpfulness, and GoalSuccessRate built-in evaluators. AgentCore writes CloudWatch results, while connecting those service metrics to the CDK-created custom-metric alarms remains an explicit integration step.](docs/images/production-evaluation.png)

Amazon Bedrock AgentCore Evaluations can provide continuous production monitoring. CDK creates supporting IAM, log, alarm, and SNS resources, but it does not create the online evaluation configuration. The CLI example in [Configure Production Monitoring](#6-configure-production-monitoring) selects three built-in evaluators and samples 3% of traffic. Publishing or mapping evaluation scores into the CDK alarms' `AgentEvaluation/{environment}` namespace is not implemented.

### Six-Stage Deployment Pipeline

![Diagram showing a recommended six-stage delivery model. This repository automates unit tests and lint in GitHub Actions, provides SDK capabilities for tool, trajectory, and LLM-as-judge evaluation, includes a manual post-deployment smoke script, and documents optional AgentCore online evaluation configuration. Wiring Stages 2 through 6 into one blocking deployment workflow is left to the adopter.](docs/images/deployment-pipeline.png)

This is a recommended delivery model, not a fully provisioned pipeline. The repository's `sdk-ci.yml` automates Stage 1. The SDK implements the Stage 2-4 checks, `scripts/post_deploy_eval.py` supports a manual Stage 5 smoke run, and the README CLI example configures optional Stage 6 monitoring. A deployment orchestrator that blocks promotion on Stages 2-5 is not included.

1. **Unit Tests and Lint**: Code quality baseline
2. **Tool Correctness**: ToolSelectionGrader and TrajectoryOrderGrader verify selection and ordering
3. **Trajectory Tests**: Multi-step workflow validation
4. **LLM-as-Judge**: OutputEvaluator, HelpfulnessEvaluator, GoalSuccessRateEvaluator score quality
5. **Staging Validation**: Manual SDK smoke evaluation against a deployed AgentCore Runtime
6. **Online Evaluation**: Optional production sampling configured after deployment

## Prerequisites

- **AWS Account** with access to Amazon Bedrock and Amazon Bedrock AgentCore
- **AWS CDK** v2.241+ installed and bootstrapped: `npm install -g aws-cdk`
- **Python 3.14+**: Check with `python --version`
- **[uv](https://docs.astral.sh/uv/) package manager (v0.4+)**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **GitHub account** (for CI/CD pipeline via GitHub Actions)
- **Amazon Bedrock model access**: Request access to the following foundation models in Amazon Bedrock:
  - Amazon Titan Text Embeddings V2
  - Claude Sonnet 4.6 from Anthropic (available through Amazon Bedrock)

### AWS Service Quotas

Verify that you have sufficient quotas for:
- Amazon Bedrock: Claude Sonnet 4.6 (50,000 TPM minimum)
- AWS Lambda: 1,000 concurrent executions
- Amazon S3: No specific quota concerns
- Amazon EventBridge: 300 rules (well under default limits)

## Quick Start

### 1. Clone and Set Up Environment

Clone the repository:

```bash
git clone https://github.com/aws-samples/sample-evaluating-agents-on-aws-with-strands-and-agentcore.git
cd sample-evaluating-agents-on-aws-with-strands-and-agentcore
```

Install dependencies with uv:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 2. Configure AWS Credentials

```bash
aws configure --profile <profile>
```

### 3. Verify Amazon Bedrock Access

```bash
aws bedrock list-foundation-models \
  --profile <profile> \
  --region eu-west-1
```

### 4. Deploy CDK Infrastructure

Bootstrap only after confirming that STS returns the expected account. Skip this
step if that account and region are already bootstrapped:

```bash
aws sts get-caller-identity \
  --profile <profile> \
  --region eu-west-1
cdk bootstrap \
  --profile <profile> \
  aws://<12-digit-account-id>/eu-west-1
```

Deploy everything with the end-to-end script (run from the repository root):

```bash
python scripts/deploy_stack.py \
  --profile <profile> \
  --region eu-west-1 \
  --expected-account <12-digit-account-id>
```

This is the supported one-shot flow. It builds the agent container in AWS CodeBuild (no local Docker), pushes it to Amazon ECR, then runs `cdk deploy --all` with the resulting image URI so the **agent-runtime stack is included**. Note the outputs (AgentCore Runtime endpoint, S3 bucket name, etc.).

> **Why not bare `cdk deploy --all`?** The agent-runtime stack only deploys when an `agent_image_uri` context value is present (see `examples/vehicle-auction-agent/cdk/app.py`). Running `cdk deploy --all` directly with no prebuilt image **silently skips the agent runtime**. You get the data, dealer-API, evaluation, and monitoring stacks but not the agent itself. Use `deploy_stack.py` so the image is built and wired in for you.
>
> Later, to re-deploy CDK-only changes without rebuilding the image, add
> `--skip-build --image-tag <existing-git-sha>`. The script resolves that tag
> to an immutable ECR digest before `cdk diff` or deployment.

### 5. Run Build-Time Quality Checks

```bash
# Run all tests (excluding deployed infrastructure tests)
uv run pytest tests/ -m "not deployed" -v

# Run unit tests only (fast, no AWS/LLM calls)
uv run pytest tests/unit/ -v

# Run full evaluation pipeline with LLM judges (requires Amazon Bedrock access)
uv run pytest tests/integration/test_full_evaluation.py -v

# Run deployed tests only after reviewing request cost and approving this exact
# account/region. The test harness re-verifies STS before every deployed test.
AWS_PROFILE=<profile> \
AWS_ACCOUNT_ID=<12-digit-account-id> \
AWS_REGION=eu-west-1 \
ENVIRONMENT=dev \
AWS_DEPLOYED_TEST_APPROVAL="run-deployed-tests <12-digit-account-id> eu-west-1" \
uv run pytest tests/ -m deployed -v

# The six real end-to-end tests in tests/integration/test_real_agent_eval.py skip
# unless you also point them at your deployed runtime. Both variables are needed:
# the runtime withholds the privileged trajectory/usage telemetry from callers
# that do not present the evaluation token. Get the runtime ARN from
# `aws bedrock-agentcore-control list-agent-runtimes` and the secret ARN from the
# EvaluationTraceSecretArn output of the agent-eval-<env>-agent-runtime stack.
AGENT_RUNTIME_ARN=arn:aws:bedrock-agentcore:eu-west-1:<account-id>:runtime/<runtime-id> \
EVALUATION_TRACE_SECRET_ID=arn:aws:secretsmanager:eu-west-1:<account-id>:secret:agent-eval/evaluation-trace/dev-XXXXXX \
uv run pytest tests/integration/test_real_agent_eval.py -m deployed -v
```

All tests should pass with no failures. A successful run produces output similar to:

```text
===== X passed in Y.YYs =====
```

If any tests fail, review the error output and confirm your AWS credentials and Amazon Bedrock model access are configured correctly.

After completing these steps, you have deployed the agent evaluation infrastructure and verified it works correctly. You can now configure production monitoring or customize the evaluation pipeline for your use case.

### 6. Configure Production Monitoring

To enable Amazon Bedrock AgentCore Online Evaluation, use the AWS Console or CLI:

Compare the STS result with the expected account, review the evaluation traffic
cost, and obtain approval before running the mutating
`create-online-evaluation-config` command:

```bash
aws sts get-caller-identity \
  --profile <profile> \
  --region eu-west-1
aws bedrock-agentcore-control create-online-evaluation-config \
  --online-evaluation-config-name dealer-search-online-eval \
  --rule '{"samplingConfig":{"samplingPercentage":3.0},"sessionConfig":{"sessionTimeoutMinutes":30}}' \
  --data-source-config '{"cloudWatchLogs":{"logGroupNames":["/aws/bedrock-agentcore/agent-eval-dev"],"serviceNames":["agent-eval-dev-agent-runtime"]}}' \
  --evaluators '[{"evaluatorId":"Builtin.Helpfulness"},{"evaluatorId":"Builtin.GoalSuccessRate"},{"evaluatorId":"Builtin.ToolSelection"}]' \
  --evaluation-execution-role-arn arn:aws:iam::<12-digit-account-id>:role/agent-eval-agentcore-dev \
  --profile <profile> \
  --region eu-west-1
```

`samplingPercentage` is a percentage from 0.01 to 100, so `3.0` samples 3% of traffic.

## Repository Structure

```text
.
├── README.md                          # This file
├── CONTRIBUTING.md                    # Contribution guidelines
├── LICENSE                            # MIT-0 License
├── pyproject.toml                     # uv package configuration
├── eval_config.yaml                   # Evaluation config (tools, rubrics, thresholds, test cases)
├── .gitignore                         # Git ignore patterns
├── .github/workflows/                 # GitHub Actions CI/CD (sdk-ci.yml, publish.yml)
│
├── docs/                              # Documentation
│   ├── EVALUATION_GUIDE.md     # Reference guide for using this framework
│   ├── SECURITY.md                    # Security best practices
│   ├── TROUBLESHOOTING.md             # Common issues and fixes
│   ├── PERFORMANCE.md                 # Performance tuning guide
│   └── images/                        # Architecture diagrams (PNG)
│
├── src/agentic_evaluation/            # THE PRODUCT: evaluation SDK (the agentic-evaluation wheel)
│   ├── __init__.py
│   ├── config.py                      # YAML config loader
│   ├── evaluators.py                  # strands-evals Evaluator subclasses
│   ├── judges.py                      # Pluggable LLM judge backends
│   ├── generate_cases.py              # Test-case generator helpers
│   ├── run_experiment.py              # Experiment builders + run_all_layers()
│   ├── test_cases.py                  # TestCase, TestCaseRegistry
│   ├── thresholds.py                  # EvaluationThresholds dataclass
│   ├── adapters/                      # strands_local / agentcore / http task_fns
│   └── examples/                      # Reusable evaluator presets
│
├── examples/vehicle-auction-agent/    # THE EXAMPLE: reference agent the SDK is tested against
│   ├── agent/                         # Agent code (AgentCore Runtime)
│   │   ├── app.py                     # Strands agent: 7 local tools + Memory + Gateway
│   │   ├── requirements.txt           # Agent dependencies
│   │   ├── utils/gateway.py           # AgentCore Gateway MCP client (dealer profile)
│   │   └── utils/geo.py               # Haversine + bounding box
│   ├── cdk/                           # AWS CDK infrastructure (Python)
│   │   ├── app.py                     # CDK app entry point
│   │   └── lib/
│   │       ├── data_pipeline_stack.py # EventBridge + Lambda + S3
│   │       ├── agent_runtime_stack.py # AgentCore Runtime (L2 construct)
│   │       ├── dealer_api_stack.py    # DynamoDB + API Gateway
│   │       ├── evaluation_stack.py    # CloudWatch + SNS + IAM
│   │       └── monitoring_stack.py    # Dashboards and alarms
│   └── lambda/functions/              # Lambda functions
│       ├── data_ingestion/            # BigQuery to S3 pipeline
│       └── dealer_api/                # Dealer profile API
│
├── tests/                             # Test suites
│   ├── conftest.py                    # Pytest fixtures
│   ├── sdk/                           # SDK-only tests (no AWS, no LLM)
│   │   ├── unit/                      # cli, judges, plugins, layer gating, scoping
│   │   └── integration/              # quickstart end-to-end
│   ├── unit/
│   │   ├── test_advanced_tools.py     # Example-agent tool tests
│   │   └── test_thresholds.py         # Threshold config tests
│   └── integration/                   # Live/deployed tests (require AWS)
│       ├── test_full_evaluation.py    # Build-time gate with LLM judges
│       ├── test_real_agent_eval.py    # Full pipeline against the live agent
│       ├── test_deployed_stack.py     # Post-deployment checks
│       └── test_e2e_smoke.py          # E2E smoke tests
│
└── scripts/                           # Deploy + data helpers for the example
    ├── deploy_stack.py                # Build image (CodeBuild) then cdk deploy
    ├── deploy_codebuild.py            # Build the agent image in CodeBuild
    ├── post_deploy_eval.py            # Smoke-eval a deployed runtime
    ├── seed_dealer_data.py            # Seed dealer profiles into DynamoDB
    └── validate_deployment.py         # Post-deployment health check
```

> New project? Scaffold a config with `agentic-eval init` (see the preceding SDK quickstart section).

## Testing Approaches

### Run all assessment layers

```python
from agentic_evaluation import run_all_layers
from strands_evals.types.evaluation import EnvironmentState

def agent_task(case):
    # Replace with your real agent call
    metrics = {"latency_ms": 100, "total_tokens": 10, "estimated_cost_usd": 0.001}
    return {
        "output": "response",
        "trajectory": ["tool_a"],
        "environment_state": [EnvironmentState(name="metrics", state=metrics)],
        "metadata": metrics,
    }

results = run_all_layers(task_fn=agent_task)
print(f"All passed: {results['all_passed']}")
# Layers: layer_1 (tool usage), layer_2 (reasoning), layer_3 (output quality), domain
```

### Config-driven evaluation

```python
from agentic_evaluation import load_config, run_all_layers, TestCaseRegistry

cfg = load_config("eval_config.yaml")
registry = TestCaseRegistry.from_config(cfg.test_cases)
results = run_all_layers(task_fn=agent_task, registry=registry)
```

### Multi-trial for production reliability

```python
results = run_all_layers(task_fn=agent_task, num_trials=5)
# Gate on pass^k: all trials must pass
for layer in ["layer_1", "layer_2", "layer_3", "domain"]:
    info = results[layer]
    print(f"{layer}: pass@5={info['pass_at_k']}, pass^5={info['pass_all_k']}")
```

### Scaffold a new project

```bash
agentic-eval init --name "my-agent" --tools "search,analyze"
# Creates eval_config.yaml and a task-function stub
```

See [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) for the complete API reference and usage patterns.

## Security

This implementation follows AWS security recommended practices:

- **Avoids hardcoded secrets**: Uses AWS Secrets Manager and IAM roles
- **Least privilege IAM**: Granular permissions for each component
- **Encryption at rest**: S3 uses SSE-S3; DynamoDB, Lambda settings, logs,
  alerts, and the evaluation secret use KMS keys
- **Encryption in transit**: Amazon Bedrock API TLS 1.2+ for all API calls
- **Authenticated managed networking**: AgentCore Runtime uses managed public
  network mode with IAM or Cognito authorization; the endpoint is not anonymous
- **Security scanning**: Ruff, Bandit, pip-audit, detect-secrets, cfn-lint, and Checkov

See [docs/SECURITY.md](docs/SECURITY.md) for complete security implementation.

## Troubleshooting

Common issues and solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| Evaluation timeouts | Lambda configured with 3-minute timeout | Increase Lambda timeout to 10 minutes |
| Embedding dimension mismatch | Using Amazon Titan v1 instead of v2 | Verify `amazon.titan-embed-text-v2:0` model |
| Shadow mode degradation | Data drift or index loading issues | Check CloudWatch logs for pipeline failures |
| High evaluation costs | Sampling rate too high | Start with 1-3% sampling, increase gradually |
| CDK deployment failures | Missing Amazon Bedrock model access | Request model access in AWS Console |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for 20+ detailed troubleshooting scenarios with CloudWatch Insights queries and debug commands.

## Performance

**Latency Targets** (design targets under typical conditions):
- Response latency P50: <2s
- Response latency P99: <10s
- Tool selection: <50ms
- Semantic search (LanceDB): <100ms

**Throughput:**
- Designed to support up to 1,500+ concurrent users under typical conditions
- Can handle up to 50,000+ daily queries in testing
- Typically processes 1,500-2,500 vehicles per auction cycle

**Cost Optimization:**
- Local LanceDB materialization: avoids an Amazon S3 round trip for each vector query and reuses one table per Runtime process
- Blue-green deployment: Designed for zero-downtime data refresh
- Evaluation sampling: 1-5% production traffic (adjustable)

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for benchmarking methodology and tuning parameters.

## Cost Estimate

**Cost:** This sample uses AWS services. For pricing details, see the pricing page for each service. To avoid ongoing charges, delete the resources when you finish testing (see [Cleaning Up](#cleaning-up)).

Illustrative monthly costs at a production-scale workload (1,500 concurrent users, 50K daily queries):

| Component | Monthly Cost (USD) |
|-----------|-------------------|
| Claude Sonnet 4.6 from Anthropic (via Amazon Bedrock) | $600-900 |
| Amazon Titan Text Embeddings V2 | $100-150 |
| AWS Lambda (data pipeline) | $50-75 |
| Amazon S3 (LanceDB source rows and vectors) | $20-30 |
| Amazon EventBridge | $5-10 |
| CloudWatch Logs and Metrics | $50-100 |
| Four customer-managed AWS KMS keys | About $4 plus request charges |
| **Total** | **About $829-1,269, plus omitted usage-dependent services** |

**Dev/Test Environment**: Approximately $55-105/month with minimal traffic and
shorter log retention, plus usage-dependent model and AgentCore requests.

## Cleaning Up

You are responsible for AWS charges. Teardown is intentionally not a one-line
operation because the data and access-log buckets are retained in every
environment.

Before mutation:

1. Select an explicit profile, expected 12-digit account, and `eu-west-1`.
2. Inventory CloudFormation and billable resources, including retained S3
   objects and versions, ECR digests, logs, secrets, evaluation configs, and
   AgentCore resources.
3. Create a retention and backup manifest. Classify every stateful resource as
   retain, back up, migrate, delete, or unknown; resolve every `unknown`.
4. Review the destroy change and recurring-cost impact.
5. Obtain approval for the exact stack deletion set and separate approval for
   every retained-data deletion.

Immediately before an approved stack deletion, verify identity again:

```bash
aws sts get-caller-identity \
  --profile <profile> \
  --region eu-west-1
cd examples/vehicle-auction-agent/cdk
AWS_ACCOUNT_ID=<12-digit-account-id> \
AWS_REGION=eu-west-1 \
uv run cdk destroy --all \
  --profile <profile> \
  -c environment=<environment>
```

Do not manually delete retained buckets, object versions, tables, logs, secrets,
users, datasets, or ECR digests unless they are explicitly listed in the
approved retention manifest.

```bash
# Independently inventory residuals after CloudFormation finishes.
aws cloudformation list-stacks \
  --profile <profile> --region eu-west-1
aws s3api list-buckets \
  --profile <profile> --region eu-west-1
aws logs describe-log-groups \
  --log-group-name-prefix /aws/ \
  --profile <profile> --region eu-west-1
aws dynamodb list-tables \
  --profile <profile> --region eu-west-1
aws ecr describe-repositories \
  --profile <profile> --region eu-west-1
aws bedrock-agentcore-control list-agent-runtimes \
  --profile <profile> --region eu-west-1
```

A deleted stack is not evidence that retained or orphaned resources stopped
incurring cost.

## Contributing

Contributions are welcome! To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Key areas for contribution:**
- Additional custom evaluators (compliance, cost, latency)
- Multi-region deployment patterns
- Alternative data sources (Snowflake, Redshift, DynamoDB)
- Enhanced monitoring dashboards
- Performance benchmarking scripts

## Resources

### Documentation
- [Strands Agents SDK](https://strandsagents.com/)
- [strands-agents-evals GitHub](https://github.com/strands-agents/evals)
- [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
- [Amazon Bedrock AgentCore Evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html)
- [AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)
- [AgentCore Starter Toolkit](https://aws.github.io/bedrock-agentcore-starter-toolkit/)

### Blog Posts
- [Evaluating production AI agents on AWS with Strands and AgentCore](https://aws.amazon.com/blogs/machine-learning/evaluating-production-ai-agents-on-aws-with-strands-and-agentcore/)
- [Operationalize generative AI workloads with Amazon Bedrock, Part 1: GenAIOps](https://aws.amazon.com/blogs/machine-learning/operationalize-generative-ai-workloads-and-scale-to-hundreds-of-use-cases-with-amazon-bedrock-part-1-genaiops/)

## Production security recommendations

This repository is an educational AWS sample. The reference deployment already
implements several controls (AWS WAF with rate-based rules, API Gateway stage
throttling, Amazon Bedrock Guardrails, encryption at rest, least-privilege
IAM with confused-deputy guards, and ECR scan-on-push). Before using this code for
production or any real workload, we recommend the additional hardening below. Treat
these as fix-before-production items.

### Agent endpoint
- **Amazon Bedrock Guardrails**: keep them enabled and fail-closed. The sample
  configures HIGH prompt-attack and misconduct filters plus PII redaction
  (EMAIL, PHONE). Bid placement is structurally unavailable because the runtime
  exposes no bid-write tool; the system prompt gives an explicit capability
  refusal without misclassifying benign bidding questions as unsafe. For
  production, extend PII entities and topic controls to match your data and
  actual tool capabilities.
- **AWS WAF and API Gateway**: keep WAF rate-based rules and API Gateway
  rate-limiting/quotas in front of every public endpoint. Note that WAF cannot front
  the AgentCore Runtime data-plane directly; gate browser/API traffic through
  CloudFront or API Gateway.
- **Authentication**: the default IAM deployment is deliberately single-tenant
  and ignores caller-selected identity. Cognito mode derives the dealer from a
  verified JWT claim. In both modes, the server injects that identity into the
  only dealer-profile tool, so the model cannot list dealers or select another
  dealer ID.

### Secrets and supply chain
- **Secrets management**: when wiring a real data source (the sample mocks
  BigQuery via `MOCK_BIGQUERY=true`), store any GCP or third-party credentials in
  **AWS Secrets Manager** with automatic rotation and least-privilege IAM
  conditions. Do not embed credentials in code or environment variables.
- **Container image integrity**: ECR scan-on-push is enabled in the sample. For
  production, add image scanning in CI (e.g. Trivy) and consider **AWS Signer** for
  container image signing with verification at runtime pull.

### Anti-scraping and model-behavior extraction
A determined caller can send many structured queries to reverse-engineer the
agent's tools, ranking logic, and prompts ("model scraping" / "prompt extraction").
The API Gateway throughput limit alone does not distinguish heavy legitimate use
from adversarial probing. For production, layer on:

1. **Per-identity rate limiting (highest priority)**: enforce per-API-key or
   per-Cognito-user quotas (e.g. 50 req/min, 500 req/hour) via API Gateway usage
   plans with per-key caps, or a DynamoDB-backed counter in a Lambda authorizer.
2. **Probing-pattern detection**: log query sequences per identity and alert/block
   on high volume + low result diversity (single-parameter sweeps, sequential ID
   probing, identical queries from rotating IPs).
3. **Response minimization**: normal runtime responses contain only the public
   result, model identifier, and session ID. Trajectories, tool results, usage,
   and LanceDB freshness are returned only when the caller presents the
   Secrets Manager-backed evaluation token.
4. **Output-side Guardrails**: configure topic avoidance so the agent refuses to
   explain its own architecture, tools, or prompts.
5. **Authentication requirement**: retain IAM SigV4 or Cognito authorization;
   never add an anonymous invoke path. For shared deployments, put an
   identity-aware quota layer in front of the Runtime.

The next production control is an identity-aware quota layer. Response
minimization and authenticated invocation are already implemented.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

## Authors

- **Amit Deol** - Senior Prototyping Architect, AWS Prototyping and Cloud Engineering (PACE)
- **Hin Yee Liu** - Senior Prototype Engagement Manager, AWS

## Acknowledgments

- The industry collaborator whose real-world use case inspired this reference deployment
- The **Strands Agents** open-source project for the Strands Agents SDK and evaluation framework
- **Amazon Bedrock AgentCore** team for native evaluation support

## Conclusion

This reference implementation demonstrates production evaluation patterns for AI agents on AWS. The three-layer framework (tool usage, reasoning, and output quality) plus domain-specific checks provides quality gates across the full agent lifecycle. Get started with the SDK quickstart in under 60 seconds, or deploy the full reference architecture to see the complete pipeline in action.
