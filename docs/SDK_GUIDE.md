# agentic-evaluation Software Development Kit (SDK) Guide

## Introduction

The agentic-evaluation SDK provides a framework-agnostic evaluation pipeline for AI agents. This guide shows how to integrate the SDK with agents built on different frameworks: Strands, Amazon Bedrock AgentCore, LangChain, CrewAI, OpenAI, or custom HTTP backends. Any agent you can wrap in a `task_fn(case)` callable that returns output, trajectory, and operational `EnvironmentState` can typically be evaluated. Whether you are evaluating tool selection, reasoning quality, or output correctness, the SDK offers a consistent interface across frameworks.

## Terminology

This SDK uses a layered model. The layer names are canonical (matching the
accompanying blog post) but each maps onto standard evaluation terminology, and
the SDK accepts/emits both:

| Canonical | Standard alias | Maps to (DeepEval / Ragas, popular agent-evaluation frameworks) |
|-----------|----------------|----------------------------|
| `layer_1` (Tool Usage) | `tool_correctness` | Tool Correctness / Tool Call Accuracy |
| `layer_2` (Reasoning) | `process_evaluation` | trajectory LLM-as-judge (process) |
| `layer_3` (Output Quality) | `outcome_evaluation` | Agent Goal Accuracy / Task Completion |
| `domain` | `operational_metrics` | latency / cost / safety / non-functional |

- `run_all_layers()` results are keyed by **both** names (the alias is the same
  object).
- `eval_config.yaml` `evaluation_layers` accepts either name.
- "Build time" (the CI gate) is **offline evaluation**; "production" sampling is
  **online evaluation**, the terms used across LangSmith, Braintrust, and Anthropic.

## Core contract

Every adapter ultimately produces a `task_fn` with this shape:

```python
from agentic_evaluation import TaskFnResult
from strands_evals.types.evaluation import EnvironmentState

def task_fn(case) -> TaskFnResult:
    metrics = {
        "latency_ms": 420,
        "total_tokens": 120,
        "estimated_cost_usd": 0.001,
    }
    return {
        "output": "<final agent response>",
        "trajectory": ["tool_a", "tool_b"],
        "environment_state": [EnvironmentState(name="metrics", state=metrics)],
        "metadata": metrics,  # Optional fallback for direct/static fixtures.
    }
```

`case` is an instance of `strands_evals.Case`, built from your
`agentic_evaluation.test_cases.TestCase` config entries by `build_cases_from_registry()`.
Read the query with `case.input`. The SDK gives the
runner the test case; your adapter is responsible for invoking the agent and
extracting the final output, tool trajectory, and measured operational state.
Live domain evaluators read the `metrics` environment state because
`strands_evals` does not propagate task-result metadata. The built-in keys are
`latency_ms` (int, milliseconds, read by `LatencyEvaluator`), `total_tokens` (int) and
`estimated_cost_usd` (float, read by `CostEvaluator`), and `last_refresh_time`
(ISO-8601 str, read by `DataFreshnessEvaluator`). Enabled evaluators fail closed
when their required measurements are absent.

## Prerequisites

- Python 3.14+
- Git
- [uv](https://docs.astral.sh/uv/) package manager (v0.4+)

## Install

### Run the quickstart

```bash
# Core SDK only (no LLM judge, no AWS):
pip install agentic-evaluation

# With Strands LLM-as-judge:
pip install "agentic-evaluation[strands]"

# With AgentCore runtime adapter:
pip install "agentic-evaluation[agentcore]"

# Everything:
pip install "agentic-evaluation[all]"
```

> `agentcore` here is the package extra identifier for Amazon Bedrock AgentCore integration. A Python "extra" is an optional dependency group installed via `pip install pkg[extra]`.

Verify the installation:
```bash
python -c "import agentic_evaluation; print('SDK installed')"
# or confirm the CLI is on PATH:
agentic-eval --help
```

## CLI

Use the command-line interface (CLI):

```bash
# Scaffold a new project:
agentic-eval init --name my-agent --tools "search,answer"

# Validate the YAML without running:
agentic-eval validate --config eval_config.yaml

# Run all layers against a task_fn:
agentic-eval run --config eval_config.yaml --task-fn my_pkg.tasks:run

# Restrict layers (for example, skip LLM judge):
agentic-eval run --config eval_config.yaml --task-fn my_pkg.tasks:run \
  --layers layer_1,domain --output results.json
```

## Strands Agents

```python
from strands import Agent
from agentic_evaluation.adapters.strands_local import make_task_fn

agent = Agent(model="anthropic.claude-sonnet-4-6", tools=[search, answer])
task_fn = make_task_fn(agent)
```

> **Note:** `anthropic.claude-sonnet-4-6` here is a Strands framework model identifier for Claude Sonnet 4.6 from Anthropic, accessed through Amazon Bedrock. When you call Amazon Bedrock APIs directly, use the full Bedrock model ID or a cross-region inference profile ID (such as `eu.anthropic.claude-sonnet-4-6`) instead.

## Amazon Bedrock AgentCore Runtime

```python
from agentic_evaluation.adapters.agentcore import make_task_fn

task_fn = make_task_fn(
    runtime_arn="arn:aws:bedrock-agentcore:...",
    region="eu-west-1",
)
```

The adapter defaults to a 5-second connection timeout, a 120-second read
timeout, and three standard retry attempts. Pass a `botocore.config.Config`
through `boto_client_config` when a workload needs different bounds.

## Plain HTTP / OpenAI / LangChain / CrewAI / custom

Frameworks that expose "send a prompt, get a response and the list of
tools called" can typically be integrated with this pattern. Use the HTTP adapter or write a `task_fn` directly:

```python
import time
from openai import OpenAI
from strands_evals.types.evaluation import EnvironmentState

client = OpenAI()

def task_fn(case):
    t0 = time.time()
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=case.input,
        tools=[...],
    )
    metrics = {"latency_ms": int((time.time() - t0) * 1000)}
    return {
        "output": resp.output_text,
        "trajectory": [step.tool_name for step in resp.tool_uses],
        "environment_state": [EnvironmentState(name="metrics", state=metrics)],
        "metadata": metrics,
    }
```

Populate `total_tokens` and `estimated_cost_usd` from the provider's measured
usage before enabling the cost evaluator.

You can adapt the same `task_fn` pattern for LangChain by calling `AgentExecutor.invoke`
and extracting the trajectory from the response. For CrewAI, call `Crew.kickoff` and
map the output format accordingly. The pattern typically works the same way for custom HTTP backends that follow the expected interface.

## Custom evaluators via plugins

Third parties can register evaluators by name and reference them in YAML:

```toml
# pyproject.toml of your evaluator package
[project.entry-points."agentic_evaluation.evaluators"]
my_pii_check = "my_pkg.evals:PIIEvaluator"
```

```yaml
# eval_config.yaml
plugin_evaluators:
  - name: my_pii_check
    params: {threshold: 0.95}
```

## Custom judge backend

The default LLM judge is `StrandsJudgeBackend`. Its Bedrock client uses the same
bounded timeout/retry defaults as the AgentCore adapter. Construct the backend
directly to override them:

```python
from agentic_evaluation import StrandsJudgeBackend, run_all_layers

judge = StrandsJudgeBackend(
    connect_timeout_seconds=5,
    read_timeout_seconds=120,
    max_attempts=3,
)
results = run_all_layers(task_fn, judge_backend=judge)
```

Use `NoOpJudgeBackend` to skip Layer 2/3 entirely, or implement your own
`JudgeBackend` Protocol:

```python
from agentic_evaluation import JudgeBackend, build_judge

class MyJudge:
    def layer2_evaluators(self, *, model, rubric, tool_descriptions): ...
    def layer3_evaluators(self, *, model, rubric): ...
```

Register it via the `agentic_evaluation.judges` entry point and reference it by name in
YAML (`judge_backend: my_judge`).

## Troubleshooting

**Installation verification fails**: verify that Python 3.14+ is installed and pip is current with `pip install --upgrade pip`.

**Framework adapter import errors**: install the matching extra, for example `pip install "agentic-evaluation[strands]"` for Strands or `pip install "agentic-evaluation[agentcore]"` for Amazon Bedrock AgentCore.

**Task function returns the wrong format**: return a dict with `output` (str), `trajectory` (list[str]), and `metadata` (dict).

For evaluation-specific issues, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

## Cleaning Up

After completing your evaluation work, remove artifacts to keep your environment tidy and avoid ongoing charges.

> **Cost note:** Amazon Bedrock AgentCore runtimes, Amazon CloudWatch Logs, and Amazon S3 objects incur charges while they remain active or stored. To avoid ongoing charges, delete them when you no longer need them.

1. **To remove the package from a shared environment**, run:
   ```bash
   pip uninstall agentic-evaluation
   ```

2. **Remove the virtual environment** (if you created a dedicated one):
   ```bash
   rm -rf .venv
   ```

3. **Remove generated configuration and result files**:
   ```bash
   rm -f eval_config.yaml results.json
   ```

4. **Inventory AWS resources** using the explicit profile, expected account, and
   region used for deployment:
   ```bash
   aws bedrock-agentcore-control list-agent-runtimes \
       --profile <profile> \
       --region eu-west-1
   aws logs describe-log-groups \
       --log-group-name-prefix /aws/bedrock-agentcore \
       --profile <profile> \
       --region eu-west-1
   ```

5. **For infrastructure teardown**, follow the retention, backup, approval, STS
   verification, and post-delete residual-inventory process in the main
   [Cleaning Up guide](../README.md#cleaning-up). Never treat a deleted runtime
   or stack as evidence that S3 data, object versions, logs, secrets, or ECR
   images were removed.

## Conclusion

The SDK's framework-agnostic design lets you evaluate agents regardless of implementation. Start with the quickstart to see the evaluation pipeline in action, then adapt the task_fn pattern to your framework. For advanced use cases, implement custom evaluators or judge backends using the plugin system.
