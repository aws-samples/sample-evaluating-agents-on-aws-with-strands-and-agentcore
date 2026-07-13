# SDK Quickstart

## Introduction

This quickstart runs the complete four-layer evaluation pipeline locally in under 60 seconds. It needs no AWS account, no LLM calls, and no credentials. It demonstrates how the agentic-evaluation SDK scores an agent across tool usage, reasoning, output quality, and domain checks using mock data. Use it to see the pipeline end to end before wiring in a real agent or the full AWS reference architecture.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Git (to clone the repository)

## Run the demo

```bash
# From the repository root:
cd evaluating-agents-on-aws-with-strands-and-agentcore
uv sync
uv run python quickstart/run_demo.py
```

You should see all layers pass.

You should see output similar to:

```text
Layer 1 (Tool Usage): PASSED
Layer 2 (Reasoning): PASSED
Layer 3 (Output Quality): PASSED
Domain Layer: PASSED

All evaluation layers: PASSED
```

After all layers pass, you have successfully run the four-layer evaluation pipeline. You can now modify the mock agent in `quickstart/run_demo.py` or proceed to plug in your own agent as described in the following section.

## Plug in your own agent

Three lines:

```python
# 1. Replace the mock with a real adapter
from agentic_evaluation.adapters.http import make_task_fn
task_fn = make_task_fn(endpoint_url="https://your-agent.example.com/invoke")

# 2. Point the runner at it
from agentic_evaluation import run_all_layers
results = run_all_layers(task_fn=task_fn)
```

That's it. Layer 1 (tool selection / order), Layer 2 (helpfulness / reasoning),
Layer 3 (output quality / goal success), and the domain layer (latency, cost,
safety, freshness, scoping) all run against your agent.

## What the demo evaluates

| Layer | What it checks | Backed by |
|-------|----------------|-----------|
| Layer 1 | Did the agent pick the right tools, in roughly the right order? | Deterministic graders |
| Layer 2 | Was the reasoning helpful? | LLM judge (skipped here, `noop`) |
| Layer 3 | Was the output relevant and complete? | LLM judge (skipped here, `noop`) |
| Domain  | Latency, cost, safety, data freshness, schema scoping | Deterministic graders |

To switch on real LLM judging, set `judge_backend: "strands"` in
`eval_config.yaml` and provide Amazon Bedrock credentials.

## Troubleshooting

**`uv` not found**: Install `uv` with `pip install uv` or follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

**Python version mismatch**: The SDK requires Python 3.14+. Verify your version with `python --version`. Use `uv python install 3.14` to install a compatible version.

**`uv sync` fails with dependency errors**: Delete the `.venv` directory and run `uv sync` again to rebuild the environment from scratch.

**Demo output shows failures**: Verify you are running from the repository root (`evaluating-agents-on-aws-with-strands-and-agentcore`) and that `uv sync` completed without errors. If a specific layer fails, check that the corresponding test case file under `quickstart/` has not been modified.

**Import errors for `agentic_evaluation`**: Run `uv run python -c "import agentic_evaluation; print('ok')"` to confirm the package is installed in the `uv` environment.

## Cleaning Up

To remove resources created during the quickstart:

```bash
# Remove the virtual environment
rm -rf .venv

# Remove any generated evaluation result files
rm -f evaluation_results.json
```

If you deployed AWS resources using the full reference architecture, see the main README [Cleaning Up](#-cleaning-up) section for infrastructure teardown.

## Conclusion

You have run a complete four-layer evaluation pipeline in under 60 seconds. To evaluate your own agent, replace the mock task_fn with a real adapter. See SDK_GUIDE.md for framework-specific examples. For production deployments, see the main README for the full reference architecture on AWS.
