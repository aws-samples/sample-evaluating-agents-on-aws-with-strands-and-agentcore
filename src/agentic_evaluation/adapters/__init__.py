# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Reference task_fn adapters for popular agent frameworks.

Each adapter exposes a ``make_task_fn(...)`` that returns a callable
compatible with :func:`agentic_evaluation.run_experiment.run_all_layers`.

Available adapters:

- :mod:`agentic_evaluation.adapters.strands_local` — local Strands ``Agent``
- :mod:`agentic_evaluation.adapters.agentcore` — deployed Bedrock AgentCore runtime
- :mod:`agentic_evaluation.adapters.http` — generic HTTP endpoint (LangChain, CrewAI,
  custom FastAPI, etc.)

Most users will write a thin adapter for their framework following the same
contract: take an agent handle in the closure, return a function ``(case)
-> {"output", "trajectory", "metadata"}``.
"""
