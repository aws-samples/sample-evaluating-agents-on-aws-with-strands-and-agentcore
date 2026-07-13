# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Generic HTTP adapter.

Works with any agent exposed over HTTP — LangChain, CrewAI, custom FastAPI,
etc. Customize the JSON keys to match your endpoint's request/response shape.

Usage::

    from agentic_evaluation.adapters.http import make_task_fn
    from agentic_evaluation import run_all_layers

    results = run_all_layers(
        task_fn=make_task_fn(
            endpoint_url="https://api.example.com/agent/invoke",
            headers={"Authorization": "Bearer <token>"},
        )
    )
"""

from __future__ import annotations

import time
from typing import Callable

import requests
from strands_evals import Case

from agentic_evaluation.exceptions import TaskFnError
from agentic_evaluation.types import TaskFnResult


def make_task_fn(
    endpoint_url: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 30,
    input_key: str = "prompt",
    output_key: str = "output",
    trajectory_key: str = "trajectory",
) -> Callable[[Case], TaskFnResult]:
    """Wrap a JSON HTTP endpoint as a ``task_fn``.

    Args:
        endpoint_url: Full URL of the agent endpoint.
        headers: Optional HTTP headers (e.g. auth).
        timeout_seconds: Request timeout.
        input_key: Request JSON key for the prompt.
        output_key: Response JSON key for the output text.
        trajectory_key: Response JSON key for the tool-name list.
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    if headers:
        session.headers.update(headers)

    def task_fn(case: Case) -> TaskFnResult:
        start = time.perf_counter()

        resp = session.post(
            endpoint_url,
            json={input_key: case.input},
            timeout=timeout_seconds,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise TaskFnError(f"Agent returned HTTP {resp.status_code}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        data = resp.json()

        return {
            "output": data.get(output_key, ""),
            "trajectory": data.get(trajectory_key, []),
            "metadata": {
                "latency_ms": elapsed_ms,
                "status_code": resp.status_code,
            },
        }

    return task_fn
