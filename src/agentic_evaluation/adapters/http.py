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
from collections.abc import Callable
from dataclasses import dataclass

import requests
from strands_evals import Case
from strands_evals.types.evaluation import EnvironmentState

from agentic_evaluation.exceptions import TaskFnError
from agentic_evaluation.types import TaskFnResult


@dataclass(frozen=True, slots=True)
class JsonKeys:
    """The JSON field names an endpoint uses, mapped to the ``task_fn`` contract.

    Defaults match the shape this SDK's own examples serve. Override only the
    fields your endpoint spells differently::

        make_task_fn(url, keys=JsonKeys(input="question", output="answer"))

    Attributes:
        input: Request key carrying the prompt.
        output: Response key carrying the output text.
        trajectory: Response key carrying the tool-name list.

    .. versionadded:: 0.4.0
    """

    input: str = "prompt"
    output: str = "output"
    trajectory: str = "trajectory"


DEFAULT_KEYS = JsonKeys()


def make_task_fn(
    endpoint_url: str,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 30,
    *,
    keys: JsonKeys = DEFAULT_KEYS,
) -> Callable[[Case], TaskFnResult]:
    """Wrap a JSON HTTP endpoint as a ``task_fn``.

    Args:
        endpoint_url: Full URL of the agent endpoint.
        headers: Optional HTTP headers (e.g. auth).
        timeout_seconds: Request timeout.
        keys: The endpoint's JSON field names. See :class:`JsonKeys`.

    Returns:
        A ``task_fn`` that POSTs each case to the endpoint.

    .. versionchanged:: 0.4.0
        ``input_key`` / ``output_key`` / ``trajectory_key`` are replaced by the
        ``keys`` value object, so mapping a new field extends :class:`JsonKeys`
        instead of growing this signature.
    """
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    if headers:
        session.headers.update(headers)

    def task_fn(case: Case) -> TaskFnResult:
        start = time.perf_counter()

        resp = session.post(
            endpoint_url,
            json={keys.input: case.input},
            timeout=timeout_seconds,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise TaskFnError(f"Agent returned HTTP {resp.status_code}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        data = resp.json()

        return {
            "output": data.get(keys.output, ""),
            "trajectory": data.get(keys.trajectory, []),
            "environment_state": [
                EnvironmentState(
                    name="metrics",
                    state={
                        "latency_ms": elapsed_ms,
                    },
                )
            ],
            "metadata": {
                "latency_ms": elapsed_ms,
                "status_code": resp.status_code,
            },
        }

    return task_fn
