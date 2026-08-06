# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Per-conversation state shared by the adapters.

A multi-turn test case is expanded into one :class:`~strands_evals.Case` per
turn, all carrying the same ``conversation_id``. Every adapter therefore needs
the same lifecycle: create something on a conversation's first turn (a runtime
session id, an ``Agent``), reuse it for the following turns, and drop it after
the last one. That lifecycle lives here so the adapters agree on it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any


def conversation_key(metadata: Mapping[str, Any]) -> str | None:
    """Identify the conversation a case belongs to.

    Args:
        metadata: The case's metadata.

    Returns:
        A key unique to the conversation *within this evaluation run*, or None
        for a single-turn case. The run id is part of the key so concurrent
        trials of the same case never share state.
    """
    conversation_id = metadata.get("conversation_id")
    if not conversation_id:
        return None
    run_id = metadata.get("evaluation_run_id", "default")
    return f"{run_id}:{conversation_id}"


def is_last_turn(metadata: Mapping[str, Any]) -> bool:
    """Report whether a case is the final turn of its conversation.

    Args:
        metadata: The case's metadata.

    Returns:
        True when this is the last turn, so conversation state can be released.
    """
    return metadata.get("turn_index") == metadata.get("turn_count")


class ConversationScope[T]:
    """State created on a conversation's first turn and dropped after its last.

    Access is locked because ``strands_evals`` may evaluate cases concurrently.

    Args:
        factory: Builds a fresh value. Called once per conversation, and once
            per call for single-turn cases (which are never stored).
    """

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._entries: dict[str, T] = {}
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        """The lock guarding this scope, for callers that must extend it.

        The local Strands adapter mutates a shared agent's message history and
        has to hold the same lock across the whole invocation, not just the
        lookup.
        """
        return self._lock

    def acquire(self, key: str | None) -> T:
        """Get a conversation's value, creating it on first use.

        Args:
            key: A :func:`conversation_key` result. None yields a fresh,
                unstored value.

        Returns:
            The value for that conversation.
        """
        if key is None:
            return self._factory()
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                value = self._factory()
                self._entries[key] = value
            return value

    def release(self, key: str | None, metadata: Mapping[str, Any]) -> None:
        """Drop a conversation's value once its last turn has been evaluated.

        Args:
            key: A :func:`conversation_key` result. None is a no-op, since
                single-turn cases store nothing.
            metadata: The case's metadata, used to detect the final turn.
        """
        if key is not None and is_last_turn(metadata):
            with self._lock:
                self._entries.pop(key, None)
