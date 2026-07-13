# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for agent/utils/log_redactor.py.

Redaction must work in the real runtime topology: the app logs through a module
child logger (``logging.getLogger(__name__)``) while the filter is installed at
the root. A logger-level filter would not see child-propagated records, so the
filter must live on the root handlers.
"""

import importlib
import io
import logging

import pytest

from utils import log_redactor


@pytest.fixture
def root_with_stream(monkeypatch):
    """Fresh root logger with a single stream handler, restored after the test."""
    monkeypatch.setenv("LOG_REDACTION_ENABLED", "true")
    importlib.reload(log_redactor)

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_filters = root.filters[:]
    saved_level = root.level

    for h in saved_handlers:
        root.removeHandler(h)
    for f in saved_filters:
        root.removeFilter(f)

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    yield buf

    root.removeHandler(handler)
    for f in root.filters[:]:
        root.removeFilter(f)
    for h in saved_handlers:
        root.addHandler(h)
    for f in saved_filters:
        root.addFilter(f)
    root.setLevel(saved_level)


def test_redacts_actor_and_session_via_child_logger(root_with_stream):
    """Mirrors app.py:640 — a child logger emitting the actual log line."""
    log_redactor.install_redaction_filter()

    logging.getLogger("agent.app").info(
        "Processing: actor=%s, session=%s", "DEALER123", "SESS999"
    )

    out = root_with_stream.getvalue()
    assert "DEALER123" not in out
    assert "SESS999" not in out
    assert "actor=[REDACTED]" in out
    assert "session=[REDACTED]" in out


def test_redacts_dealer_context_prompt_form(root_with_stream):
    """The interpolated dealer-context string app.py:650 builds is also redacted."""
    log_redactor.install_redaction_filter()

    logging.getLogger("agent.app").info("[Dealer context: actor=%s]", "SECRET")

    out = root_with_stream.getvalue()
    assert "SECRET" not in out
    assert "actor=[REDACTED]" in out


def test_install_is_idempotent(root_with_stream):
    """Calling install twice must not stack duplicate filters on a handler."""
    log_redactor.install_redaction_filter()
    log_redactor.install_redaction_filter()

    handler = logging.getLogger().handlers[0]
    redaction_filters = [
        f for f in handler.filters if isinstance(f, log_redactor.SensitiveFieldFilter)
    ]
    assert len(redaction_filters) == 1


def test_disabled_via_env_passes_through(monkeypatch):
    """With redaction disabled, the filter is a no-op and nothing is installed."""
    monkeypatch.setenv("LOG_REDACTION_ENABLED", "false")
    importlib.reload(log_redactor)

    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    for h in saved_handlers:
        root.removeHandler(h)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    try:
        log_redactor.install_redaction_filter()
        assert not any(
            isinstance(f, log_redactor.SensitiveFieldFilter) for f in handler.filters
        )
        logging.getLogger("agent.app").info("Processing: actor=%s", "PLAINTEXT")
        assert "PLAINTEXT" in buf.getvalue()
    finally:
        root.removeHandler(handler)
        for h in saved_handlers:
            root.addHandler(h)
        # Restore module default for later tests.
        monkeypatch.setenv("LOG_REDACTION_ENABLED", "true")
        importlib.reload(log_redactor)
