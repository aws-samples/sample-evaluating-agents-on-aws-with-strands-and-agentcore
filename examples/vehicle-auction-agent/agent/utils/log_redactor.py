# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Minimal log redaction for fields NOT covered by Bedrock Guardrails.

Bedrock Guardrails already handles PII anonymization (EMAIL, PHONE) in model
I/O. This filter covers the gap: application-level logger.info() calls that
log dealer_id, session_id, and raw query content before those values reach
the model. Without this, CloudWatch logs expose business-intent data.

Design choice: logging.Filter is the lightest-weight integration point.
No dependencies, no latency impact, no state. Disable via env for local dev.
"""

import logging
import os
import re

REDACTION_ENABLED = os.environ.get("LOG_REDACTION_ENABLED", "true").lower() == "true"

# Patterns for application-level sensitive data that bypasses Bedrock Guardrails.
_ACTOR_ID_PATTERN = re.compile(r"(actor=)[A-Za-z0-9_\-]{1,64}")
_SESSION_ID_PATTERN = re.compile(r"(session=)[A-Za-z0-9_\-]+")

_REDACTED = "[REDACTED]"


class SensitiveFieldFilter(logging.Filter):
    """Redacts dealer/session identifiers from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact identifiers in place, always keeping the record.

        Args:
            record: The record about to be emitted. Its message is rewritten in
                place; lazy ``%``-style args are interpolated first so patterns
                can match values the caller passed separately.

        Returns:
            Always ``True`` — this filter rewrites records, it never drops them.
        """
        if not REDACTION_ENABLED:
            return True
        if record.args:
            try:
                record.msg = record.msg % record.args
                record.args = None
            except (TypeError, ValueError):
                # A caller's args do not match its format string. Leave both
                # alone: the stdlib handler will surface the same error, and
                # logging from inside a filter would recurse.
                pass
        msg = str(record.msg)
        msg = _ACTOR_ID_PATTERN.sub(r"\g<1>" + _REDACTED, msg)
        msg = _SESSION_ID_PATTERN.sub(r"\g<1>" + _REDACTED, msg)
        record.msg = msg
        return True


def install_redaction_filter() -> None:
    """Attach redaction filter to the root logger's handlers. Idempotent.

    A ``logging.Filter`` on a *logger* only sees records emitted directly to
    that logger, not records that propagate up from child loggers. The app logs
    via module child loggers (``logging.getLogger(__name__)``), so the filter
    must sit on the root *handlers*, which see every propagated record. The
    AgentCore/Lambda runtime configures the root handler during init, before
    this module is imported. If no handler exists yet, fall back to the root
    logger so records logged directly to root are still redacted.
    """
    if not REDACTION_ENABLED:
        return
    root = logging.getLogger()
    targets = root.handlers or [root]
    for target in targets:
        if not any(isinstance(f, SensitiveFieldFilter) for f in target.filters):
            target.addFilter(SensitiveFieldFilter())
