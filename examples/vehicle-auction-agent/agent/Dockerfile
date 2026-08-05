# syntax=docker/dockerfile:1.6
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AgentCore Runtime requires ARM64 / linux containers.
# Pinned to the reviewed Debian 12 multi-platform manifest published with the
# current Python patch release. Combined with the explicit ARM64 platform, this
# resolves one immutable base image.
FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.14.6-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY . .

# Run as non-root for least-privilege container security. The Debian slim base
# includes an essential Perl interpreter for package-management scripts, but
# this Python-only runtime does not use it and Debian has no patched package for
# the current critical advisories. Create the user first, then remove that
# otherwise unreachable interpreter from the final runtime attack surface.
RUN adduser --disabled-password --no-create-home appuser \
    && dpkg --purge --force-depends --force-remove-essential perl-base \
    && ! command -v perl
USER appuser

# AgentCore HTTP protocol contract: listen on 0.0.0.0:8080, expose
# /invocations and /ping. BedrockAgentCoreApp.run() handles both.
EXPOSE 8080

# Probe the AgentCore /ping endpoint so orchestrators can detect readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/ping', timeout=4).status==200 else 1)"

CMD ["python", "app.py"]
