# syntax=docker/dockerfile:1.6
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# AgentCore Runtime requires ARM64 / linux containers.
# Pinned to an exact patch tag: the public ECR mirror does not carry the
# rolling 3.14-slim tag (only specific patch releases like 3.14.4-slim).
FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.14.4-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root for least-privilege container security.
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# AgentCore HTTP protocol contract: listen on 0.0.0.0:8080, expose
# /invocations and /ping. BedrockAgentCoreApp.run() handles both.
EXPOSE 8080

# Probe the AgentCore /ping endpoint so orchestrators can detect readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/ping', timeout=4).status==200 else 1)"

CMD ["python", "app.py"]
