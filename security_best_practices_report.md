# Architecture, Code Quality, and Security Review

**Review date:** 2026-07-22

## Executive Summary

No unresolved critical or high-severity finding remains in the reviewed source,
deployment scripts, synthesized infrastructure, or diagrams. The review
retained LanceDB and hardened its publication, refresh, filtering, and cache
lifecycle rather than replacing it.

The dev deployment was updated in the explicitly verified `eu-west-1` target.
AgentCore runtime version 17 is `READY` on immutable image digest
`sha256:05da559f99691b9e8d8ac8aa2262d1e3f432a3925c6e9a8937e7c79ec074a8a6`.
The final live evaluation passed every layer, all five stacks are in sync, and
the final CDK diff is zero. One vendor medium-severity base-image finding
remains documented below; no vendor-fixed package is currently available.

## Critical Severity

No open findings.

## High Severity

No open findings.

## Medium Severity

### M-01: Vendor util-linux vulnerability has no available fix

**Impact:** A locally privileged attacker able to present a crafted block-device
image could trigger a `libblkid` use-after-free read, causing limited
information disclosure or denial of service.

Amazon ECR reports `CVE-2026-13595` (CVSS 5.3) in Debian's
`util-linux 2.38.1-5+deb12u3` and provides no fixed version. The runtime is a
managed AgentCore container, runs as non-root, and does not mount or probe
caller-controlled block devices, so the vulnerable path is not reachable
through the application interface. Keep the pinned base image under scan and
rebuild when Debian publishes a fixed package.

## Resolved Findings

### R-01: Trusted dealer identity and tenant isolation

The runtime rejects caller-selected identity in the request body, derives the
dealer from the configured trust boundary, and exposes a zero-argument,
identity-scoped dealer-profile tool
(`examples/vehicle-auction-agent/agent/app.py:302`,
`examples/vehicle-auction-agent/agent/app.py:355`). IAM mode is explicitly
single-tenant; Cognito mode uses a verified dealer claim. Negative
cross-identity tests are included.

### R-02: AWS account, region, cost, and approval safety

Every repository mutation path now uses the shared `eu-west-1` allowlist,
explicit profile, expected 12-digit account, STS comparison, fresh
pre-mutation verification, and exact confirmation phrase
(`scripts/aws_safety.py:13`, `scripts/aws_safety.py:17`,
`scripts/aws_safety.py:43`, `scripts/aws_safety.py:61`). Deployment also
resolves an immutable ECR digest and runs CDK diff before approval.

### R-03: Evaluation correctness and fail-closed telemetry

Each case executes once per trial and all applicable layers evaluate a shared
run artifact (`src/agentic_evaluation/run_experiment.py:416`,
`src/agentic_evaluation/run_experiment.py:547`). Declared-layer routing,
multi-turn execution, tool-parameter grading, session judges, and
missing-telemetry failures are implemented and covered by focused tests.
Shared Strands agents restore their original history; factory-created agents
preserve only the intended multi-turn conversation
(`src/agentic_evaluation/adapters/strands_local.py:90`).

The live release test now asserts `all_passed` rather than returning a false
green on threshold failures
(`tests/integration/test_real_agent_eval.py:278`). Multi-turn refinements must
repeat the complete active filter set
(`examples/vehicle-auction-agent/agent/app.py:1194`). Runtime and judge clients
use bounded connection/read timeouts and retries
(`src/agentic_evaluation/adapters/agentcore.py:152`,
`src/agentic_evaluation/judges.py:86`), and case/layer progress is logged.

Ordinary searches now restore short-term session context without injecting
cross-session preferences. Long-term AgentCore Memory retrieval is enabled only
when the current request explicitly asks for remembered preferences or facts
(`examples/vehicle-auction-agent/agent/app.py:1279`,
`examples/vehicle-auction-agent/agent/app.py:1307`). This removed stale search
filters from unrelated responses while preserving LanceDB and AgentCore Memory.
The location-aware quality assertion also accepts truthful zero-nearby results
instead of requiring fabricated inventory (`eval_config.yaml:158`).

### R-04: LanceDB publication and warm-runtime integrity

Ingestion rejects empty, duplicate, malformed, non-finite, wrong-dimension, and
low-success candidates before promotion
(`examples/vehicle-auction-agent/lambda/functions/data_ingestion/handler.py:475`).
It writes an immutable snapshot, computes SHA-256, and promotes only the
manifest (`examples/vehicle-auction-agent/lambda/functions/data_ingestion/handler.py:513`).
The runtime validates the manifest and checksum, periodically refreshes under a
lock, retains the last-known-good table after failures, bounds cache
generations, and pushes scalar filters into LanceDB
(`examples/vehicle-auction-agent/agent/app.py:536`,
`examples/vehicle-auction-agent/agent/app.py:577`,
`examples/vehicle-auction-agent/agent/app.py:643`,
`examples/vehicle-auction-agent/agent/app.py:1000`).

### R-05: Telemetry disclosure and log handling

Normal callers receive a minimized response. Detailed trajectories, tool
results, usage, and freshness require a constant-time checked,
Secrets Manager-backed evaluation token. Logs redact actor, session, and dealer
context.

### R-06: Infrastructure encryption and IAM policy scope

Four rotating, retained customer-managed KMS keys protect DynamoDB, Lambda
settings, CloudWatch logs, SNS alerts, and the evaluation secret. KMS actions
are explicit; CDK-generated `kms:*`, `kms:GenerateDataKey*`, and
`kms:ReEncrypt*` are expanded before synthesis
(`examples/vehicle-auction-agent/cdk/lib/security.py:55`,
`examples/vehicle-auction-agent/cdk/lib/security.py:106`). CloudWatch Logs
grants are constrained to the regional service principal and exact log-group
encryption contexts (`examples/vehicle-auction-agent/cdk/lib/security.py:70`).
The final five templates contain no wildcard IAM actions.
The app-wide `LambdaInvokeBoundary` aspect rejects Function URLs, non-service
principals, and invoke permissions without a scoped `SourceArn`
(`examples/vehicle-auction-agent/cdk/lib/security.py:58`). Live policies confirm
that EventBridge can invoke only the ingestion function through its named rule,
API Gateway can invoke only the dealer function through its named API routes,
and no project Lambda has a Function URL.

### R-07: Supply-chain and CI enforcement

The runtime image is selected by digest, deployable dependencies have a lock,
and known vulnerable transitive versions were updated. CI now runs the complete
offline suite with a 70% floor and pinned Bandit, pip-audit, detect-secrets,
CDK, cfn-lint, wildcard-action, and Checkov gates
(`.github/workflows/sdk-ci.yml:43`, `.github/workflows/sdk-ci.yml:77`).

### R-08: Architecture diagrams

All five Draw.io sources were rebuilt and exported with Draw.io Desktop
30.4.1. All 47 connectors have source and target anchors, orthogonal routing,
square corners, filled block arrowheads, and no curved edges. The diagrams now
match the implemented identity boundary, LanceDB manifest flow, shared
evaluation artifact, and manual versus automated deployment stages.

## Low Severity and Operational Follow-Up

### L-01: Coverage is strong on risk-bearing paths but incomplete overall

The offline suite reports 74% statement coverage. The HTTP adapter, generation
utility, example evaluators, and several CLI error paths remain lightly tested
or untested. The new CI floor is 70%; raise it as these secondary paths gain
coverage.

### L-02: Upstream deprecation warnings remain

The passing suite emits deprecations from `bedrock-agentcore`,
`strands-agents-evals`, and `strands-agents` under Python 3.14. They do not
originate in repository code, but dependency upgrades should be tested before
Python 3.16 removes the deprecated asyncio API.

### L-03: New CDK feature flags require a separate migration review

CDK 2.1131.0 reports 35 newer recommended feature flags as unconfigured. Most
do not affect constructs used here, but enabling flags globally can alter
logical IDs, policies, or replacement behavior. Review them as a dedicated
migration with synth/diff and replacement analysis rather than silently
changing deployed defaults.

### L-04: Alert topic has no subscribers

The `agent-eval-alerts-dev` SNS topic exists and alarms are attached, but it has
zero subscriptions. Add an owned notification target before relying on these
alarms for operational response.

### L-05: Managed runtime emits benign startup warnings

Each fresh AgentCore runtime log stream emits `Invalid HTTP request received`
before successful invocations. LanceDB also warns while creating a new
ephemeral table on a cold microVM. Neither warning has an accompanying
application exception, failed invocation, endpoint error, DLQ message, or
breaching alarm. Track them during AgentCore/LanceDB upgrades; do not suppress
them in application code without an upstream fix.

### L-06: Production values remain adopter-owned

The production CORS domain is intentionally an `example.com` placeholder.
CloudTrail, GuardDuty, production identity-aware quotas, online-evaluation
configuration, custom quality alarms, and alert subscriptions remain
account-level or operator-managed prerequisites documented in the security
guide.

## Validation Performed

- Offline pytest: **175 passed**, **39 deployed deselected**, **74% coverage**.
- Ruff lint and formatting: passed.
- Python `compileall` and `git diff --check`: passed.
- Bandit: no medium/high findings.
- pip-audit, root frozen export: no known vulnerabilities.
- pip-audit, deployable agent lock: no known vulnerabilities.
- detect-secrets, filtered source/config scan: no findings.
- AWS CDK: all five stacks synthesized.
- cfn-lint: all five templates passed; only documented CDK `W3005` ignored.
- Checkov: **179 passed, 0 failed**.
- IAM action audit: no wildcard actions.
- Draw.io: five valid XML sources, 47/47 conforming connectors, five RGB PNGs.
- ECR scan: 0 critical, 0 high, 1 medium (`CVE-2026-13595`, no fix available).
- Deployment validator: **8/8 passed**.
- Deployed infrastructure/data/API tests: **26 passed, 1 intentionally skipped**.
- CloudFormation: all five stacks `UPDATE_COMPLETE` and `IN_SYNC`; final CDK
  diff: zero.
- Lambda exposure: 0 Function URLs; only scoped EventBridge/API Gateway
  service-principal invoke permissions with exact source ARNs.
- CloudWatch: all seven project alarms `OK`; ingestion DLQ empty; no application
  errors or exceptions after the v17 deployment.
- Live `hp_001`: correct three-tool trajectory and **10,715 tokens**, below the
  unchanged 15,000-token gate.
- Live full pipeline: Layer 1 `1.00/1.00/1.00`; Layer 2 `0.93/1.00`; Layer 3
  `0.96/1.00`; domain `1.00/1.00/0.99/1.00`; all layers passed.

## Deployment State

The approved dev update is deployed and validated in `eu-west-1`. LanceDB
remains active at `lancedb/manifest.json`, Guardrail version 2 remains attached,
and runtime version 17 was updated in place without replacing stateful
resources. The deploying operator selected an explicit profile, the expected
12-digit account, and `eu-west-1`, and `scripts/aws_safety.py` verified all
three against STS before the mutation. Concrete account identifiers are
deliberately omitted from this repository.
