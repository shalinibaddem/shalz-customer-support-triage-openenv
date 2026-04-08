---
title: shalz-customer-support-triage-openenv
emoji: "📞"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Customer Support Triage OpenEnv

Customer Support Triage OpenEnv is a deterministic benchmark that simulates the work of a SaaS support specialist handling inbound tickets under routing, urgency, and policy constraints. The benchmark emphasizes operational support behavior rather than one-shot text classification.

## What It Tests

The environment evaluates whether an agent can:

- understand a noisy customer support request
- classify the issue correctly
- assign an operationally appropriate priority
- request missing information when policy requires it
- route or escalate to the correct internal team
- draft a safe customer response without overpromising
- finish with the correct disposition

## Environment Design

The agent receives a structured observation that includes:

- ticket details
- customer profile
- account context
- policy snippets
- allowed internal queues
- action history
- current ticket status
- remaining step budget

The action space is typed and multi-step:

- `classify_ticket`
- `set_priority`
- `request_customer_info`
- `route_ticket`
- `draft_response`
- `escalate_ticket`
- `resolve_ticket`

Rewards are shaped across the trajectory:

- positive reward for first correct subgoals such as classification, routing, or escalation
- smaller reward for safe response drafting
- penalties for repeated actions, invalid payloads, wrong routing, premature resolution, and forbidden claims
- terminal bonus derived from the deterministic rubric score

## Tasks

`billing_refund_easy`

- duplicate billing charge dispute
- expected to route to billing operations
- tests safe handling of refund language

`account_access_medium`

- account access issue with incomplete verification details
- expected to request required identity fields before closure
- tests avoidance of premature resolution

`enterprise_policy_hard`

- enterprise launch risk during compliance-related suspension
- expected to route to enterprise support and escalate to account management
- tests policy compliance under pressure and churn risk

## Project Structure

`customer_support_openenv/models.py`

- typed Pydantic schemas for observations, actions, rewards, rubrics, and internal state

`customer_support_openenv/tasks.py`

- benchmark tasks and deterministic expectations

`customer_support_openenv/graders.py`

- rubric scoring logic for full and partial credit

`customer_support_openenv/rewards.py`

- dense reward shaping and terminal bonus logic

`customer_support_openenv/environment.py`

- environment state machine with `reset()`, `step()`, and `state()`

`scripts/run_baseline.py`

- reproducible OpenAI baseline runner that saves per-task results to JSON

`scripts/smoke_test.py`

- quick local check using scripted trajectories

`app/space_app.py`

- FastAPI service exposing `/`, `/healthz`, `/reset`, `/step`, and `/state`

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the smoke test:

```bash
python scripts/smoke_test.py
```

Run tests:

```bash
pytest
```

Run the HTTP app:

```bash
uvicorn app.space_app:app --host 0.0.0.0 --port 7860
```

## Inference And Baseline

The submission evaluator expects a root-level `inference.py`. This repository includes it and uses the OpenAI Python client for all model calls.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example local run:

```bash
set API_BASE_URL=https://your-openai-compatible-endpoint/v1
set MODEL_NAME=your-model-id
set HF_TOKEN=your-token
python inference.py
```

Compatibility runner:

```bash
python scripts/run_baseline.py
```

The inference script:

- uses a fixed task order
- runs a consistent prompt format
- uses deterministic decoding with `temperature=0`
- saves `baseline_results.json`
- emits structured stdout logs using `[START]`, `[STEP]`, and `[END]`

## Docker

Build:

```bash
docker build -t customer-support-openenv .
```

Run:

```bash
docker run -p 7860:7860 customer-support-openenv
```

## Hugging Face Space

This repository includes a FastAPI app and a Dockerfile suitable for a containerized HF Space. The root URL returns `200`, and the service exposes programmatic `reset()`, `step()`, and `state()` endpoints for automated checks.

## Validation Notes

The benchmark is intentionally compact and auditable:

- deterministic task definitions
- typed observations and actions
- serializable internal state
- interpretable reward components
- reproducible fixed-order baseline
