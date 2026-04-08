from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_support_openenv.environment import CustomerSupportTriageEnv
from customer_support_openenv.models import Action


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: Action | dict[str, Any] = Field(
        ...,
        description="Structured support action payload that will be passed to env.step().",
    )


ENV = CustomerSupportTriageEnv()
app = FastAPI(
    title="Customer Support Triage OpenEnv",
    version="0.1.0",
    description=(
        "Deterministic SaaS customer support triage benchmark with reset(), step(), "
        "and state() HTTP endpoints."
    ),
)


def ensure_initialized() -> None:
    try:
        ENV.state()
    except RuntimeError:
        ENV.reset()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment": "customer-support-triage-openenv",
        "task_ids": ENV.task_ids,
        "endpoints": ["/reset", "/step", "/state", "/healthz"],
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/reset")
def reset_get(task_id: str | None = None) -> dict[str, Any]:
    observation = ENV.reset(task_id=task_id)
    return {
        "observation": observation.model_dump(mode="json"),
        "done": False,
    }


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> dict[str, Any]:
    observation = ENV.reset(task_id=payload.task_id if payload else None)
    return {
        "observation": observation.model_dump(mode="json"),
        "done": False,
    }


@app.post("/step")
def step(payload: StepRequest) -> dict[str, Any]:
    ensure_initialized()
    observation, reward, done, info = ENV.step(payload.action)
    return {
        "observation": observation.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    ensure_initialized()
    return ENV.state().model_dump(mode="json")
