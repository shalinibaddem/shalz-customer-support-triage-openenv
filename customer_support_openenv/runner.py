from __future__ import annotations

from typing import Any, Callable

from .environment import CustomerSupportTriageEnv
from .models import Action, Observation


PolicyFn = Callable[[Observation], Action | dict[str, Any]]


def serialize_observation(observation: Observation) -> dict[str, Any]:
    return observation.model_dump(mode="json")


def serialize_state(env: CustomerSupportTriageEnv) -> dict[str, Any]:
    return env.state().model_dump(mode="json")


def run_episode(
    env: CustomerSupportTriageEnv,
    policy_fn: PolicyFn,
    task_id: str,
) -> dict[str, Any]:
    observation = env.reset(task_id=task_id)
    done = False
    rewards: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    info: dict[str, Any] = {}

    while not done:
        action = policy_fn(observation)
        action_model = action if isinstance(action, Action) else Action.model_validate(action)
        trace.append(action_model.model_dump(mode="json"))
        observation, reward, done, info = env.step(action_model)
        rewards.append(reward.model_dump(mode="json"))

    state = env.state()
    return {
        "task_id": task_id,
        "trace": trace,
        "rewards": rewards,
        "final_score": state.final_score,
        "grader_breakdown": state.grader_breakdown,
        "final_status": state.observation.current_status,
    }
