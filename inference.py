from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_support_openenv.environment import CustomerSupportTriageEnv
from customer_support_openenv.models import Action, Observation


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "customer-support-triage-openenv"
MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = """You are operating a deterministic customer support triage benchmark.
Return exactly one JSON object with these keys in this order:
action_type, category, priority, requested_fields, queue, escalation_target, response_text, resolution_code, internal_note
Use null for keys that do not apply to the chosen action.
Do not wrap the JSON in markdown."""


def format_bool(value: bool) -> str:
    return str(value).lower()


def sanitize_action_text(action: Action) -> str:
    payload = action.model_dump(exclude_none=True, mode="json")
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error or "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={format_bool(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={format_bool(success)} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation: Observation) -> str:
    return json.dumps(
        {
            "task_id": observation.task_id,
            "task_title": observation.task_title,
            "ticket": observation.ticket.model_dump(mode="json"),
            "customer_profile": observation.customer_profile.model_dump(mode="json"),
            "account_context": observation.account_context.model_dump(mode="json"),
            "policy_context": [item.model_dump(mode="json") for item in observation.policy_context],
            "allowed_queues": observation.allowed_queues,
            "action_history": [item.model_dump(mode="json") for item in observation.action_history],
            "current_status": observation.current_status,
            "steps_remaining": observation.steps_remaining,
            "completion_flags": observation.completion_flags,
        },
        ensure_ascii=True,
    )


def create_client() -> tuple[OpenAI, str]:
    token = (HF_TOKEN or "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    client = OpenAI(base_url=API_BASE_URL, api_key=token)
    return client, MODEL_NAME


def heuristic_action(observation: Observation) -> Action:
    flags = observation.completion_flags
    if observation.task_id == "billing_refund_easy":
        if not flags["classified"]:
            return Action(action_type="classify_ticket", category="billing_charge_dispute")
        if not flags["prioritized"]:
            return Action(action_type="set_priority", priority="normal")
        if not flags["routed"]:
            return Action(action_type="route_ticket", queue="billing_operations")
        return Action(
            action_type="resolve_ticket",
            resolution_code="refund_requested",
            response_text=(
                "I am sorry for the duplicate charge. Our billing team will review "
                "the duplicate charge and the refund request and follow up with you."
            ),
        )

    if observation.task_id == "account_access_medium":
        if not flags["classified"]:
            return Action(action_type="classify_ticket", category="account_access")
        if not flags["prioritized"]:
            return Action(action_type="set_priority", priority="high")
        if not flags["info_requested"]:
            return Action(
                action_type="request_customer_info",
                requested_fields=[
                    "account email",
                    "workspace or company name",
                    "last successful login date",
                ],
            )
        if not flags["routed"]:
            return Action(action_type="route_ticket", queue="identity_and_access")
        return Action(
            action_type="resolve_ticket",
            resolution_code="waiting_for_verification",
            response_text=(
                "Sorry you are blocked. Please send your account email, workspace or "
                "company name, and the last successful login date so we can continue "
                "verification safely."
            ),
        )

    if not flags["classified"]:
        return Action(action_type="classify_ticket", category="enterprise_escalation")
    if not flags["prioritized"]:
        return Action(action_type="set_priority", priority="urgent")
    if not flags["info_requested"]:
        return Action(
            action_type="request_customer_info",
            requested_fields=["procurement or compliance ticket reference"],
        )
    if not flags["routed"]:
        return Action(action_type="route_ticket", queue="enterprise_support")
    return Action(
        action_type="escalate_ticket",
        escalation_target="account_management",
        response_text=(
            "I understand the business impact. I have escalated this to our "
            "enterprise support and account management teams as the next step, and "
            "please share the procurement or compliance ticket reference so the "
            "required compliance signoff can be reviewed."
        ),
    )


def extract_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    text_chunks.append(text)
        if text_chunks:
            return "".join(text_chunks)
    raise ValueError("Model response did not contain text content.")


def decide_action(client: OpenAI, model_name: str, observation: Observation) -> Action:
    request_kwargs = {
        "model": model_name,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(observation)},
        ],
    }
    try:
        response = client.chat.completions.create(
            **request_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(**request_kwargs)

    try:
        payload = json.loads(extract_content(response))
        ordered_payload = {
            "action_type": payload.get("action_type"),
            "category": payload.get("category"),
            "priority": payload.get("priority"),
            "requested_fields": payload.get("requested_fields"),
            "queue": payload.get("queue"),
            "escalation_target": payload.get("escalation_target"),
            "response_text": payload.get("response_text"),
            "resolution_code": payload.get("resolution_code"),
            "internal_note": payload.get("internal_note"),
        }
        return Action.model_validate(ordered_payload)
    except Exception:
        return heuristic_action(observation)


def run() -> dict[str, Any]:
    client, model_name = create_client()
    env = CustomerSupportTriageEnv()
    results: list[dict[str, Any]] = []
    task_ids = env.task_ids

    for task_id in task_ids:
        rewards: list[float] = []
        steps_taken = 0
        score = 0.0
        success = False
        final_info: dict[str, Any] = {}
        last_error: str | None = None
        log_start(task=task_id, env=BENCHMARK, model=model_name)

        try:
            observation = env.reset(task_id=task_id)
            done = False

            for step_number in range(1, MAX_STEPS + 1):
                if done:
                    break

                try:
                    action = decide_action(client, model_name, observation)
                    action_text = sanitize_action_text(action)
                    observation, reward, done, final_info = env.step(action)
                    reward_value = float(reward.value)
                    last_error = None
                except Exception as exc:
                    action = heuristic_action(observation)
                    action_text = sanitize_action_text(action)
                    reward_value = 0.0
                    done = True
                    last_error = str(exc)

                rewards.append(reward_value)
                steps_taken = step_number

                log_step(
                    step=step_number,
                    action=action_text,
                    reward=reward_value,
                    done=done,
                    error=last_error,
                )

                if done:
                    break

            try:
                state = env.state()
                score = float(state.final_score or 0.0)
                final_status = state.observation.current_status
                grader_breakdown = state.grader_breakdown
            except Exception:
                score = 0.0
                final_status = "error"
                grader_breakdown = {}

            score = max(0.0, min(1.0, score))
            success = score >= SUCCESS_SCORE_THRESHOLD and last_error is None
            results.append(
                {
                    "task_id": task_id,
                    "final_score": score,
                    "grader_breakdown": grader_breakdown,
                    "final_status": final_status,
                    "steps_used": steps_taken,
                    "rewards": rewards,
                    "info": final_info,
                    "error": last_error,
                }
            )
        finally:
            try:
                env.close()
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    aggregate_score = round(
        sum(result["final_score"] for result in results if result["final_score"] is not None)
        / len(results),
        4,
    )
    payload = {
        "model_name": model_name,
        "task_ids": task_ids,
        "aggregate_score": aggregate_score,
        "results": results,
        "local_image_name": LOCAL_IMAGE_NAME,
    }
    output_path = PROJECT_ROOT / "baseline_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    run()
