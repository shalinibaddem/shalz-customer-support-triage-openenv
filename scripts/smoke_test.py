from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_support_openenv.environment import CustomerSupportTriageEnv
from customer_support_openenv.models import Action


SCRIPTED_TRAJECTORIES: dict[str, list[Action]] = {
    "billing_refund_easy": [
        Action(action_type="classify_ticket", category="billing_charge_dispute"),
        Action(action_type="set_priority", priority="normal"),
        Action(action_type="route_ticket", queue="billing_operations"),
        Action(
            action_type="draft_response",
            response_text=(
                "I’m sorry for the duplicate charge. I’m sending this to our billing "
                "team for review and a refund request so they can verify the extra "
                "charge and follow up with you."
            ),
        ),
        Action(
            action_type="resolve_ticket",
            resolution_code="refund_requested",
            response_text=(
                "I’m sorry for the duplicate charge. Our billing team will review the "
                "charge and handle the refund request. We’ll follow up once the review "
                "is complete."
            ),
        ),
    ],
    "account_access_medium": [
        Action(action_type="classify_ticket", category="account_access"),
        Action(action_type="set_priority", priority="high"),
        Action(
            action_type="request_customer_info",
            requested_fields=[
                "account email",
                "workspace or company name",
                "last successful login date",
            ],
        ),
        Action(action_type="route_ticket", queue="identity_and_access"),
        Action(
            action_type="draft_response",
            response_text=(
                "I’m sorry you’re locked out. To verify the account safely, please "
                "reply with your account email, your workspace or company name, and "
                "the last successful login date so we can continue the verification."
            ),
        ),
        Action(
            action_type="resolve_ticket",
            resolution_code="waiting_for_verification",
            response_text=(
                "I’m sorry for the disruption. Before we can help restore access, "
                "please send your account email, workspace or company name, and the "
                "last successful login date so we can complete verification."
            ),
        ),
    ],
    "enterprise_policy_hard": [
        Action(action_type="classify_ticket", category="enterprise_escalation"),
        Action(action_type="set_priority", priority="urgent"),
        Action(
            action_type="request_customer_info",
            requested_fields=["procurement or compliance ticket reference"],
        ),
        Action(action_type="route_ticket", queue="enterprise_support"),
        Action(
            action_type="draft_response",
            response_text=(
                "I understand the business impact to your launch. I’m escalating this "
                "to our enterprise support and account management teams now, and I’ll "
                "need your procurement or compliance ticket reference so they can "
                "coordinate the next step and confirm compliance signoff."
            ),
        ),
        Action(
            action_type="escalate_ticket",
            escalation_target="account_management",
            response_text=(
                "I understand the business impact here. I have escalated this to our "
                "enterprise support and account management teams as the next step, and "
                "please share the procurement or compliance ticket reference so they "
                "can work through the compliance signoff required for reactivation."
            ),
        ),
    ],
}


def main() -> None:
    env = CustomerSupportTriageEnv()
    results: dict[str, dict[str, object]] = {}

    for task_id, actions in SCRIPTED_TRAJECTORIES.items():
        env.reset(task_id=task_id)
        done = False
        step_count = 0
        last_info: dict[str, object] = {}
        while not done:
            action = actions[step_count]
            _, reward, done, last_info = env.step(action)
            step_count += 1
            if step_count > 10:
                raise RuntimeError(f"Smoke test exceeded expected steps for {task_id}")

        state = env.state()
        results[task_id] = {
            "final_score": state.final_score,
            "grader_breakdown": state.grader_breakdown,
            "last_reward": reward.model_dump(mode="json"),
            "info": last_info,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
