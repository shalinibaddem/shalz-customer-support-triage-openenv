from __future__ import annotations

from customer_support_openenv.environment import CustomerSupportTriageEnv
from customer_support_openenv.graders import grade_task
from customer_support_openenv.models import Action
from customer_support_openenv.tasks import get_task


def play_actions(env: CustomerSupportTriageEnv, task_id: str, actions: list[Action]) -> float:
    env.reset(task_id=task_id)
    for action in actions:
        _, _, done, _ = env.step(action)
        if done:
            break
    state = env.state()
    rubric = get_task(task_id).rubric
    return grade_task(state, rubric)["final_score"]


def test_easy_perfect_trajectory_gets_full_score() -> None:
    env = CustomerSupportTriageEnv()
    score = play_actions(
        env,
        "billing_refund_easy",
        [
            Action(action_type="classify_ticket", category="billing_charge_dispute"),
            Action(action_type="set_priority", priority="normal"),
            Action(action_type="route_ticket", queue="billing_operations"),
            Action(
                action_type="resolve_ticket",
                resolution_code="refund_requested",
                response_text=(
                    "I’m sorry for the duplicate charge. Our billing team will review "
                    "the charge and the refund request and follow up with you."
                ),
            ),
        ],
    )
    assert score == 1.0


def test_wrong_trajectory_scores_low() -> None:
    env = CustomerSupportTriageEnv()
    score = play_actions(
        env,
        "billing_refund_easy",
        [
            Action(action_type="classify_ticket", category="technical_bug"),
            Action(action_type="set_priority", priority="urgent"),
            Action(action_type="route_ticket", queue="technical_support"),
            Action(
                action_type="resolve_ticket",
                resolution_code="resolved_with_guidance",
                response_text="We already refunded this and fixed the bug.",
            ),
        ],
    )
    assert score <= 0.2


def test_medium_partial_success_is_intermediate() -> None:
    env = CustomerSupportTriageEnv()
    score = play_actions(
        env,
        "account_access_medium",
        [
            Action(action_type="classify_ticket", category="account_access"),
            Action(action_type="set_priority", priority="high"),
            Action(
                action_type="request_customer_info",
                requested_fields=["account email", "workspace or company name"],
            ),
            Action(action_type="route_ticket", queue="identity_and_access"),
            Action(
                action_type="resolve_ticket",
                resolution_code="waiting_for_verification",
                response_text=(
                    "Sorry you’re blocked. Please send your account email and workspace "
                    "or company name so we can verify the account."
                ),
            ),
        ],
    )
    assert 0.4 <= score < 1.0


def test_hard_task_grader_is_deterministic() -> None:
    env = CustomerSupportTriageEnv()
    actions = [
        Action(action_type="classify_ticket", category="enterprise_escalation"),
        Action(action_type="set_priority", priority="urgent"),
        Action(
            action_type="request_customer_info",
            requested_fields=["procurement or compliance ticket reference"],
        ),
        Action(action_type="route_ticket", queue="enterprise_support"),
        Action(
            action_type="escalate_ticket",
            escalation_target="account_management",
            response_text=(
                "I understand the business impact. I have escalated this to our "
                "enterprise support and account management teams as the next step, "
                "and we need the procurement or compliance ticket reference so the "
                "compliance signoff can be reviewed."
            ),
        ),
    ]
    first_score = play_actions(env, "enterprise_policy_hard", actions)
    second_score = play_actions(env, "enterprise_policy_hard", actions)
    assert first_score == second_score
