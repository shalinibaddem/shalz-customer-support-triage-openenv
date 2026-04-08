from __future__ import annotations

from customer_support_openenv.environment import CustomerSupportTriageEnv
from customer_support_openenv.models import Action


def test_reset_returns_clean_observation() -> None:
    env = CustomerSupportTriageEnv()
    observation = env.reset(task_id="billing_refund_easy")

    assert observation.task_id == "billing_refund_easy"
    assert observation.current_status == "new"
    assert observation.steps_remaining == 6
    assert observation.action_history == []


def test_state_reflects_progress_after_step() -> None:
    env = CustomerSupportTriageEnv()
    env.reset(task_id="billing_refund_easy")
    env.step(Action(action_type="classify_ticket", category="billing_charge_dispute"))

    state = env.state()
    assert state.observation.current_status == "triaging"
    assert state.observation.completion_flags["classified"] is True
    assert len(state.action_trace) == 1


def test_episode_terminates_on_valid_terminal_action() -> None:
    env = CustomerSupportTriageEnv()
    env.reset(task_id="billing_refund_easy")
    env.step(Action(action_type="classify_ticket", category="billing_charge_dispute"))
    env.step(Action(action_type="set_priority", priority="normal"))
    env.step(Action(action_type="route_ticket", queue="billing_operations"))
    env.step(
        Action(
            action_type="resolve_ticket",
            resolution_code="refund_requested",
            response_text=(
                "I’m sorry for the duplicate charge. The billing team will review "
                "the refund request and follow up with you."
            ),
        )
    )

    state = env.state()
    assert state.done is True
    assert state.final_score is not None


def test_repeated_actions_are_penalized() -> None:
    env = CustomerSupportTriageEnv()
    env.reset(task_id="billing_refund_easy")
    action = Action(action_type="set_priority", priority="normal")
    env.step(action)
    _, reward, _, _ = env.step(action)

    assert reward.components["repeated_action"] == -0.05


def test_invalid_actions_increment_invalid_count() -> None:
    env = CustomerSupportTriageEnv()
    env.reset(task_id="billing_refund_easy")
    _, reward, _, info = env.step({"action_type": "route_ticket"})

    state = env.state()
    assert state.invalid_action_count == 1
    assert reward.value < 0
    assert info["invalid_action"] is True
