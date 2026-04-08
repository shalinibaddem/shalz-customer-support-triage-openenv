from __future__ import annotations

from .graders import (
    action_payload_signature,
    check_classification,
    check_escalation,
    check_info_request,
    check_policy_compliance,
    check_priority,
    check_resolution_or_escalation,
    check_response_quality,
    check_routing,
    contains_phrase,
    grade_task,
)
from .models import Action, EnvironmentState, Reward, TaskRubric


def compute_step_reward(
    prev_state: EnvironmentState,
    new_state: EnvironmentState,
    action: Action,
    rubric: TaskRubric,
    *,
    invalid_action: bool = False,
    invalid_reason: str = "",
) -> Reward:
    components: dict[str, float] = {}
    reasons: list[str] = []

    if invalid_action:
        components["invalid_action"] = -0.10
        reason = invalid_reason or "Invalid action payload."
        return Reward(value=-0.10, components=components, reason=reason)

    previous_signature = (
        action_payload_signature(prev_state.last_action) if prev_state.last_action else None
    )
    current_signature = action_payload_signature(action)
    if previous_signature == current_signature:
        components["repeated_action"] = -0.05
        reasons.append("Repeated the previous action without adding new information.")

    if not prev_state.achieved_subgoals.get("classification") and check_classification(new_state) == 1.0:
        components["classification"] = 0.15
        reasons.append("Captured the correct issue category.")

    if not prev_state.achieved_subgoals.get("priority") and check_priority(new_state) == 1.0:
        components["priority"] = 0.10
        reasons.append("Assigned the correct urgency.")

    if not prev_state.achieved_subgoals.get("info_request") and check_info_request(new_state) == 1.0:
        components["info_request"] = 0.15
        reasons.append("Requested the required verification details.")

    if not prev_state.achieved_subgoals.get("routing") and check_routing(new_state) == 1.0:
        components["routing"] = 0.20
        reasons.append("Routed the ticket to the correct queue.")

    if not prev_state.achieved_subgoals.get("escalation") and check_escalation(new_state) == 1.0:
        if new_state.required_escalation_target is not None:
            components["escalation"] = 0.15
            reasons.append("Escalated to the required specialized team.")

    if not prev_state.achieved_subgoals.get("response") and check_response_quality(new_state) >= 0.8:
        components["response"] = 0.10
        reasons.append("Drafted a safe response with the needed details.")

    if not prev_state.achieved_subgoals.get("resolution") and check_resolution_or_escalation(new_state) == 1.0:
        components["resolution"] = 0.15
        reasons.append("Completed the correct terminal action.")

    if action.action_type == "resolve_ticket":
        info_ready = check_info_request(new_state) == 1.0
        routing_ready = check_routing(new_state) == 1.0
        if new_state.required_info_fields and not info_ready:
            components["premature_resolution"] = -0.15
            reasons.append("Resolved before collecting the required customer details.")
        elif not routing_ready:
            components["premature_resolution"] = -0.15
            reasons.append("Resolved before routing the ticket correctly.")

    if action.action_type == "route_ticket" and action.queue not in new_state.valid_queues:
        components["wrong_route"] = -0.10
        reasons.append("Sent the ticket to the wrong internal queue.")

    if action.action_type == "escalate_ticket":
        if new_state.required_escalation_target is None:
            components["unneeded_escalation"] = -0.10
            reasons.append("Escalated a task that should have stayed in standard support flow.")
        elif action.escalation_target != new_state.required_escalation_target:
            components["wrong_escalation"] = -0.10
            reasons.append("Escalated to the wrong specialized team.")

    if action.response_text and any(
        contains_phrase(action.response_text, phrase)
        for phrase in new_state.response_must_not_include
    ):
        components["forbidden_claim"] = -0.20
        reasons.append("The customer response included a forbidden promise or claim.")

    if new_state.done and new_state.final_score is not None:
        components["terminal_bonus"] = round(0.25 * new_state.final_score, 4)
        reasons.append("Applied the terminal rubric bonus.")

    reward_value = round(sum(components.values()), 4)
    if not reasons:
        reasons.append("Action recorded without new progress.")

    return Reward(
        value=reward_value,
        components=components,
        reason=" ".join(reasons),
    )


def refresh_subgoals(state: EnvironmentState) -> dict[str, bool]:
    return {
        "classification": check_classification(state) == 1.0,
        "priority": check_priority(state) == 1.0,
        "info_request": check_info_request(state) == 1.0,
        "routing": check_routing(state) == 1.0,
        "escalation": (
            check_escalation(state) == 1.0 if state.required_escalation_target is not None else False
        ),
        "response": check_response_quality(state) >= 0.8,
        "resolution": check_resolution_or_escalation(state) == 1.0,
        "policy_compliance": check_policy_compliance(state) == 1.0,
    }


def finalize_state_scores(state: EnvironmentState, rubric: TaskRubric) -> EnvironmentState:
    breakdown = grade_task(state, rubric)
    state.grader_breakdown = breakdown
    state.final_score = breakdown["final_score"]
    state.achieved_subgoals = refresh_subgoals(state)
    return state
