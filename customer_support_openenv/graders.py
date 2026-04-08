from __future__ import annotations

import re
from typing import Iterable

from .models import Action, EnvironmentState, TaskRubric


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def action_payload_signature(action: Action) -> tuple:
    payload = action.model_dump(exclude_none=True)
    action_type = payload.pop("action_type")
    normalized_items: list[tuple[str, object]] = []
    for key, value in sorted(payload.items()):
        if isinstance(value, list):
            normalized_value = tuple(str(item).strip().lower() for item in value)
        elif isinstance(value, str):
            normalized_value = value.strip().lower()
        else:
            normalized_value = value
        normalized_items.append((key, normalized_value))
    return (action_type, tuple(normalized_items))


def get_actions_by_type(state: EnvironmentState, action_type: str) -> list[Action]:
    return [action for action in state.action_trace if action.action_type == action_type]


def latest_action(state: EnvironmentState, action_type: str) -> Action | None:
    actions = get_actions_by_type(state, action_type)
    return actions[-1] if actions else None


def all_response_texts(state: EnvironmentState) -> list[str]:
    responses: list[str] = []
    for action in state.action_trace:
        if action.response_text:
            responses.append(action.response_text)
    return responses


def latest_response_text(state: EnvironmentState) -> str:
    responses = all_response_texts(state)
    return responses[-1] if responses else ""


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


def check_classification(state: EnvironmentState) -> float:
    return 1.0 if any(
        action.category == state.expected_category
        for action in get_actions_by_type(state, "classify_ticket")
    ) else 0.0


def check_priority(state: EnvironmentState) -> float:
    return 1.0 if any(
        action.priority == state.expected_priority
        for action in get_actions_by_type(state, "set_priority")
    ) else 0.0


def check_routing(state: EnvironmentState) -> float:
    if not state.valid_queues:
        return 1.0
    return 1.0 if any(
        action.queue in state.valid_queues
        for action in get_actions_by_type(state, "route_ticket")
    ) else 0.0


def check_info_request(state: EnvironmentState) -> float:
    if not state.required_info_fields:
        return 1.0

    requests = get_actions_by_type(state, "request_customer_info")
    if not requests:
        return 0.0

    required = {normalize_text(field) for field in state.required_info_fields}
    best_coverage = 0.0
    for request in requests:
        requested = {normalize_text(field) for field in (request.requested_fields or [])}
        coverage = len(required & requested) / len(required)
        best_coverage = max(best_coverage, coverage)
    return best_coverage


def check_escalation(state: EnvironmentState) -> float:
    if state.required_escalation_target is None:
        escalations = get_actions_by_type(state, "escalate_ticket")
        return 1.0 if not escalations else 0.0

    return 1.0 if any(
        action.escalation_target == state.required_escalation_target
        for action in get_actions_by_type(state, "escalate_ticket")
    ) else 0.0


def check_resolution_or_escalation(state: EnvironmentState) -> float:
    if state.terminal_action == "resolve_ticket":
        terminal_actions = get_actions_by_type(state, "resolve_ticket")
        if not terminal_actions:
            return 0.0
        return 1.0 if any(
            action.resolution_code == state.expected_resolution_code
            for action in terminal_actions
        ) else 0.0

    terminal_actions = get_actions_by_type(state, "escalate_ticket")
    if not terminal_actions:
        return 0.0

    return 1.0 if any(
        action.escalation_target == state.required_escalation_target
        and action.response_text
        for action in terminal_actions
    ) else 0.0


def check_policy_compliance(state: EnvironmentState) -> float:
    response_texts = all_response_texts(state)
    if not response_texts:
        return 0.0

    for text in response_texts:
        if any(contains_phrase(text, phrase) for phrase in state.response_must_not_include):
            return 0.0
    return 1.0


def check_response_quality(state: EnvironmentState) -> float:
    response = latest_response_text(state)
    if not response:
        return 0.0

    required_hits = 0
    if state.response_must_include:
        required_hits = sum(
            1 for phrase in state.response_must_include if contains_phrase(response, phrase)
        )
        required_score = required_hits / len(state.response_must_include)
    else:
        required_score = 1.0

    any_of_scores: list[float] = []
    for phrase_group in state.required_response_any_of:
        any_of_scores.append(1.0 if contains_any_phrase(response, phrase_group) else 0.0)
    grouped_score = sum(any_of_scores) / len(any_of_scores) if any_of_scores else 1.0

    policy_score = check_policy_compliance(state)
    return round(required_score * 0.7 + grouped_score * 0.2 + policy_score * 0.1, 4)


def grade_task(state: EnvironmentState, rubric: TaskRubric) -> dict[str, float]:
    breakdown = {
        "classification": check_classification(state),
        "priority": check_priority(state),
        "info_request": check_info_request(state),
        "routing": check_routing(state),
        "escalation": check_escalation(state),
        "response": check_response_quality(state),
        "resolution": check_resolution_or_escalation(state),
        "policy_compliance": check_policy_compliance(state),
    }

    applicable_weights = {
        "classification": rubric.classification_weight,
        "priority": rubric.priority_weight,
        "info_request": rubric.info_request_weight if state.required_info_fields else 0.0,
        "routing": rubric.routing_weight if state.valid_queues else 0.0,
        "escalation": (
            rubric.escalation_weight if state.required_escalation_target is not None else 0.0
        ),
        "response": rubric.response_weight,
        "resolution": rubric.resolution_weight,
    }
    weighted_sum = sum(
        breakdown[dimension] * weight for dimension, weight in applicable_weights.items()
    )
    total_weight = sum(applicable_weights.values()) or 1.0
    breakdown["final_score"] = round(weighted_sum / total_weight, 4)
    return breakdown


def grade_trajectory(state: EnvironmentState, rubric: TaskRubric | None = None) -> float:
    score_rubric = rubric or TaskRubric()
    return grade_task(state, score_rubric)["final_score"]
