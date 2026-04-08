from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .graders import action_payload_signature
from .models import (
    Action,
    ActionRecord,
    EnvironmentState,
    Observation,
    Reward,
    StepInfo,
    TaskSpec,
    TicketStatus,
)
from .rewards import compute_step_reward, finalize_state_scores, refresh_subgoals
from .tasks import get_all_tasks, get_task


class CustomerSupportTriageEnv:
    """Deterministic multi-step support triage benchmark."""

    def __init__(self, default_max_steps: int | None = None):
        self._tasks = get_all_tasks()
        self._task_order = list(self._tasks)
        self._task_index = 0
        self._default_max_steps = default_max_steps
        self._current_task: TaskSpec | None = None
        self._state: EnvironmentState | None = None

    @property
    def task_ids(self) -> list[str]:
        return list(self._task_order)

    def reset(self, task_id: str | None = None) -> Observation:
        next_task_id = task_id or self._next_task_id()
        task = get_task(next_task_id)
        if self._default_max_steps is not None:
            task.max_steps = self._default_max_steps

        observation = Observation(
            task_id=task.task_id,
            task_title=task.task_title,
            ticket=task.ticket,
            customer_profile=task.customer_profile,
            account_context=task.account_context,
            policy_context=task.policy_context,
            allowed_queues=task.allowed_queues,
            action_history=[],
            current_status="new",
            steps_remaining=task.max_steps,
            completion_flags={
                "classified": False,
                "prioritized": False,
                "info_requested": False if task.expectation.required_info_fields else True,
                "routed": False if task.expectation.valid_queues else True,
                "escalated": False if task.expectation.required_escalation_target else True,
                "responded": False,
                "terminal_action_taken": False,
            },
        )
        self._current_task = task
        self._state = EnvironmentState(
            task_id=task.task_id,
            observation=observation,
            expected_category=task.expectation.expected_category,
            expected_priority=task.expectation.expected_priority,
            required_info_fields=task.expectation.required_info_fields,
            valid_queues=task.expectation.valid_queues,
            required_escalation_target=task.expectation.required_escalation_target,
            expected_resolution_code=task.expectation.expected_resolution_code,
            response_must_include=task.expectation.response_must_include,
            response_must_not_include=task.expectation.response_must_not_include,
            required_response_any_of=task.expectation.required_response_any_of,
            terminal_action=task.expectation.terminal_action,
            achieved_subgoals={},
        )
        self._state.achieved_subgoals = refresh_subgoals(self._state)
        return self._state.observation.model_copy(deep=True)

    def state(self) -> EnvironmentState:
        self._require_state()
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        self._current_task = None
        self._state = None

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        self._require_state()
        prev_state = self._state.model_copy(deep=True)

        if self._state.done:
            reward = Reward(
                value=-0.10,
                components={"already_done": -0.10},
                reason="Episode already ended. Call reset() before stepping again.",
            )
            self._state.invalid_action_count += 1
            self._state.last_reward = reward
            return self._state.observation.model_copy(deep=True), reward, True, self._build_info(
                invalid_action=True,
                notes=["Step ignored because the episode is already complete."],
            )

        try:
            parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        except ValidationError as exc:
            self._state.invalid_action_count += 1
            self._decrement_step_budget()
            if self._state.observation.steps_remaining == 0:
                self._state.done = True
                self._finalize()
            reward = compute_step_reward(
                prev_state,
                self._state,
                Action(action_type="draft_response", response_text="placeholder"),
                self._current_task.rubric,
                invalid_action=True,
                invalid_reason=str(exc),
            )
            self._state.last_reward = reward
            return self._state.observation.model_copy(deep=True), reward, self._state.done, self._build_info(
                invalid_action=True,
                notes=["Action failed schema validation."],
            )

        if self._state.last_action is not None:
            previous_signature = action_payload_signature(self._state.last_action)
            current_signature = action_payload_signature(parsed_action)
            if previous_signature == current_signature:
                self._state.repeated_action_count += 1

        self._apply_action(parsed_action)
        self._decrement_step_budget()

        if parsed_action.action_type in {"resolve_ticket", "escalate_ticket"}:
            self._state.done = True
        elif self._state.observation.steps_remaining == 0:
            self._state.done = True

        if self._state.done:
            self._finalize()
        else:
            self._state.achieved_subgoals = refresh_subgoals(self._state)

        reward = compute_step_reward(
            prev_state,
            self._state,
            parsed_action,
            self._current_task.rubric,
        )
        self._state.last_action = parsed_action
        self._state.last_reward = reward
        return (
            self._state.observation.model_copy(deep=True),
            reward,
            self._state.done,
            self._build_info(),
        )

    def _apply_action(self, action: Action) -> None:
        assert self._state is not None

        current_status = self._state.observation.current_status
        next_status = self._status_after_action(current_status, action)
        self._state.observation.current_status = next_status
        self._state.action_trace.append(action)
        self._state.observation.action_history.append(
            ActionRecord(
                step_index=len(self._state.observation.action_history),
                action_type=action.action_type,
                summary=self._action_summary(action),
                payload=action.model_dump(exclude_none=True),
            )
        )
        self._update_completion_flags(action)

    def _update_completion_flags(self, action: Action) -> None:
        flags = self._state.observation.completion_flags
        if action.action_type == "classify_ticket" and action.category == self._state.expected_category:
            flags["classified"] = True
        if action.action_type == "set_priority" and action.priority == self._state.expected_priority:
            flags["prioritized"] = True
        if action.action_type == "request_customer_info":
            requested_fields = {item.strip().lower() for item in (action.requested_fields or [])}
            required_fields = {item.strip().lower() for item in self._state.required_info_fields}
            if required_fields.issubset(requested_fields):
                flags["info_requested"] = True
        if action.action_type == "route_ticket" and action.queue in self._state.valid_queues:
            flags["routed"] = True
        if (
            action.action_type == "escalate_ticket"
            and action.escalation_target == self._state.required_escalation_target
        ):
            flags["escalated"] = True
        if action.response_text:
            flags["responded"] = True
        if action.action_type in {"resolve_ticket", "escalate_ticket"}:
            flags["terminal_action_taken"] = True

    def _status_after_action(self, current_status: TicketStatus, action: Action) -> TicketStatus:
        if action.action_type in {"classify_ticket", "set_priority", "draft_response"}:
            return "triaging"
        if action.action_type == "request_customer_info":
            return "waiting_for_customer"
        if action.action_type == "route_ticket":
            return "routed"
        if action.action_type == "escalate_ticket":
            return "escalated"
        if action.action_type == "resolve_ticket":
            if action.resolution_code == "waiting_for_verification":
                return "waiting_for_customer"
            return "resolved"
        return current_status

    def _action_summary(self, action: Action) -> str:
        if action.action_type == "classify_ticket":
            return f"Classified ticket as {action.category}."
        if action.action_type == "set_priority":
            return f"Set priority to {action.priority}."
        if action.action_type == "request_customer_info":
            fields = ", ".join(action.requested_fields or [])
            return f"Requested customer details: {fields}."
        if action.action_type == "route_ticket":
            return f"Routed ticket to {action.queue}."
        if action.action_type == "draft_response":
            return "Drafted a customer-facing response."
        if action.action_type == "escalate_ticket":
            return f"Escalated ticket to {action.escalation_target}."
        return f"Resolved ticket as {action.resolution_code}."

    def _decrement_step_budget(self) -> None:
        assert self._state is not None
        self._state.observation.steps_remaining = max(0, self._state.observation.steps_remaining - 1)

    def _finalize(self) -> None:
        assert self._state is not None
        assert self._current_task is not None
        finalize_state_scores(self._state, self._current_task.rubric)

    def _build_info(
        self,
        *,
        invalid_action: bool = False,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        info = StepInfo(
            task_id=self._state.task_id,
            done=self._state.done,
            invalid_action=invalid_action,
            grader_breakdown=self._state.grader_breakdown,
            final_score=self._state.final_score,
            notes=notes or [],
        )
        return info.model_dump()

    def _next_task_id(self) -> str:
        task_id = self._task_order[self._task_index % len(self._task_order)]
        self._task_index += 1
        return task_id

    def _require_state(self) -> None:
        if self._state is None or self._current_task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
