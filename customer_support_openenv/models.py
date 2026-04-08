from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ActionType = Literal[
    "classify_ticket",
    "set_priority",
    "request_customer_info",
    "route_ticket",
    "draft_response",
    "escalate_ticket",
    "resolve_ticket",
]

TicketCategory = Literal[
    "billing_refund",
    "billing_charge_dispute",
    "account_access",
    "identity_verification",
    "technical_bug",
    "enterprise_escalation",
    "policy_question",
    "general_inquiry",
]

PriorityLevel = Literal["low", "normal", "high", "urgent"]
TicketStatus = Literal[
    "new",
    "triaging",
    "waiting_for_customer",
    "routed",
    "escalated",
    "resolved",
]
TicketChannel = Literal["email", "web_form", "chat"]
CustomerTier = Literal["free", "pro", "business", "enterprise"]
ProductArea = Literal["billing", "auth", "workspace", "api", "compliance"]
PlanStatus = Literal["active", "past_due", "trial", "suspended", "canceled"]
PaymentStatus = Literal["paid", "failed", "refunded", "n/a"]
SentimentHint = Literal["calm", "frustrated", "angry", "urgent"]
ResolutionCode = Literal[
    "refund_requested",
    "waiting_for_verification",
    "routed_to_support",
    "escalated_to_enterprise",
    "resolved_with_guidance",
]


class SupportBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ticket(SupportBaseModel):
    ticket_id: str
    subject: str
    body: str
    created_at: str
    channel: TicketChannel
    language: str


class CustomerProfile(SupportBaseModel):
    customer_id: str
    full_name: str
    tier: CustomerTier
    region: str
    tenure_days: int = Field(ge=0)
    sentiment_hint: SentimentHint


class AccountContext(SupportBaseModel):
    product_area: ProductArea
    plan_status: PlanStatus
    last_payment_status: PaymentStatus
    renewal_date: str | None = None
    open_incidents: list[str] = Field(default_factory=list)
    previous_tickets: list[str] = Field(default_factory=list)


class PolicySnippet(SupportBaseModel):
    policy_id: str
    title: str
    content: str


class ActionRecord(SupportBaseModel):
    step_index: int = Field(ge=0)
    action_type: ActionType
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Observation(SupportBaseModel):
    task_id: str
    task_title: str
    ticket: Ticket
    customer_profile: CustomerProfile
    account_context: AccountContext
    policy_context: list[PolicySnippet]
    allowed_queues: list[str]
    action_history: list[ActionRecord]
    current_status: TicketStatus
    steps_remaining: int = Field(ge=0)
    completion_flags: dict[str, bool]


class Action(SupportBaseModel):
    action_type: ActionType
    category: TicketCategory | None = None
    priority: PriorityLevel | None = None
    requested_fields: list[str] | None = None
    queue: str | None = None
    escalation_target: str | None = None
    response_text: str | None = None
    resolution_code: ResolutionCode | None = None
    internal_note: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "Action":
        required_fields: dict[ActionType, tuple[str, ...]] = {
            "classify_ticket": ("category",),
            "set_priority": ("priority",),
            "request_customer_info": ("requested_fields",),
            "route_ticket": ("queue",),
            "draft_response": ("response_text",),
            "escalate_ticket": ("escalation_target", "response_text"),
            "resolve_ticket": ("resolution_code", "response_text"),
        }

        missing = [
            field_name
            for field_name in required_fields[self.action_type]
            if getattr(self, field_name) in (None, [], "")
        ]
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                f"action_type='{self.action_type}' requires fields: {missing_fields}"
            )
        return self


class Reward(SupportBaseModel):
    value: float
    components: dict[str, float] = Field(default_factory=dict)
    reason: str


class TaskRubric(SupportBaseModel):
    classification_weight: float = Field(default=0.15, ge=0.0)
    priority_weight: float = Field(default=0.10, ge=0.0)
    info_request_weight: float = Field(default=0.15, ge=0.0)
    routing_weight: float = Field(default=0.20, ge=0.0)
    escalation_weight: float = Field(default=0.15, ge=0.0)
    response_weight: float = Field(default=0.10, ge=0.0)
    resolution_weight: float = Field(default=0.15, ge=0.0)

    @property
    def total_weight(self) -> float:
        return (
            self.classification_weight
            + self.priority_weight
            + self.info_request_weight
            + self.routing_weight
            + self.escalation_weight
            + self.response_weight
            + self.resolution_weight
        )


class TaskExpectation(SupportBaseModel):
    expected_category: TicketCategory
    expected_priority: PriorityLevel
    required_info_fields: list[str] = Field(default_factory=list)
    valid_queues: list[str] = Field(default_factory=list)
    required_escalation_target: str | None = None
    expected_resolution_code: ResolutionCode | None = None
    response_must_include: list[str] = Field(default_factory=list)
    response_must_not_include: list[str] = Field(default_factory=list)
    required_response_any_of: list[list[str]] = Field(default_factory=list)
    terminal_action: Literal["resolve_ticket", "escalate_ticket"]


class TaskSpec(SupportBaseModel):
    task_id: str
    task_title: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    ticket: Ticket
    customer_profile: CustomerProfile
    account_context: AccountContext
    policy_context: list[PolicySnippet]
    allowed_queues: list[str]
    max_steps: int = Field(default=6, ge=1)
    expectation: TaskExpectation
    rubric: TaskRubric = Field(default_factory=TaskRubric)


class EnvironmentState(SupportBaseModel):
    task_id: str
    observation: Observation
    expected_category: TicketCategory
    expected_priority: PriorityLevel
    required_info_fields: list[str] = Field(default_factory=list)
    valid_queues: list[str] = Field(default_factory=list)
    required_escalation_target: str | None = None
    expected_resolution_code: ResolutionCode | None = None
    response_must_include: list[str] = Field(default_factory=list)
    response_must_not_include: list[str] = Field(default_factory=list)
    required_response_any_of: list[list[str]] = Field(default_factory=list)
    terminal_action: Literal["resolve_ticket", "escalate_ticket"]
    achieved_subgoals: dict[str, bool] = Field(default_factory=dict)
    invalid_action_count: int = Field(default=0, ge=0)
    repeated_action_count: int = Field(default=0, ge=0)
    done: bool = False
    final_score: float | None = None
    last_action: Action | None = None
    last_reward: Reward | None = None
    action_trace: list[Action] = Field(default_factory=list)
    grader_breakdown: dict[str, float] = Field(default_factory=dict)


class StepInfo(SupportBaseModel):
    task_id: str
    done: bool
    invalid_action: bool = False
    grader_breakdown: dict[str, float] = Field(default_factory=dict)
    final_score: float | None = None
    notes: list[str] = Field(default_factory=list)
