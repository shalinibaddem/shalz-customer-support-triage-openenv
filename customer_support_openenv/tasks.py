from __future__ import annotations

from .models import (
    AccountContext,
    CustomerProfile,
    TaskExpectation,
    TaskRubric,
    TaskSpec,
    Ticket,
)
from .policies import get_policies


DEFAULT_RUBRIC = TaskRubric()


TASKS: dict[str, TaskSpec] = {
    "billing_refund_easy": TaskSpec(
        task_id="billing_refund_easy",
        task_title="Duplicate Billing Charge Refund Request",
        difficulty="easy",
        description=(
            "A business-tier customer reports a duplicate charge and asks for one "
            "of the charges to be refunded."
        ),
        ticket=Ticket(
            ticket_id="T-1001",
            subject="Charged twice this month",
            body=(
                "Hi support, I was billed twice for our March subscription and I "
                "need one of those charges refunded. I checked our card statement "
                "and there are two identical charges from your company on the same "
                "day. Please fix this as soon as you can."
            ),
            created_at="2026-03-25T08:15:00Z",
            channel="email",
            language="en",
        ),
        customer_profile=CustomerProfile(
            customer_id="CUST-201",
            full_name="Maya Patel",
            tier="business",
            region="IN",
            tenure_days=418,
            sentiment_hint="frustrated",
        ),
        account_context=AccountContext(
            product_area="billing",
            plan_status="active",
            last_payment_status="paid",
            renewal_date="2026-04-01",
            open_incidents=[],
            previous_tickets=[
                "Asked about invoice export in January.",
                "No unresolved issues on the account.",
            ],
        ),
        policy_context=get_policies("refund_window", "priority_sla"),
        allowed_queues=["general_support", "billing_operations", "technical_support"],
        max_steps=6,
        expectation=TaskExpectation(
            expected_category="billing_charge_dispute",
            expected_priority="normal",
            required_info_fields=[],
            valid_queues=["billing_operations"],
            required_escalation_target=None,
            expected_resolution_code="refund_requested",
            response_must_include=[
                "duplicate charge",
                "billing",
                "review",
            ],
            response_must_not_include=[
                "refund has been processed",
                "we already refunded",
                "refund is guaranteed",
            ],
            required_response_any_of=[["refund request", "refund review"]],
            terminal_action="resolve_ticket",
        ),
        rubric=DEFAULT_RUBRIC,
    ),
    "account_access_medium": TaskSpec(
        task_id="account_access_medium",
        task_title="Account Access Issue With Missing Verification Details",
        difficulty="medium",
        description=(
            "A pro-tier user cannot log in after changing email and does not provide "
            "enough information to safely restore access."
        ),
        ticket=Ticket(
            ticket_id="T-2001",
            subject="Locked out after changing my email",
            body=(
                "I updated my email on the account last week and now I can't sign in "
                "at all. The reset emails aren't helping and I need this fixed today "
                "because I'm blocked from my dashboard. My name is Daniel and the "
                "workspace is for my team at Northwind."
            ),
            created_at="2026-03-25T09:40:00Z",
            channel="web_form",
            language="en",
        ),
        customer_profile=CustomerProfile(
            customer_id="CUST-305",
            full_name="Daniel Cho",
            tier="pro",
            region="US",
            tenure_days=189,
            sentiment_hint="urgent",
        ),
        account_context=AccountContext(
            product_area="auth",
            plan_status="active",
            last_payment_status="paid",
            renewal_date="2026-04-12",
            open_incidents=[],
            previous_tickets=[
                "Email change completed by self-serve flow 6 days ago.",
                "One prior password reset request in February.",
            ],
        ),
        policy_context=get_policies("priority_sla", "identity_verification"),
        allowed_queues=["general_support", "identity_and_access", "technical_support"],
        max_steps=6,
        expectation=TaskExpectation(
            expected_category="account_access",
            expected_priority="high",
            required_info_fields=[
                "account email",
                "workspace or company name",
                "last successful login date",
            ],
            valid_queues=["identity_and_access"],
            required_escalation_target=None,
            expected_resolution_code="waiting_for_verification",
            response_must_include=[
                "sorry",
                "account email",
                "workspace or company name",
                "last successful login date",
            ],
            response_must_not_include=[
                "we restored access",
                "password reset has been completed",
                "we confirmed you are the owner",
            ],
            required_response_any_of=[["verify", "verification"]],
            terminal_action="resolve_ticket",
        ),
        rubric=DEFAULT_RUBRIC,
    ),
    "enterprise_policy_hard": TaskSpec(
        task_id="enterprise_policy_hard",
        task_title="Enterprise Suspension Escalation Under Policy Constraints",
        difficulty="hard",
        description=(
            "An enterprise admin threatens churn after a suspended workspace blocks a "
            "launch, but reactivation requires compliance signoff."
        ),
        ticket=Ticket(
            ticket_id="T-3001",
            subject="Workspace suspended before launch, unacceptable",
            body=(
                "We are three days away from a major customer launch and your system "
                "suspended our workspace during procurement review. This is blocking "
                "our launch checklist and my leadership team is already asking if we "
                "need to move off your platform. Reactivate the workspace immediately "
                "and confirm when it's done. If this drags, we will escalate on our "
                "side and reconsider renewal."
            ),
            created_at="2026-03-25T10:05:00Z",
            channel="email",
            language="en",
        ),
        customer_profile=CustomerProfile(
            customer_id="CUST-901",
            full_name="Alicia Romero",
            tier="enterprise",
            region="US",
            tenure_days=905,
            sentiment_hint="angry",
        ),
        account_context=AccountContext(
            product_area="compliance",
            plan_status="suspended",
            last_payment_status="paid",
            renewal_date="2026-06-30",
            open_incidents=["Compliance review pending signoff for workspace WS-7781."],
            previous_tickets=[
                "Dedicated account manager assigned.",
                "Procurement review opened yesterday by compliance operations.",
            ],
        ),
        policy_context=get_policies(
            "priority_sla",
            "enterprise_escalation",
            "compliance_reactivation",
        ),
        allowed_queues=[
            "general_support",
            "enterprise_support",
            "billing_operations",
            "technical_support",
        ],
        max_steps=6,
        expectation=TaskExpectation(
            expected_category="enterprise_escalation",
            expected_priority="urgent",
            required_info_fields=["procurement or compliance ticket reference"],
            valid_queues=["enterprise_support"],
            required_escalation_target="account_management",
            expected_resolution_code="escalated_to_enterprise",
            response_must_include=[
                "business impact",
                "escalated",
                "enterprise",
            ],
            response_must_not_include=[
                "i have reactivated your workspace",
                "this will be fixed in the next few minutes",
                "we can bypass compliance",
            ],
            required_response_any_of=[
                ["compliance", "signoff"],
                ["next step", "follow up"],
            ],
            terminal_action="escalate_ticket",
        ),
        rubric=DEFAULT_RUBRIC,
    ),
}


def get_all_tasks() -> dict[str, TaskSpec]:
    return {task_id: task.model_copy(deep=True) for task_id, task in TASKS.items()}


def get_task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id].model_copy(deep=True)
    except KeyError as exc:
        available = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task_id '{task_id}'. Available tasks: {available}") from exc
