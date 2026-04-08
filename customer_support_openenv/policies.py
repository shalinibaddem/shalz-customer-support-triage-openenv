from __future__ import annotations

from .models import PolicySnippet


POLICIES: dict[str, PolicySnippet] = {
    "refund_window": PolicySnippet(
        policy_id="refund_window",
        title="Refund Review Policy",
        content=(
            "Refunds for duplicate or erroneous charges must be reviewed by the "
            "billing operations team before confirmation to the customer. Support "
            "may acknowledge the issue and state that a refund review or request "
            "has been submitted, but must not claim the refund has already been "
            "processed unless billing has completed the action."
        ),
    ),
    "priority_sla": PolicySnippet(
        policy_id="priority_sla",
        title="Priority And SLA Guidance",
        content=(
            "Use normal priority for standard billing disputes with no active outage, "
            "high priority for account access problems blocking active users, and "
            "urgent priority for enterprise-impacting incidents tied to launches, "
            "contract risk, or business continuity."
        ),
    ),
    "identity_verification": PolicySnippet(
        policy_id="identity_verification",
        title="Identity Verification Requirements",
        content=(
            "Before changing login credentials, restoring access, or confirming "
            "account ownership, support must collect the account email, workspace "
            "or company name, and the last successful login date. Do not promise "
            "a password reset or restored access before verification is complete."
        ),
    ),
    "enterprise_escalation": PolicySnippet(
        policy_id="enterprise_escalation",
        title="Enterprise Escalation Policy",
        content=(
            "Enterprise customers facing launch-blocking or churn-risk events should "
            "be routed to enterprise support and escalated to account management. "
            "Support may acknowledge business impact and explain the escalation path, "
            "but cannot promise immediate reactivation without required approvals."
        ),
    ),
    "compliance_reactivation": PolicySnippet(
        policy_id="compliance_reactivation",
        title="Compliance Reactivation Restrictions",
        content=(
            "Workspaces suspended during procurement or compliance review cannot be "
            "reactivated until compliance signoff is documented. Support must request "
            "the relevant procurement or compliance ticket reference when missing and "
            "must not bypass the signoff requirement."
        ),
    ),
}


def get_policy(policy_id: str) -> PolicySnippet:
    return POLICIES[policy_id].model_copy(deep=True)


def get_policies(*policy_ids: str) -> list[PolicySnippet]:
    return [get_policy(policy_id) for policy_id in policy_ids]
