"""Conditional routing edges for the ClaimLens LangGraph workflow."""

from __future__ import annotations

from src.agents.state import ClaimWorkflowState


def route_after_audit(state: ClaimWorkflowState) -> str:
    """Evaluate audit results and recommendation decisions to route the claim.

    Routing Precedence & Invariants:
    1. 'siu_referral': Fraud triggers or high-risk criteria met (highest precedence).
    2. 'flagged_for_audit': Arithmetic invariants or domain compliance failed.
    3. 'manual_adjuster_review': Standard manual review threshold met, or claim is within deductible ($0 net indemnity).
    4. 'settlement_routing': Routine low-risk claim eligible for settlement disbursement.
    5. Fail-Closed Default: Any unknown or unhandled decision routes to 'manual_adjuster_review'.
    """
    recommendation = state.get("recommendation", {})
    judge_review = state.get("llm_as_judge_review", {})
    decision = str(recommendation.get("decision", "")).upper()
    verdict = str(judge_review.get("verdict", "")).upper()

    # 1. SIU referral takes top precedence
    if decision == "SIU_REFERRAL":
        return "siu_referral"

    # 2. Audit compliance flags escalate to supervisor review
    if verdict == "FLAGGED_FOR_AUDIT":
        return "flagged_for_audit"

    # 3. Explicit manual review or zero-indemnity deductible claims
    if decision in {"MANUAL_ADJUSTER_REVIEW", "CLAIM_WITHIN_DEDUCTIBLE"}:
        return "manual_adjuster_review"

    # 4. Standard routine settlements
    if decision in {"ROUTINE_REVIEW", "AUTO_APPROVE"}:
        return "settlement_routing"

    # 5. Fail-closed safety: unhandled decisions default to manual review
    return "manual_adjuster_review"

