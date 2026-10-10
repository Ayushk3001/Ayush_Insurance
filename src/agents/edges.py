"""Conditional routing edges for the ClaimLens LangGraph workflow."""

from __future__ import annotations

from src.agents.state import ClaimWorkflowState


def route_after_audit(state: ClaimWorkflowState) -> str:
    """Evaluate audit results and recommendation decisions to route the claim.

    Returns one of:
    - 'siu_referral': Fraud triggers or high-risk criteria met
    - 'flagged_for_audit': Arithmetic invariants or domain compliance failed
    - 'manual_adjuster_review': Standard manual review threshold met
    - 'settlement_routing': Routine low-risk claim eligible for settlement
    """
    recommendation = state.get("recommendation", {})
    judge_review = state.get("llm_as_judge_review", {})
    decision = str(recommendation.get("decision", "")).upper()
    verdict = str(judge_review.get("verdict", "")).upper()

    if decision == "SIU_REFERRAL":
        return "siu_referral"
    if verdict == "FLAGGED_FOR_AUDIT":
        return "flagged_for_audit"
    if decision == "MANUAL_ADJUSTER_REVIEW":
        return "manual_adjuster_review"
    return "settlement_routing"
