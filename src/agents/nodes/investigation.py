"""Node 3: Investigation Agent node and logic."""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.state import ClaimWorkflowState
from src.agents.utils import normalize_claim_type, parse_number


def run_investigation_agent(
    claim_data: Dict[str, Any],
    similar_claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize intake facts and retrieved precedents without inferring coverage."""
    claim_id = claim_data.get("claim_id", "NEW_CLAIM")
    claim_type = normalize_claim_type(claim_data)
    amount = parse_number(claim_data.get("claim_amount"))
    tenure = parse_number(claim_data.get("policyholder_tenure_years"), 0.0)
    previous_claims = max(0, int(parse_number(claim_data.get("previous_claims_count"))))

    precedents: List[str] = []
    approved_count = 0
    denied_count = 0
    fraud_flagged_count = 0
    status_counts: Dict[str, int] = {}

    for claim in similar_claims or []:
        status = str(claim.get("claim_status", "unknown") or "unknown").strip().lower()
        status_counts[status] = status_counts.get(status, 0) + 1
        claim_amount = parse_number(claim.get("claim_amount"))
        precedents.append(
            f"Claim {claim.get('claim_id', 'unknown')}: "
            f"${claim_amount:,.2f} ({status})"
        )
        if status == "approved":
            approved_count += 1
        elif status == "denied":
            denied_count += 1
        flag = claim.get("is_fraud_flagged_ground_truth", False)
        if str(flag).strip().lower() in {"1", "true", "yes"} or flag is True:
            fraud_flagged_count += 1

    fraud_note = (
        f"; {fraud_flagged_count} with historical fraud flags"
        if fraud_flagged_count
        else ""
    )
    summary = (
        f"Claim {claim_id}: policy tenure {tenure:.1f} years, "
        f"{previous_claims} prior claims; {claim_type} claim for ${amount:,.2f}. "
        f"Retrieved {len(similar_claims or [])} historical records: "
        f"{approved_count} approved, {denied_count} denied{fraud_note}."
    )

    return {
        "agent_name": "Investigation Agent",
        "findings": summary,
        "similar_claims_count": len(similar_claims or []),
        "historical_approved_count": approved_count,
        "historical_denied_count": denied_count,
        "historical_fraud_count": fraud_flagged_count,
        "historical_status_counts": status_counts,
        "key_precedents": precedents[:3],
    }


def investigation_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """LangGraph node wrapping the Investigation Agent."""
    claim_data = state["claim_data"]
    similar_claims = state.get("similar_historical_claims", [])
    investigation = run_investigation_agent(claim_data, similar_claims)

    path = list(state.get("execution_path", []))
    path.append("investigation_agent")
    return {
        "investigation": investigation,
        "execution_path": path,
    }
