"""Node 1: Intake & Classification Agent node."""

from __future__ import annotations

from typing import Any, Dict

from src.agents.classification_agent import classification_agent
from src.agents.state import ClaimWorkflowState
from src.agents.utils import extract_claim_description, normalize_claim_type, parse_number


def intake_and_classification_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """Execute triage classification and build retrieval query."""
    claim_data = state["claim_data"]
    classification = classification_agent.classify_and_prioritize(claim_data)

    claim_type = normalize_claim_type(claim_data)
    claim_state = claim_data.get("state")
    claim_amount = max(0.0, parse_number(claim_data.get("claim_amount")))
    description = extract_claim_description(claim_data)

    if description:
        query = (
            f"{claim_type} insurance claim in {claim_state}: {description} "
            f"(loss amount ${claim_amount:,.2f})"
        )
    else:
        query = f"{claim_type} insurance claim in state {claim_state}, amount ${claim_amount:,.2f}"

    path = list(state.get("execution_path", []))
    path.append("intake_and_classification")

    return {
        "classification": classification,
        "query": query,
        "execution_path": path,
    }
