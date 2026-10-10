"""Node 2: Hybrid Retrieval and ML Risk Scoring node."""

from __future__ import annotations

from typing import Any, Dict

from src.agents.state import ClaimWorkflowState
from src.agents.utils import normalize_claim_type, parse_number
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.ml_models import claims_ml_service


def retrieval_and_ml_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """Execute hybrid BM25 + dense retrieval and run ML risk scoring."""
    claim_data = state["claim_data"]
    query = state.get("query", "")
    top_k = state.get("top_k_similar", 5)
    claim_type = normalize_claim_type(claim_data)
    claim_state = claim_data.get("state")
    claim_amount = max(0.0, parse_number(claim_data.get("claim_amount")))

    min_amount_filter = None
    max_amount_filter = None
    if claim_amount >= 25_000.0:
        min_amount_filter = 12_000.0
    elif 0 < claim_amount <= 5_000.0:
        max_amount_filter = 8_000.0

    similar_claims = hybrid_retriever.search(
        query=query,
        top_k=top_k,
        claim_type=claim_type,
        state=claim_state,
        min_amount=min_amount_filter,
        max_amount=max_amount_filter,
    ) or []

    if len(similar_claims) < 2 and (min_amount_filter is not None or max_amount_filter is not None):
        similar_claims = hybrid_retriever.search(
            query=query,
            top_k=top_k,
            claim_type=claim_type,
            state=claim_state,
        ) or []

    ml_results = claims_ml_service.predict(claim_data) or {}

    path = list(state.get("execution_path", []))
    path.append("retrieval_and_ml")

    return {
        "similar_historical_claims": similar_claims,
        "ml_predictions": ml_results,
        "execution_path": path,
    }
