"""LangGraph workflow state schema for the ClaimLens multi-agent system."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ClaimWorkflowState(TypedDict, total=False):
    """Typed state dictionary managed across the LangGraph multi-agent workflow.

    Each node in the state graph reads relevant upstream keys and returns updates
    to this shared state.
    """

    # --- Initial Intake Inputs ---
    claim_data: Dict[str, Any]
    top_k_similar: int
    claim_id: str
    query: str

    # --- Node Outputs ---
    classification: Dict[str, Any]
    similar_historical_claims: List[Dict[str, Any]]
    ml_predictions: Dict[str, Any]
    investigation: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendation: Dict[str, Any]
    llm_as_judge_review: Dict[str, Any]

    # --- Graph Routing & Observability Metadata ---
    routing_destination: str
    workflow_status: str
    execution_path: List[str]
    workflow_engine: str
