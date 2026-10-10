"""LangGraph StateGraph builder and compiler for ClaimLens."""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.agents.state import ClaimWorkflowState
from src.agents.edges import route_after_audit
from src.agents.nodes import (
    intake_and_classification_node,
    retrieval_and_ml_node,
    investigation_node,
    risk_assessment_node,
    recommendation_node,
    quality_audit_node,
    siu_referral_node,
    manual_adjuster_review_node,
    flagged_for_audit_node,
    settlement_routing_node,
)


def build_claims_graph():
    """Build and compile the multi-agent claims workflow StateGraph."""
    workflow = StateGraph(ClaimWorkflowState)

    # 1. Register Core Pipeline Nodes
    workflow.add_node("intake_and_classification", intake_and_classification_node)
    workflow.add_node("retrieval_and_ml", retrieval_and_ml_node)
    workflow.add_node("investigation_agent", investigation_node)
    workflow.add_node("risk_agent", risk_assessment_node)
    workflow.add_node("recommendation_agent", recommendation_node)
    workflow.add_node("quality_audit", quality_audit_node)

    # 2. Register Terminal Routing Nodes
    workflow.add_node("siu_referral", siu_referral_node)
    workflow.add_node("manual_adjuster_review", manual_adjuster_review_node)
    workflow.add_node("flagged_for_audit", flagged_for_audit_node)
    workflow.add_node("settlement_routing", settlement_routing_node)

    # 3. Core Sequential Edges
    workflow.add_edge(START, "intake_and_classification")
    workflow.add_edge("intake_and_classification", "retrieval_and_ml")
    workflow.add_edge("retrieval_and_ml", "investigation_agent")
    workflow.add_edge("investigation_agent", "risk_agent")
    workflow.add_edge("risk_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", "quality_audit")

    # 4. Conditional Branching from Quality Audit
    workflow.add_conditional_edges(
        "quality_audit",
        route_after_audit,
        {
            "siu_referral": "siu_referral",
            "flagged_for_audit": "flagged_for_audit",
            "manual_adjuster_review": "manual_adjuster_review",
            "settlement_routing": "settlement_routing",
        },
    )

    # 5. Connect Terminals to END
    workflow.add_edge("siu_referral", END)
    workflow.add_edge("manual_adjuster_review", END)
    workflow.add_edge("flagged_for_audit", END)
    workflow.add_edge("settlement_routing", END)

    return workflow.compile()
