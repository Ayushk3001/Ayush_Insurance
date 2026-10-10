"""Terminal LangGraph routing nodes for triage escalation and queues."""

from __future__ import annotations

from typing import Any, Dict

from src.agents.state import ClaimWorkflowState


def siu_referral_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """Terminal Node: Routes claim to Special Investigation Unit review queue."""
    path = list(state.get("execution_path", []))
    path.append("siu_referral_terminal")
    return {
        "routing_destination": "SIU_INVESTIGATION_QUEUE",
        "workflow_status": "ROUTED_TO_SPECIAL_INVESTIGATION_UNIT",
        "execution_path": path,
    }


def manual_adjuster_review_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """Terminal Node: Routes claim to licensed adjuster worklist."""
    path = list(state.get("execution_path", []))
    path.append("manual_adjuster_review_terminal")
    return {
        "routing_destination": "ADJUSTER_WORKLIST",
        "workflow_status": "ROUTED_TO_MANUAL_ADJUSTER_REVIEW",
        "execution_path": path,
    }


def flagged_for_audit_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """Terminal Node: Escalates audit flagged claim to claims supervisor."""
    path = list(state.get("execution_path", []))
    path.append("audit_supervisor_terminal")
    return {
        "routing_destination": "SUPERVISOR_AUDIT_QUEUE",
        "workflow_status": "ESCALATED_DUE_TO_AUDIT_FLAGS",
        "execution_path": path,
    }


def settlement_routing_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """Terminal Node: Routes claim to routine settlement queue."""
    path = list(state.get("execution_path", []))
    path.append("settlement_terminal")
    return {
        "routing_destination": "STANDARD_SETTLEMENT_QUEUE",
        "workflow_status": "ROUTED_TO_STANDARD_SETTLEMENT",
        "execution_path": path,
    }
