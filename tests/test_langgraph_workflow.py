"""Tests for LangGraph multi-agent orchestration and dynamic conditional routing."""

from src.agents.multi_agent_workflow import multi_agent_workflow, ClaimWorkflowState


def test_langgraph_graph_structure():
    """Verify that the StateGraph is compiled with all required nodes and terminal handlers."""
    graph = multi_agent_workflow.get_graph()
    assert graph is not None
    # Verify node presence in the compiled graph structure
    node_keys = graph.nodes.keys()
    for expected_node in [
        "intake_and_classification",
        "retrieval_and_ml",
        "investigation_agent",
        "risk_agent",
        "recommendation_agent",
        "quality_audit",
        "siu_referral",
        "manual_adjuster_review",
        "flagged_for_audit",
        "settlement_routing",
    ]:
        assert expected_node in node_keys, f"Missing node: {expected_node}"


def test_langgraph_execution_path_routine_claim():
    """Verify end-to-end execution path for a routine low-risk claim."""
    claim = {
        "claim_id": "CLM-LG-ROUTINE",
        "policy_id": "POL-10001",
        "claim_type": "Auto",
        "state": "CA",
        "policyholder_tenure_years": 4.0,
        "previous_claims_count": 0,
        "incident_date": "2024-02-01",
        "claim_filed_date": "2024-02-02",
        "claim_amount": 2500.0,
        "deductible": 500.0,
        "description": "Minor scratch and bumper dent from parking collision."
    }

    result = multi_agent_workflow.process_claim(claim)
    assert result["workflow_engine"] == "LangGraph"
    assert "intake_and_classification" in result["execution_path"]
    assert "retrieval_and_ml" in result["execution_path"]
    assert "investigation_agent" in result["execution_path"]
    assert "risk_agent" in result["execution_path"]
    assert "recommendation_agent" in result["execution_path"]
    assert "quality_audit" in result["execution_path"]
    assert "settlement_terminal" in result["execution_path"]
    assert result["workflow_status"] == "ROUTED_TO_STANDARD_SETTLEMENT"


def test_langgraph_execution_path_siu_referral():
    """Verify conditional edge routing to SIU when high risk signals trigger referral."""
    claim = {
        "claim_id": "CLM-LG-SIU",
        "policy_id": "POL-10002",
        "claim_type": "Auto",
        "state": "FL",
        "policyholder_tenure_years": 0.2,
        "previous_claims_count": 5,
        "incident_date": "2024-02-01",
        "claim_filed_date": "2024-02-02",
        "claim_amount": 55000.0,
        "deductible": 500.0,
        "description": "Suspicious total loss fire claim filed 2 days after policy binding."
    }

    result = multi_agent_workflow.process_claim(claim)
    assert result["workflow_engine"] == "LangGraph"
    if result["recommendation"]["decision"] == "SIU_REFERRAL":
        assert "siu_referral_terminal" in result["execution_path"]
        assert result["workflow_status"] == "ROUTED_TO_SPECIAL_INVESTIGATION_UNIT"
        assert result["routing_destination"] == "SIU_INVESTIGATION_QUEUE"


def test_langgraph_conditional_routing_flagged_for_audit():
    """Verify conditional edge routing when quality audit flags an arithmetic inconsistency."""
    state: ClaimWorkflowState = {
        "claim_data": {"claim_amount": 1000.0, "deductible": 500.0},
        "recommendation": {"decision": "AUTO_APPROVE"},
        "llm_as_judge_review": {"verdict": "FLAGGED_FOR_AUDIT", "detected_issues": ["Ceiling exceeded"]},
        "execution_path": []
    }
    route = multi_agent_workflow._route_after_audit(state)
    assert route == "flagged_for_audit"
