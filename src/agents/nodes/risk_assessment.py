"""Node 4: Risk Assessment Agent node and logic."""

from __future__ import annotations

from typing import Any, Dict

from src.agents.state import ClaimWorkflowState
from src.agents.utils import parse_fraud_percent


def run_risk_agent(
    claim_data: Dict[str, Any],
    ml_results: Dict[str, Any],
    investigation: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose model outputs carefully; distinguish global importance from local reasons."""
    fraud_probability = parse_fraud_percent(ml_results)
    risk_tier = ml_results.get("risk_tier")
    if not risk_tier:
        if fraud_probability is None:
            risk_tier = "Unknown"
        elif fraud_probability >= 65:
            risk_tier = "High Risk"
        elif fraud_probability >= 35:
            risk_tier = "Moderate Risk"
        else:
            risk_tier = "Low Risk"

    raw_drivers = ml_results.get("top_risk_drivers") or []
    drivers = [str(item) for item in raw_drivers if str(item).strip()]
    global_importance_notes = [
        driver for driver in drivers if "global feature importance" in driver.lower()
    ]
    case_specific_indicators = [
        driver for driver in drivers if "global feature importance" not in driver.lower()
    ]

    if fraud_probability is None:
        risk_narrative = (
            "The ML service did not provide a valid fraud probability. "
            "Do not infer a probability or automatically classify the claim as low risk."
        )
    else:
        risk_narrative = (
            f"The ML service reported a fraud probability of {fraud_probability:.1f}% "
            f"and tier '{risk_tier}'. This is a model estimate, not proof of fraud."
        )

    if case_specific_indicators:
        risk_narrative += " Case-level indicators: " + "; ".join(case_specific_indicators) + "."
    if global_importance_notes:
        risk_narrative += (
            " Model-wide feature-importance context (not case-specific explanations): "
            + "; ".join(global_importance_notes)
            + "."
        )

    return {
        "agent_name": "Risk Assessment Agent",
        "fraud_probability_percent": fraud_probability,
        "risk_tier": risk_tier,
        "risk_narrative": risk_narrative,
        "risk_drivers": drivers,
        "case_specific_indicators": case_specific_indicators,
        "global_feature_importance_notes": global_importance_notes,
    }


def risk_assessment_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """LangGraph node wrapping the Risk Assessment Agent."""
    claim_data = state["claim_data"]
    ml_results = state.get("ml_predictions", {})
    investigation = state.get("investigation", {})
    risk_analysis = run_risk_agent(claim_data, ml_results, investigation)

    path = list(state.get("execution_path", []))
    path.append("risk_agent")
    return {
        "risk_assessment": risk_analysis,
        "execution_path": path,
    }
