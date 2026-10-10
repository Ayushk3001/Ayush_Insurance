"""ClaimLens Multi-Agent Workflow Orchestrator Facade.

This module exposes the unified MultiAgentClaimsWorkflow entrypoint, coordinating
agents via the underlying LangGraph StateGraph engine defined in `src.agents.graph`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.agents.graph import build_claims_graph
from src.agents.state import ClaimWorkflowState
from src.agents.edges import route_after_audit
from src.agents.nodes import (
    build_claim_checklist,
    run_investigation_agent,
    run_llm_as_judge,
    run_recommendation_agent,
    run_risk_agent,
    SIU_THRESHOLD_PERCENT,
    MANUAL_REVIEW_AMOUNT_THRESHOLD,
    HIGH_AMOUNT_SIU_THRESHOLD,
)
from src.agents.utils import (
    call_llm_completion,
    extract_claim_description,
    get_openai_client,
    normalize_claim_type,
    parse_fraud_percent,
    parse_number,
)
from src.utils.config import settings


class MultiAgentClaimsWorkflow:
    """Production facade orchestrating claims triage via compiled LangGraph StateGraph."""

    SIU_THRESHOLD_PERCENT = SIU_THRESHOLD_PERCENT
    MANUAL_REVIEW_AMOUNT_THRESHOLD = MANUAL_REVIEW_AMOUNT_THRESHOLD
    HIGH_AMOUNT_SIU_THRESHOLD = HIGH_AMOUNT_SIU_THRESHOLD

    def __init__(self) -> None:
        self.api_key = getattr(settings, "OPENAI_API_KEY", None)
        self.base_url = getattr(settings, "OPENAI_BASE_URL", None)
        self.model = getattr(settings, "OPENAI_MODEL", None)
        self.client: Optional[OpenAI] = get_openai_client()
        self._compiled_graph = build_claims_graph()

    # --- Backward-compatible utility delegates ---

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        return parse_number(value, default)

    @staticmethod
    def _claim_type(claim_data: Dict[str, Any]) -> str:
        return normalize_claim_type(claim_data)

    @staticmethod
    def _description(claim_data: Dict[str, Any]) -> str:
        return extract_claim_description(claim_data)

    @staticmethod
    def _fraud_percent(ml_results: Dict[str, Any]) -> Optional[float]:
        return parse_fraud_percent(ml_results)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        return call_llm_completion(system_prompt, user_prompt, client=self.client, model=self.model)

    def _build_claim_checklist(self, claim_data: Dict[str, Any], deductible: float) -> List[str]:
        return build_claim_checklist(claim_data, deductible)

    # --- Direct Agent Invocations (also used by unit tests) ---

    def run_investigation_agent(
        self,
        claim_data: Dict[str, Any],
        similar_claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize intake facts and retrieved precedents without inferring coverage."""
        return run_investigation_agent(claim_data, similar_claims)

    def run_risk_agent(
        self,
        claim_data: Dict[str, Any],
        ml_results: Dict[str, Any],
        investigation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Expose model outputs carefully; distinguish global importance from local reasons."""
        return run_risk_agent(claim_data, ml_results, investigation)

    def run_recommendation_agent(
        self,
        claim_data: Dict[str, Any],
        investigation: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        ml_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recommend a handling pathway with bounded amounts and no implied payment authorization."""
        return run_recommendation_agent(
            claim_data, investigation, risk_analysis, ml_results, client=self.client
        )

    def run_llm_as_judge(
        self,
        claim_data: Dict[str, Any],
        investigation: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        recommendation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deterministic audit checks plus optional LLM critique."""
        return run_llm_as_judge(
            claim_data, investigation, risk_analysis, recommendation, client=self.client
        )

    def _route_after_audit(self, state: ClaimWorkflowState) -> str:
        """Evaluate audit findings and route the claim through the StateGraph."""
        return route_after_audit(state)

    def get_graph(self):
        """Return the compiled LangGraph workflow instance for inspection or visualization."""
        if self._compiled_graph is None:
            self._compiled_graph = build_claims_graph()
        return self._compiled_graph

    # --- Core Pipeline Execution ---

    def process_claim(
        self, claim_data: Dict[str, Any], top_k_similar: int = 5
    ) -> Dict[str, Any]:
        """Execute the multi-agent claims pipeline via compiled LangGraph StateGraph."""
        if self._compiled_graph is None:
            self._compiled_graph = build_claims_graph()

        initial_state: ClaimWorkflowState = {
            "claim_data": claim_data,
            "top_k_similar": top_k_similar,
            "claim_id": str(claim_data.get("claim_id", "NEW_CLAIM")),
            "execution_path": [],
            "workflow_engine": "LangGraph StateGraph",
        }

        final_state = self._compiled_graph.invoke(initial_state)

        return {
            "claim_id": final_state.get("claim_id", claim_data.get("claim_id", "NEW_CLAIM")),
            "classification": final_state.get("classification", {}),
            "ml_predictions": final_state.get("ml_predictions", {}),
            "similar_historical_claims": final_state.get("similar_historical_claims", []),
            "investigation": final_state.get("investigation", {}),
            "risk_assessment": final_state.get("risk_assessment", {}),
            "recommendation": final_state.get("recommendation", {}),
            "llm_as_judge_review": final_state.get("llm_as_judge_review", {}),
            "workflow_engine": "LangGraph",
            "execution_path": final_state.get("execution_path", []),
            "workflow_status": final_state.get("workflow_status", "COMPLETED"),
            "routing_destination": final_state.get("routing_destination", "STANDARD_SETTLEMENT_QUEUE"),
        }


# Singleton export for application-wide workflow execution
multi_agent_workflow = MultiAgentClaimsWorkflow()

__all__ = ["MultiAgentClaimsWorkflow", "multi_agent_workflow", "ClaimWorkflowState"]
