"""Node 6: LLM-as-Judge & Quality Audit node and invariant rules."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.agents.state import ClaimWorkflowState
from src.agents.utils import (
    call_llm_completion,
    extract_claim_description,
    normalize_claim_type,
    parse_number,
)


def run_llm_as_judge(
    claim_data: Dict[str, Any],
    investigation: Dict[str, Any],
    risk_analysis: Dict[str, Any],
    recommendation: Dict[str, Any],
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Deterministic audit checks plus optional LLM critique."""
    amount = max(0.0, parse_number(claim_data.get("claim_amount")))
    deductible_value = claim_data.get("deductible")
    if deductible_value is None:
        deductible_value = claim_data.get("policy_deductible", 500.0)
    deductible = max(0.0, parse_number(deductible_value, 500.0))
    ceiling = round(max(0.0, amount - deductible), 2)
    decision = str(recommendation.get("decision", ""))
    issues: List[str] = []
    factual_score = 10.0
    completeness_score = 10.0
    compliance_score = 10.0

    # Financial constraints.
    authorized = parse_number(recommendation.get("authorized_net_payout"), 0.0)
    recommended = parse_number(recommendation.get("recommended_net_payout"), 0.0)
    if authorized < 0 or authorized > ceiling + 0.01:
        factual_score -= 5.0
        compliance_score -= 6.0
        issues.append("Authorized payout is negative or exceeds the simplified net ceiling.")
    if recommended < 0 or recommended > ceiling + 0.01:
        factual_score -= 4.0
        compliance_score -= 5.0
        issues.append("Recommended payout is negative or exceeds the simplified net ceiling.")
    if amount <= deductible and (authorized > 0 or recommended > 0):
        factual_score -= 4.0
        compliance_score -= 5.0
        issues.append("Claim is within the entered deductible but a positive payout was recommended or authorized.")
    if decision == "SIU_REFERRAL" and authorized > 0:
        factual_score -= 5.0
        compliance_score -= 6.0
        issues.append("SIU referral must not authorize disbursement while review is pending.")

    # Checklist relevance guardrails.
    claim_type = normalize_claim_type(claim_data)
    description = extract_claim_description(claim_data).lower()
    checklist = [str(step) for step in recommendation.get("actionable_steps", [])]
    checklist_text = " ".join(checklist).lower()
    is_theft = any(word in description for word in ("theft", "stolen", "break-in", "burglary", "robbery"))
    is_water = any(word in description for word in ("water", "burst pipe", "plumb", "flood", "leak", "leaking"))
    is_glass = any(word in description for word in ("windshield", "glass", "stone struck", "cracked windshield"))
    is_auto = claim_type == "Auto"
    is_renters_or_home = claim_type in {"Renters", "Home"}

    irrelevant_patterns: List[Tuple[bool, str]] = [
        (is_renters_or_home and is_theft and any(w in checklist_text for w in ("business interruption", "commercial casualty", "water mitigation", "plumber repair invoice")),
         "Checklist contains commercial or water-damage steps for a theft claim."),
        (is_renters_or_home and is_water and any(w in checklist_text for w in ("business interruption", "commercial casualty", "driver exchange", "body shop repair")),
         "Checklist contains unrelated commercial or auto-collision steps for a water-damage claim."),
        (is_auto and is_glass and any(w in checklist_text for w in ("dwelling policy", "landlord", "business interruption", "body shop collision")),
         "Checklist contains steps unrelated to an auto-glass claim."),
        (is_auto and any(w in checklist_text for w in ("dwelling policy", "landlord maintenance", "commercial casualty", "business interruption")),
         "Checklist contains steps unrelated to an auto claim."),
        (is_renters_or_home and any(w in checklist_text for w in ("commercial casualty", "business interruption logs", "cpa-verified commercial")),
         "Checklist contains commercial-claims requirements for a personal-lines claim."),
    ]
    for condition, message in irrelevant_patterns:
        if condition and message not in issues:
            issues.append(message)
            completeness_score -= 3.0

    if not checklist:
        completeness_score -= 4.0
        issues.append("Recommendation has no actionable checklist.")
    if not recommendation.get("action_statement"):
        completeness_score -= 2.0
        issues.append("Recommendation is missing an action statement.")

    # The system should not label global feature importance as an individual cause.
    drivers = risk_analysis.get("risk_drivers", []) or []
    if any("global feature importance" in str(driver).lower() for driver in drivers):
        narrative = str(risk_analysis.get("risk_narrative", "")).lower()
        if "not case-specific" not in narrative and "model-wide" not in narrative:
            issues.append("Global feature importance is presented without clarifying that it is not a case-specific explanation.")
            completeness_score -= 1.5

    factual_score = max(0.0, min(10.0, factual_score))
    completeness_score = max(0.0, min(10.0, completeness_score))
    compliance_score = max(0.0, min(10.0, compliance_score))
    overall_score = round((factual_score + completeness_score + compliance_score) / 3.0, 1)
    verdict = "PASS" if overall_score >= 8.0 and not issues else "FLAGGED_FOR_AUDIT"

    critique = (
        "Deterministic validation passed the configured checks. Coverage and payment still require authorized policy review."
        if not issues
        else "Audit warning: " + "; ".join(issues)
    )
    llm_critique = None
    judge_system = (
        "You are an independent quality auditor for an insurance claim triage assistant. "
        "Check incident-specific checklist relevance, financial arithmetic, unsupported assumptions, "
        "and whether model-wide feature importance is misrepresented as a local explanation. "
        "Do not infer coverage from incomplete policy information."
    )
    judge_prompt = (
        f"Claim data: {json.dumps(claim_data, default=str)}\n"
        f"Investigation: {json.dumps(investigation, default=str)}\n"
        f"Risk analysis: {json.dumps(risk_analysis, default=str)}\n"
        f"Recommendation: {json.dumps(recommendation, default=str)}\n"
        f"Deterministic issues: {json.dumps(issues)}\n"
        "Return a concise critique, specifically identifying any irrelevant checklist items."
    )
    llm_critique = call_llm_completion(judge_system, judge_prompt, client=client)
    if llm_critique:
        critique = llm_critique

    return {
        "judge_name": "Claims Quality Audit",
        "judge_mode": "rules_plus_llm" if llm_critique else "deterministic_rules_only",
        "verdict": verdict,
        "overall_quality_score": overall_score,
        "metric_scores": {
            "factual_consistency": round(factual_score, 1),
            "completeness": round(completeness_score, 1),
            "policy_compliance": round(compliance_score, 1),
        },
        "audit_critique": critique,
        "detected_issues": issues,
    }


def quality_audit_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """LangGraph node wrapping the LLM-as-Judge & Quality Audit."""
    claim_data = state["claim_data"]
    investigation = state.get("investigation", {})
    risk_analysis = state.get("risk_assessment", {})
    recommendation = state.get("recommendation", {})
    judge_review = run_llm_as_judge(
        claim_data, investigation, risk_analysis, recommendation
    )

    path = list(state.get("execution_path", []))
    path.append("quality_audit")
    return {
        "llm_as_judge_review": judge_review,
        "execution_path": path,
    }
