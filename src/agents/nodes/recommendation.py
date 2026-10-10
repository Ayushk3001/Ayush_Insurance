"""Node 5: Recommendation Agent node and logic."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.agents.state import ClaimWorkflowState
from src.agents.utils import (
    call_llm_completion,
    extract_claim_description,
    normalize_claim_type,
    parse_number,
)

SIU_THRESHOLD_PERCENT = 45.0
MANUAL_REVIEW_AMOUNT_THRESHOLD = 5_000.0
HIGH_AMOUNT_SIU_THRESHOLD = 50_000.0


def build_claim_checklist(claim_data: Dict[str, Any], deductible: float) -> List[str]:
    """Build incident-aware checklist steps."""
    claim_type = normalize_claim_type(claim_data)
    description = extract_claim_description(claim_data).lower()
    deductible_text = f"Confirm the ${deductible:,.2f} deductible and applicable policy terms."

    if claim_type == "Auto":
        if any(word in description for word in ("windshield", "glass", "stone struck", "cracked windshield")):
            return [
                "Review windshield photographs and the itemized auto-glass repair/replacement estimate.",
                "Confirm comprehensive coverage and whether a separate glass deductible or waiver applies.",
                "Check that the estimate addresses the reported glass damage and excludes unrelated damage.",
                deductible_text,
            ]
        if any(word in description for word in ("theft", "stolen", "vehicle was taken")):
            return [
                "Verify the police report, theft timeline, vehicle identification number, and ownership records.",
                "Confirm recovery status and document any recovered-vehicle damage or missing property.",
                "Review applicable comprehensive coverage, exclusions, and valuation requirements.",
                deductible_text,
            ]
        return [
            "Compare the itemized repair estimate with vehicle damage photographs and inspection findings.",
            "Verify the police report or driver-exchange details and the reported collision circumstances.",
            "Check for pre-existing or unrelated damage only where the evidence indicates a concern.",
            deductible_text,
        ]

    if claim_type in {"Renters", "Home"}:
        is_theft = any(word in description for word in ("theft", "stolen", "break-in", "burglary", "robbery"))
        is_water = any(word in description for word in ("water", "burst pipe", "plumb", "flood", "leak", "leaking"))
        is_fire = any(word in description for word in ("fire", "smoke", "burn"))
        is_weather = any(word in description for word in ("storm", "hail", "wind damage", "tornado", "hurricane"))

        if is_theft:
            steps = [
                "Verify the police report, incident timeline, and reported point of entry where applicable.",
                "Obtain an itemized inventory and match receipts, serial numbers, photographs, or alternative ownership evidence to claimed items.",
                "Check applicable personal-property coverage, item sublimits, scheduled-property endorsements, exclusions, and valuation rules.",
            ]
            if claim_type == "Renters":
                steps.append("Confirm whether the claim concerns tenant-owned contents and whether any building damage belongs to the landlord's policy.")
            else:
                steps.append("Separate stolen personal property from any claimed dwelling or structural damage.")
            steps.append(deductible_text)
            return steps

        if is_water:
            steps = [
                "Review damage photographs, the plumber or mitigation report, and the documented source of the water.",
                "Obtain an itemized inventory and repair estimates; match receipts or alternative ownership evidence to damaged belongings.",
                "Check policy provisions for the reported water source, applicable exclusions, mitigation duties, and personal-property or dwelling limits.",
            ]
            if claim_type == "Renters":
                steps.append("Confirm responsibility for tenant-owned contents versus landlord-owned building components, and request landlord maintenance records if relevant.")
            else:
                steps.append("Review reasonable emergency mitigation and structural repair invoices for damage supported by the evidence.")
            steps.append(deductible_text)
            return steps

        if is_fire:
            return [
                "Review fire department or incident reports and photographs documenting the affected areas.",
                "Obtain an itemized inventory and repair estimates, with ownership or valuation evidence for claimed contents.",
                "Check fire/smoke coverage, applicable exclusions, limits, and actual-cash-value or replacement-cost provisions.",
                deductible_text,
            ]

        if is_weather:
            return [
                "Compare photographs and itemized repair estimates with the reported storm-related damage.",
                "Verify the date and cause of loss using available incident, contractor, or weather documentation where relevant.",
                "Check applicable wind, hail, storm, water, and personal-property provisions, including exclusions and sublimits.",
                deductible_text,
            ]

        if claim_type == "Renters":
            return [
                "Obtain an itemized list of claimed personal property and supporting receipts, photographs, serial numbers, or alternative ownership evidence.",
                "Confirm renters personal-property coverage, applicable limits, exclusions, and valuation/depreciation rules.",
                "Request incident-specific documentation and repair or replacement estimates for the reported loss.",
                deductible_text,
            ]
        return [
            "Obtain itemized contractor estimates and photographs of the reported dwelling or personal-property loss.",
            "Verify ownership, cause of loss, and any supporting incident or mitigation documentation.",
            "Check applicable coverage limits, exclusions, endorsements, and valuation provisions.",
            deductible_text,
        ]

    if claim_type == "Business":
        return [
            "Obtain itemized repair/replacement estimates, relevant vendor invoices, and evidence of business-property ownership.",
            "Request business interruption or financial loss records only if that type of loss is being claimed.",
            "Check applicable commercial coverage, limits, exclusions, and valuation provisions.",
            deductible_text,
        ]

    return [
        "Obtain itemized estimates, photographs, and supporting proof of loss relevant to the reported incident.",
        "Verify ownership or insurable interest and the applicable coverage, limits, exclusions, and valuation provisions.",
        deductible_text,
    ]


def run_recommendation_agent(
    claim_data: Dict[str, Any],
    investigation: Dict[str, Any],
    risk_analysis: Dict[str, Any],
    ml_results: Dict[str, Any],
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Recommend a handling pathway with bounded amounts and no implied payment authorization."""
    amount = max(0.0, parse_number(claim_data.get("claim_amount")))
    deductible_value = claim_data.get("deductible")
    if deductible_value is None:
        deductible_value = claim_data.get("policy_deductible", 500.0)
    deductible = max(0.0, parse_number(deductible_value, 500.0))
    previous_claims = max(0, int(parse_number(claim_data.get("previous_claims_count"))))
    tenure = max(0.0, parse_number(claim_data.get("policyholder_tenure_years"), 0.0))
    fraud_probability = risk_analysis.get("fraud_probability_percent")
    risk_tier = str(risk_analysis.get("risk_tier", "Unknown"))
    max_net_payout = round(max(0.0, amount - deductible), 2)

    score_missing = fraud_probability is None
    fraud_score = parse_number(fraud_probability, -1.0)
    high_risk = risk_tier.lower() == "high risk"
    moderate_risk = risk_tier.lower() == "moderate risk"

    siu_trigger = (
        (not score_missing and fraud_score >= SIU_THRESHOLD_PERCENT)
        or high_risk
        or (
            amount >= HIGH_AMOUNT_SIU_THRESHOLD
            and not score_missing
            and fraud_score >= 40.0
            and risk_tier.lower() != "low risk"
        )
    )
    if siu_trigger:
        decision = "SIU_REFERRAL"
        action = "Refer for specialist investigation based on the configured risk trigger; the referral is not a finding of fraud."
        fast_track = False
        recommended_net = 0.0
        authorized_net = 0.0
        auth_status = "DISBURSEMENT_WITHHELD_PENDING_SIU_REVIEW"
        payout_range = (
            f"$0.00 authorized while SIU review is pending "
            f"(simplified net ceiling: ${max_net_payout:,.2f})"
        )
        steps = [
            "Document the specific rule or model signal that triggered referral and preserve supporting evidence.",
            "Request only claim-relevant proof of loss and incident documentation; explain outstanding requirements to the policyholder.",
            "Keep disbursement on hold pending authorized review; do not treat referral alone as proof of fraud.",
        ]
    elif amount <= deductible:
        decision = "CLAIM_WITHIN_DEDUCTIBLE"
        action = (
            f"The claimed amount (${amount:,.2f}) does not exceed the entered deductible "
            f"(${deductible:,.2f}); the simplified net indemnity is $0.00, subject to policy verification."
        )
        fast_track = False
        recommended_net = 0.0
        authorized_net = 0.0
        auth_status = "ZERO_NET_INDEMNITY_SUBJECT_TO_POLICY_VERIFICATION"
        payout_range = "$0.00 (claimed amount is within the entered deductible)"
        steps = [
            "Verify the estimate and confirm that the entered deductible applies to this coverage and loss type.",
            "Explain the calculated zero net amount to the policyholder, subject to policy terms and any applicable special deductible.",
        ]
    else:
        manual_review = (
            amount > MANUAL_REVIEW_AMOUNT_THRESHOLD
            or previous_claims >= 2
            or tenure < 1.0
            or moderate_risk
            or high_risk
            or (not score_missing and fraud_score >= 38.0)
        )
        if manual_review:
            decision = "MANUAL_ADJUSTER_REVIEW"
            action = "Route to a claims adjuster to verify evidence, applicable coverage, and the supported loss amount."
            fast_track = False
            auth_status = "PENDING_ADJUSTER_REVIEW"
            recommended_net = max_net_payout
            authorized_net = 0.0
            payout_range = (
                f"Pending itemized loss and policy verification; "
                f"simplified net ceiling: ${max_net_payout:,.2f}"
            )
            steps = build_claim_checklist(claim_data, deductible)
        else:
            decision = "ROUTINE_REVIEW"
            action = "Proceed through routine claim validation before any settlement authorization."
            fast_track = True
            auth_status = "PENDING_ROUTINE_COVERAGE_VALIDATION"
            recommended_net = max_net_payout
            authorized_net = 0.0
            payout_range = (
                f"Pending coverage and estimate validation; "
                f"simplified net ceiling: ${max_net_payout:,.2f}"
            )
            steps = build_claim_checklist(claim_data, deductible)

    llm_reasoning = None
    system_prompt = (
        "You are an insurance claims triage assistant. Do not decide coverage, "
        "invent payout estimates, or state that payment is authorized. Explain "
        "the recommended handling pathway using only supplied facts."
    )
    user_prompt = (
        f"Claim data: {json.dumps(claim_data, default=str)}\n"
        f"Investigation: {investigation.get('findings', '')}\n"
        f"Risk analysis: {risk_analysis.get('risk_narrative', '')}\n"
        f"Decision: {decision}\n"
        f"Payout note: {payout_range}\n"
        f"Checklist: {json.dumps(steps)}\n"
        "Write a concise two-sentence rationale. Do not introduce facts or policy terms not provided."
    )
    llm_reasoning = call_llm_completion(system_prompt, user_prompt, client=client)

    fallback_rationale = (
        f"{decision}: {action} The calculated amount is a simplified ceiling only; "
        "coverage, limits, exclusions, valuation, and policy terms require verification."
    )
    return {
        "agent_name": "Recommendation Agent",
        "decision": decision,
        "action_statement": action,
        "fast_track_eligible": fast_track,
        "net_settlement_ceiling": max_net_payout,
        "recommended_net_payout": recommended_net,
        "authorized_net_payout": authorized_net,
        "payment_authorization_status": auth_status,
        "recommended_payout": payout_range,
        "actionable_steps": steps,
        "executive_rationale": llm_reasoning or fallback_rationale,
    }


def recommendation_node(state: ClaimWorkflowState) -> Dict[str, Any]:
    """LangGraph node wrapping the Recommendation Agent."""
    claim_data = state["claim_data"]
    investigation = state.get("investigation", {})
    risk_analysis = state.get("risk_assessment", {})
    ml_results = state.get("ml_predictions", {})
    recommendation = run_recommendation_agent(
        claim_data, investigation, risk_analysis, ml_results
    )

    path = list(state.get("execution_path", []))
    path.append("recommendation_agent")
    return {
        "recommendation": recommendation,
        "execution_path": path,
    }
