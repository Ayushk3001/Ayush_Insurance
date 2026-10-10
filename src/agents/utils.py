"""Common numeric, string, and LLM invocation helpers for ClaimLens agents."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from openai import OpenAI

from src.utils.config import settings


def get_openai_client() -> Optional[OpenAI]:
    """Initialize and return an OpenAI client if configured, otherwise None."""
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    base_url = getattr(settings, "OPENAI_BASE_URL", None)

    if api_key and not str(api_key).lower().startswith("your_"):
        try:
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            return OpenAI(**client_kwargs)
        except Exception:
            return None
    return None


def parse_number(value: Any, default: float = 0.0) -> float:
    """Convert value to a finite float; never propagate NaN or infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_claim_type(claim_data: Dict[str, Any]) -> str:
    """Normalize claim type string to standardized insurance line of business."""
    raw = str(claim_data.get("claim_type", "Auto") or "Auto").strip().lower()
    aliases = {
        "tenant": "Renters",
        "renter": "Renters",
        "renters": "Renters",
        "homeowners": "Home",
        "homeowner": "Home",
        "property": "Home",
        "home": "Home",
        "commercial": "Business",
        "business": "Business",
        "auto": "Auto",
        "vehicle": "Auto",
    }
    return aliases.get(raw, raw.title() or "Unknown")


def extract_claim_description(claim_data: Dict[str, Any]) -> str:
    """Extract incident narrative description across various field aliases."""
    for key in ("description", "incident_description", "incident_notes", "notes"):
        value = claim_data.get(key)
        if value:
            return str(value).strip()
    return ""


def parse_fraud_percent(ml_results: Dict[str, Any]) -> Optional[float]:
    """Return a validated fraud risk percentage (0..100) or None."""
    if ml_results.get("fraud_probability_percent") is not None:
        value = parse_number(ml_results.get("fraud_probability_percent"), float("nan"))
    elif ml_results.get("fraud_probability") is not None:
        value = parse_number(ml_results.get("fraud_probability"), float("nan")) * 100.0
    else:
        return None
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        return None
    return round(value, 2)


def call_llm_completion(
    system_prompt: str,
    user_prompt: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Execute LLM chat completion with fallback gracefully returning None."""
    client = client or get_openai_client()
    model = model or getattr(settings, "OPENAI_MODEL", None)
    if not client or not model:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # First attempt: Pass reasoning_effort='low' for reasoning models (e.g. gpt-5-nano)
    # to prevent internal reasoning tokens from exhausting the max_tokens quota.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            reasoning_effort="low",
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception:
        pass

    # Second attempt: Standard chat completion for standard non-reasoning models
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception:
        return None
