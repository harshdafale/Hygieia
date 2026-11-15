"""Validate action citations against retrieved NASA passages."""
from __future__ import annotations

import re
import json


def parse_json_object(text: str) -> dict:
    """Read a JSON object from a model response, including fenced responses."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    value = json.loads(text[start:end + 1])
    return value if isinstance(value, dict) else {}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_nasa_actions(items: object, sources: list[dict]) -> tuple[list[str], list[dict]]:
    """Keep only steps with an exact supporting excerpt from a numbered passage.

    The model must return {step, source, evidence_quote}; source is one-based and
    the quote must occur in that retrieved passage after whitespace normalization.
    """
    actions: list[str] = []
    evidence: list[dict] = []
    if not isinstance(items, list):
        return actions, evidence
    for item in items:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "")).strip()
        quote = str(item.get("evidence_quote", "")).strip()
        try:
            number = int(item.get("source", 0))
        except (TypeError, ValueError):
            continue
        if not step or len(quote) < 20 or not 1 <= number <= len(sources):
            continue
        passage = str(sources[number - 1].get("text", ""))
        if _normalize(quote) not in _normalize(passage):
            continue
        actions.append(f"{step} [{number}]")
        evidence.append({"source": number, "quote": quote})
    return actions, evidence
