"""
backend/utils/json_utils.py
JSON parsing and serialization helpers for LLM outputs.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(text: str) -> Optional[Dict]:
    """
    Robustly extract JSON from an LLM response that may contain
    markdown fences, preamble text, or trailing explanation.
    Returns parsed dict or None on failure.
    """
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.debug(f"json_utils: JSON parse failed after extraction: {e}")

    # Try fixing common issues: trailing commas, single quotes
    fixed = re.sub(r",\s*([}\]])", r"\1", text)   # trailing commas
    fixed = fixed.replace("'", '"')                 # single → double quotes
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    logger.warning(f"json_utils: Could not extract JSON. First 200 chars: {text[:200]}")
    return None


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Serialize obj to JSON string, handling non-serializable types."""
    def default(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)
    return json.dumps(obj, indent=indent, default=default, ensure_ascii=False)


def merge_dicts(*dicts: Dict) -> Dict:
    """
    Deep merge multiple dicts. Later dicts override earlier ones.
    Lists are concatenated (deduplicated).
    """
    result: Dict = {}
    for d in dicts:
        if not d:
            continue
        for key, val in d.items():
            if key in result:
                if isinstance(result[key], list) and isinstance(val, list):
                    seen = {str(x) for x in result[key]}
                    result[key] = result[key] + [x for x in val if str(x) not in seen]
                elif isinstance(result[key], dict) and isinstance(val, dict):
                    result[key] = merge_dicts(result[key], val)
                elif val:  # non-empty wins
                    result[key] = val
            else:
                result[key] = val
    return result
