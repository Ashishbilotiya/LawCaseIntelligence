"""
backend/services/chat/query_classifier.py
Classifies user queries into one of three modes:

  CONVERSATIONAL    — greetings, chit-chat, about-the-bot questions
  GENERAL_LEGAL     — Indian legal knowledge (acts, rights, procedures)
  DOCUMENT_SPECIFIC — questions about uploaded case documents

Uses a single lightweight LLM call with a strict JSON-only prompt.
Falls back to DOCUMENT_SPECIFIC on any parse failure (safest default).
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.llm.router import invoke_llm
from .prompts import CLASSIFIER_SYSTEM

logger = logging.getLogger(__name__)


class QueryCategory(str, Enum):
    CONVERSATIONAL    = "CONVERSATIONAL"
    GENERAL_LEGAL     = "GENERAL_LEGAL"
    DOCUMENT_SPECIFIC = "DOCUMENT_SPECIFIC"


# ── Fast rule-based pre-filter (saves one LLM call for obvious cases) ──────

_CONVERSATIONAL_PATTERNS = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s*(morning|evening|afternoon|night)|"
    r"how are you|who are you|what (can|do) you do|tell me a joke|"
    r"what('s| is) your name|are you (an? )?ai|thanks?|thank you|bye|goodbye|"
    r"okay|ok|sure|great|nice|cool|got it|understood)\s*[!?.]*\s*$",
    re.IGNORECASE,
)

def classify_query(query: str) -> Dict[str, object]:
    """
    Classify a user query. Returns:
        {"category": QueryCategory, "confidence": float}
    """
    query = query.strip()
    if not query:
        return {"category": QueryCategory.CONVERSATIONAL, "confidence": 1.0}

    # ── Fast path: obvious greetings ──────────────────────────────
    if _CONVERSATIONAL_PATTERNS.match(query):
        logger.debug(f"[Classifier] Fast-path CONVERSATIONAL: {query!r}")
        return {"category": QueryCategory.CONVERSATIONAL, "confidence": 0.99}

    # ── LLM classification ────────────────────────────────────────
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM),
        HumanMessage(content=f'Query: "{query}"'),
    ]

    try:
        raw = invoke_llm(messages)
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        data = json.loads(raw)

        cat_str = str(data.get("category", "")).upper().strip()
        cat = QueryCategory(cat_str) if cat_str in QueryCategory._value2member_map_ else QueryCategory.DOCUMENT_SPECIFIC
        confidence = float(data.get("confidence", 0.8))

        logger.info(f"[Classifier] '{query[:60]}' → {cat} ({confidence:.2f})")
        return {"category": cat, "confidence": confidence}

    except Exception as e:
        logger.warning(f"[Classifier] Parse failed ({e}), defaulting to DOCUMENT_SPECIFIC")
        return {"category": QueryCategory.DOCUMENT_SPECIFIC, "confidence": 0.5}
