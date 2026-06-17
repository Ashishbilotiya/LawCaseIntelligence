"""
backend/services/chat/query_rewriter.py
For DOCUMENT_SPECIFIC queries, expands the user's question into
2-3 concise legal search statements that improve RAG retrieval.

Example:
  Input : "What did the court say about negligence?"
  Output: [
    "judicial findings regarding negligence",
    "court observations on negligence liability",
    "negligence related legal reasoning in judgment",
  ]

Falls back gracefully to [original_query] on any failure.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.services.llm.router import invoke_llm
from .prompts import REWRITER_SYSTEM

logger = logging.getLogger(__name__)


def rewrite_query(query: str) -> List[str]:
    """
    Expand a document-specific query into 2-3 retrieval search phrases.
    Always returns at least [original_query].
    """
    query = query.strip()
    if not query:
        return [query]

    messages = [
        SystemMessage(content=REWRITER_SYSTEM),
        HumanMessage(content=f'User question: "{query}"'),
    ]

    try:
        raw = invoke_llm(messages)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        phrases: List[str] = json.loads(raw)

        if not isinstance(phrases, list) or not phrases:
            raise ValueError("Expected non-empty JSON array")

        # Sanitise: keep only non-empty strings, max 3
        phrases = [str(p).strip() for p in phrases if str(p).strip()][:3]
        logger.info(f"[QueryRewriter] Expanded to {len(phrases)} phrases: {phrases}")
        return phrases

    except Exception as e:
        logger.warning(f"[QueryRewriter] Failed ({e}), using original query")
        return [query]
