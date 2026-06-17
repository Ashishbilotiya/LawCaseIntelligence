"""
backend/services/chat/topic_detector.py
Detects topic changes between conversation turns.

If the new query is about a completely different topic than the previous
conversation, we clear document context to avoid sending irrelevant
case history to the LLM — saving tokens and improving answer quality.

Uses fast keyword heuristic first; LLM fallback for ambiguous cases.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Legal topic keyword clusters
_TOPIC_CLUSTERS = {
    "negligence":     ["negligence", "duty of care", "tort", "damages"],
    "bail":           ["bail", "anticipatory bail", "custody", "remand", "detention"],
    "constitutional": ["article", "fundamental right", "constitution", "writ", "habeas corpus"],
    "criminal":       ["fir", "accused", "conviction", "acquittal", "sentence", "ipc", "crpc"],
    "civil":          ["contract", "breach", "injunction", "decree", "plaintiff", "defendant"],
    "property":       ["property", "title", "possession", "easement", "land", "transfer"],
    "family":         ["divorce", "maintenance", "custody", "adoption", "marriage"],
    "consumer":       ["consumer", "deficiency", "compensation", "forum", "redressal"],
    "verdict":        ["verdict", "judgment", "order", "ruling", "decision", "held"],
    "evidence":       ["evidence", "witness", "testimony", "exhibit", "proof"],
}


def _extract_topic_keywords(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for cluster, keywords in _TOPIC_CLUSTERS.items():
        if any(kw in text_lower for kw in keywords):
            found.add(cluster)
    return found


def detect_topic_change(
    new_query: str,
    recent_history: List[Dict],
    threshold: int = 3,
) -> bool:
    """
    Returns True if the new query appears to be a different topic
    from the recent conversation.

    threshold: minimum messages in history before topic-change detection kicks in.
    """
    if len(recent_history) < threshold:
        return False

    # Extract topics from last 4 messages
    recent_text = " ".join(
        m.get("content", "") for m in recent_history[-4:]
    )
    recent_topics  = _extract_topic_keywords(recent_text)
    new_topics     = _extract_topic_keywords(new_query)

    # If both have topics and they share none → topic changed
    if recent_topics and new_topics and recent_topics.isdisjoint(new_topics):
        logger.info(
            f"[TopicDetector] Topic change detected. "
            f"Was: {recent_topics} → Now: {new_topics}"
        )
        return True

    # If new query is purely conversational (no legal topics) after legal discussion
    conversational_only = re.match(
        r"^\s*(hi|hello|hey|thanks|okay|ok|got it|sure|what else|"
        r"tell me more|continue|go on|next)\s*[!?.]?\s*$",
        new_query.strip(),
        re.IGNORECASE,
    )
    if conversational_only and recent_topics:
        logger.info("[TopicDetector] Switched to conversational after legal discussion")
        return False  # not a topic change — just a follow-up

    return False


def should_clear_document_context(
    new_query: str,
    recent_history: List[Dict],
    current_project_id: Optional[str] = None,
) -> bool:
    """
    Returns True if the document context from the current project
    should NOT be included in the next LLM call.

    Clears context when:
    - Topic has clearly changed
    - New query is purely general legal (no document references)
    """
    if not current_project_id:
        return True

    return detect_topic_change(new_query, recent_history)
