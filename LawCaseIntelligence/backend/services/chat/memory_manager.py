"""
backend/services/chat/memory_manager.py
Sliding-window conversation memory with auto-summarization.

Strategy:
  - Keep last MAX_RECENT_MESSAGES (6) turns in full
  - When total messages > SUMMARIZE_THRESHOLD (20), summarize older turns
  - Summary is stored in the session alongside history
  - Never send full history — only summary + recent messages
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from backend.config.settings import get_settings
from .token_budget_manager import estimate_tokens, CHAT_HISTORY_BUDGET, SUMMARY_BUDGET
from .prompts import SUMMARIZATION_SYSTEM

logger = logging.getLogger(__name__)

_settings = get_settings()
MAX_RECENT_MESSAGES   = _settings.chat_max_recent_messages
SUMMARIZE_THRESHOLD   = _settings.chat_summarize_threshold
MAX_SUMMARY_TOKENS    = _settings.chat_max_summary_tokens


def get_recent_messages(chat_history: List[Dict]) -> List[Dict]:
    """Return only the last MAX_RECENT_MESSAGES turns."""
    return chat_history[-MAX_RECENT_MESSAGES:]


def should_summarize(chat_history: List[Dict]) -> bool:
    return len(chat_history) > SUMMARIZE_THRESHOLD


def summarize_old_messages(
    old_messages: List[Dict],
    existing_summary: str = "",
) -> str:
    """
    Use LLM to summarize old conversation turns into a compact paragraph.
    Returns summary string (max ~300 tokens).
    """
    from backend.services.llm.router import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    if not old_messages:
        return existing_summary

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in old_messages
        if m.get("content", "").strip()
    )

    system = SUMMARIZATION_SYSTEM

    prefix = f"Existing summary: {existing_summary}\n\n" if existing_summary else ""
    prompt = f"{prefix}Conversation to summarize:\n{history_text}"

    try:
        summary = invoke_llm([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        # Hard-trim if LLM was verbose
        if estimate_tokens(summary) > MAX_SUMMARY_TOKENS:
            words = summary.split()
            trimmed = []
            used = 0
            for w in words:
                used += 1
                trimmed.append(w)
                if used >= MAX_SUMMARY_TOKENS * 3:  # rough char limit
                    break
            summary = " ".join(trimmed) + "…"

        logger.info(f"[MemoryManager] Summary generated ({estimate_tokens(summary)} tokens)")
        return summary.strip()

    except Exception as e:
        logger.warning(f"[MemoryManager] Summarization failed: {e}")
        return existing_summary


def prune_and_summarize(
    chat_history: List[Dict],
    existing_summary: str = "",
) -> Tuple[List[Dict], str]:
    """
    If history is long, summarize old messages and keep only recent ones.
    Returns (pruned_history, updated_summary).
    """
    if not should_summarize(chat_history):
        return chat_history, existing_summary

    recent   = chat_history[-MAX_RECENT_MESSAGES:]
    to_summarize = chat_history[:-MAX_RECENT_MESSAGES]

    logger.info(
        f"[MemoryManager] Summarizing {len(to_summarize)} old messages, "
        f"keeping {len(recent)} recent"
    )

    new_summary = summarize_old_messages(to_summarize, existing_summary)
    return recent, new_summary


def build_memory_context(
    chat_history: List[Dict],
    summary: str = "",
    budget_tokens: int = CHAT_HISTORY_BUDGET,
) -> Tuple[List[Dict], str]:
    """
    Given full history and summary, return what should actually be sent to LLM.
    Applies sliding window + token budget enforcement.

    Returns (recent_messages_to_send, summary_to_send)
    """
    recent = get_recent_messages(chat_history)

    # Fit recent messages into budget
    history_texts = [m.get("content", "") for m in recent]
    history_tokens = sum(estimate_tokens(t) for t in history_texts)

    summary_tokens = estimate_tokens(summary)
    total = history_tokens + summary_tokens

    if total <= budget_tokens:
        return recent, summary

    # Over budget — reduce history first, then summary
    while recent and (sum(estimate_tokens(m.get("content","")) for m in recent) + summary_tokens) > budget_tokens:
        recent.pop(0)

    # If still over, trim summary
    if summary and (sum(estimate_tokens(m.get("content","")) for m in recent) + summary_tokens) > budget_tokens:
        max_chars = max(50, (budget_tokens - sum(estimate_tokens(m.get("content","")) for m in recent)) * 4)
        summary = summary[:max_chars] + "…"

    return recent, summary
