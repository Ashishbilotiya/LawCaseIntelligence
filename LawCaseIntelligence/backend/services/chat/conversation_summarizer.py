"""
backend/services/chat/conversation_summarizer.py
Thin public wrapper around memory_manager summarization.
Exported for use by flask_app.py session management.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .memory_manager import prune_and_summarize, should_summarize


def maybe_summarize(
    chat_history: List[Dict],
    existing_summary: str = "",
) -> Tuple[List[Dict], str]:
    """
    If history is long enough, summarize and prune.
    Otherwise return as-is.
    Returns (history, summary).
    """
    if should_summarize(chat_history):
        return prune_and_summarize(chat_history, existing_summary)
    return chat_history, existing_summary
