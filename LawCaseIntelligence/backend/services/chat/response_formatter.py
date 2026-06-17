"""
backend/services/chat/response_formatter.py
Formats the final response dict for each query category.

CONVERSATIONAL    → {answer, mode, badge}
GENERAL_LEGAL     → {answer, mode, badge, disclaimer}
DOCUMENT_SPECIFIC → {answer, mode, badge, sources, pages, retrieved_chunks}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from .query_classifier import QueryCategory
from backend.config.settings import get_settings

_settings = get_settings()


BADGES = {
    QueryCategory.CONVERSATIONAL:    {"label": "💬 General Chat",          "css": "badge-chat"},
    QueryCategory.GENERAL_LEGAL:     {"label": "⚖️ Legal Assistant",       "css": "badge-legal"},
    QueryCategory.DOCUMENT_SPECIFIC: {"label": "📄 Document Intelligence", "css": "badge-doc"},
}


def format_response(
    answer: str,
    category: QueryCategory,
    sources: Optional[List[Dict]] = None,
    retrieved_chunks: Optional[List[Dict]] = None,
    search_queries: Optional[List[str]] = None,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """
    Build a structured response dict appropriate for the query category.
    """
    badge = BADGES.get(category, BADGES[QueryCategory.DOCUMENT_SPECIFIC])

    base: Dict[str, Any] = {
        "answer":     answer,
        "mode":       category.value,
        "badge":      badge["label"],
        "badge_css":  badge["css"],
        "confidence": round(confidence, 2),
    }

    if category == QueryCategory.GENERAL_LEGAL:
        base["disclaimer"] = _settings.chat_legal_disclaimer

    if category == QueryCategory.DOCUMENT_SPECIFIC:
        base["sources"]          = sources or []
        base["pages"]            = _extract_pages(sources or [])
        base["retrieved_chunks"] = retrieved_chunks or []
        base["search_queries"]   = search_queries or []

    return base


def _extract_pages(sources: List[Dict]) -> List[str]:
    pages = []
    for s in sources:
        p = s.get("pages") or s.get("page_nums", "")
        if p and str(p) not in pages:
            pages.append(str(p))
    return pages
