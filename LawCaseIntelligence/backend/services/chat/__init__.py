# backend/services/chat/__init__.py
from .query_classifier import classify_query, QueryCategory
from .query_rewriter import rewrite_query
from .chat_router import chat_router
from .response_formatter import format_response
from .token_budget_manager import estimate_tokens, validate_and_fit, fit_chunks_to_budget
from .memory_manager import build_memory_context, prune_and_summarize
from .topic_detector import detect_topic_change, should_clear_document_context


__all__ = [
    "classify_query", "QueryCategory",
    "rewrite_query",
    "chat_router",
    "format_response",
    "estimate_tokens", "validate_and_fit", "fit_chunks_to_budget",
    "build_memory_context", "prune_and_summarize",
    "detect_topic_change", "should_clear_document_context",
    "maybe_summarize",
]
