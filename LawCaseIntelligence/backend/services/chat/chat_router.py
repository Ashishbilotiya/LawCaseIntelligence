"""
backend/services/chat/chat_router.py
Main chat orchestrator — token-aware, memory-efficient routing.
"""
from __future__ import annotations

import logging
import re as _re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.services.llm.router import invoke_llm
from .query_classifier import classify_query, QueryCategory
from .query_rewriter import rewrite_query
from .response_formatter import format_response
from .memory_manager import build_memory_context, prune_and_summarize, should_summarize
from .topic_detector import should_clear_document_context
from .prompts import CONVERSATIONAL_SYSTEM, GENERAL_LEGAL_SYSTEM, DOCUMENT_SYSTEM
from .token_budget_manager import (
    validate_and_fit,
    estimate_tokens,
    RAG_CONTEXT_BUDGET,
    CHAT_HISTORY_BUDGET,
)
from rag.reranking.context_compressor import compress_chunks, build_context_string

logger = logging.getLogger(__name__)


# ── Doc name cleaner ──────────────────────────────────────────────

def _clean_doc_name(raw: str) -> str:
    """
    Strip UUID prefix from stored filenames so the UI shows clean names.
    '5eeeac0f-d55f-4ede-b3e3-41e1c3ae5b11_Dev_Dutt_vs_Union.pdf'
    → 'Dev_Dutt_vs_Union.pdf'
    """
    cleaned = _re.sub(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
        '', str(raw), flags=_re.IGNORECASE,
    )
    return cleaned or raw


# ── System prompts ────────────────────────────────────────────────
# Centralized in .prompts


# ── History → LangChain messages ─────────────────────────────────

def _to_lc_messages(history: List[Dict]) -> List:
    msgs = []
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content:
            continue
        msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    return msgs


# ── Route handlers ────────────────────────────────────────────────

def _conversational_answer(query: str, recent_history: List[Dict], summary: str) -> str:
    system        = CONVERSATIONAL_SYSTEM
    history_texts = [m.get("content", "") for m in recent_history]
    validated     = validate_and_fit(system, query, [], history_texts, summary)
    messages      = [SystemMessage(content=system)]
    if validated["summary"]:
        messages.append(SystemMessage(content=f"Conversation so far: {validated['summary']}"))
    messages += _to_lc_messages(recent_history[-len(validated["history_messages"]):])
    messages.append(HumanMessage(content=query))
    return invoke_llm(messages)


def _general_legal_answer(query: str, recent_history: List[Dict], summary: str) -> str:
    system        = GENERAL_LEGAL_SYSTEM
    history_texts = [m.get("content", "") for m in recent_history]
    validated     = validate_and_fit(system, query, [], history_texts, summary)
    messages      = [SystemMessage(content=system)]
    if validated["summary"]:
        messages.append(SystemMessage(content=f"Conversation context: {validated['summary']}"))
    kept = validated["history_messages"]
    messages += _to_lc_messages(recent_history[-len(kept):] if kept else [])
    messages.append(HumanMessage(content=query))
    return invoke_llm(messages)


def _document_answer(
    query: str,
    project_id: str,
    recent_history: List[Dict],
    summary: str,
) -> Dict[str, Any]:
    from rag.retrieval.hybrid_retriever import hybrid_search

    # Step 1 — Rewrite query
    search_queries = rewrite_query(query)
    logger.info(f"[ChatRouter] Search queries: {search_queries}")

    # Step 2 — Multi-query hybrid retrieval
    seen_texts: set = set()
    all_hits: List[Dict] = []
    for sq in search_queries:
        for hit in hybrid_search(sq, project_id=project_id, top_k=5):
            key = hit["text"][:120]
            if key not in seen_texts:
                seen_texts.add(key)
                all_hits.append(hit)

    if not all_hits:
        return {
            "answer":           "No documents found. Please upload and process judgments first.",
            "sources":          [],
            "retrieved_chunks": [],
            "search_queries":   search_queries,
        }

    # Step 3 — Compress to RAG token budget
    compressed = compress_chunks(all_hits, budget_tokens=RAG_CONTEXT_BUDGET)
    context    = build_context_string(compressed)

    # Build sources with clean display names (UUID prefix stripped)
    sources = []
    for chunk in compressed:
        meta     = chunk.get("metadata", {})
        raw_name = meta.get("source_doc", "")
        sources.append({
            "doc":      _clean_doc_name(raw_name),
            "doc_full": raw_name,
            "pages":    meta.get("page_nums", ""),
            "score":    round(chunk.get("hybrid_score", chunk.get("score", 0)), 3),
        })

    # Step 4 — Token budget validation
    history_texts = [m.get("content", "") for m in recent_history]
    validated     = validate_and_fit(DOCUMENT_SYSTEM, query, [context], history_texts, summary)

    # Step 5 — Build LLM messages
    messages = [SystemMessage(content=DOCUMENT_SYSTEM)]
    if validated["summary"]:
        messages.append(SystemMessage(content=f"Case discussion so far: {validated['summary']}"))
    kept = validated["history_messages"]
    messages += _to_lc_messages(recent_history[-len(kept):] if kept else [])

    alloc = validated["allocation"]
    logger.info(
        f"[ChatRouter] tokens — sys={alloc.system_tokens} query={alloc.query_tokens} "
        f"rag={alloc.rag_tokens} hist={alloc.history_tokens} total={alloc.total_input_tokens}"
    )

    messages.append(HumanMessage(
        content=f"Retrieved Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    ))

    try:
        answer = invoke_llm(messages)
    except Exception as e:
        logger.error(f"[ChatRouter] Document answer generation failed: {e}")
        answer = f"Error generating answer: {e}"

    return {
        "answer":           answer,
        "sources":          sources,
        "retrieved_chunks": [
            {"text": c["text"][:300], "score": c.get("hybrid_score", c.get("score", 0))}
            for c in compressed
        ],
        "search_queries": search_queries,
    }


# ── Public entry point ────────────────────────────────────────────

def chat_router(
    query: str,
    project_id: Optional[str] = None,
    chat_history: Optional[List[Dict]] = None,
    summary: str = "",
) -> Dict[str, Any]:
    chat_history = chat_history or []
    query        = query.strip()

    if not query:
        return format_response(answer="Please type a question.", category=QueryCategory.CONVERSATIONAL)

    if should_summarize(chat_history):
        chat_history, summary = prune_and_summarize(chat_history, summary)

    recent_history, summary = build_memory_context(chat_history, summary, budget_tokens=CHAT_HISTORY_BUDGET)

    classification = classify_query(query)
    category: QueryCategory = classification["category"]
    confidence: float       = classification["confidence"]

    logger.info(f"[ChatRouter] '{query[:60]}' → {category} ({confidence:.2f})")

    if category == QueryCategory.DOCUMENT_SPECIFIC:
        if should_clear_document_context(query, recent_history, project_id):
            logger.info("[ChatRouter] Topic change — clearing document context")
            recent_history, summary = [], ""

    try:
        if category == QueryCategory.CONVERSATIONAL:
            answer = _conversational_answer(query, recent_history, summary)
            return format_response(answer, category, confidence=confidence)

        elif category == QueryCategory.GENERAL_LEGAL:
            answer = _general_legal_answer(query, recent_history, summary)
            return format_response(answer, category, confidence=confidence)

        else:
            if not project_id:
                logger.warning("[ChatRouter] DOCUMENT_SPECIFIC but no project_id — GENERAL_LEGAL fallback")
                answer = _general_legal_answer(query, recent_history, summary)
                return format_response(answer, QueryCategory.GENERAL_LEGAL, confidence=0.6)

            result = _document_answer(query, project_id, recent_history, summary)
            return format_response(
                answer=result["answer"],
                category=category,
                sources=result.get("sources"),
                retrieved_chunks=result.get("retrieved_chunks"),
                search_queries=result.get("search_queries"),
                confidence=confidence,
            )

    except Exception as e:
        logger.error(f"[ChatRouter] Unhandled error: {e}")
        return format_response(answer=f"Something went wrong: {e}", category=category, confidence=0.0)
