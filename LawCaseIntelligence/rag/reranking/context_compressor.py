"""
rag/reranking/context_compressor.py
Compresses retrieved legal chunks to fit the RAG token budget.
Doc names have UUID prefix stripped for clean display in answers and UI.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from backend.services.chat.token_budget_manager import (
    estimate_tokens,
    fit_chunks_to_budget,
    RAG_CONTEXT_BUDGET,
)

logger = logging.getLogger(__name__)

COMPRESS_THRESHOLD = RAG_CONTEXT_BUDGET
TARGET_TOKENS      = int(RAG_CONTEXT_BUDGET * 0.85)


def _clean_doc_name(raw: str) -> str:
    """
    Strip UUID prefix from stored filenames for clean display.
    '5eeeac0f-d55f-4ede-b3e3-41e1c3ae5b11_Dev_Dutt_vs_Union.pdf'
    → 'Dev_Dutt_vs_Union.pdf'
    """
    cleaned = re.sub(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
        '', str(raw), flags=re.IGNORECASE,
    )
    return cleaned or raw


def compress_chunks(
    chunks: List[Dict],
    budget_tokens: int = RAG_CONTEXT_BUDGET,
    use_llm_compression: bool = False,
) -> List[Dict]:
    """
    Step 1: Token-based selection — add highest-ranked chunks until budget reached.
    Step 2 (optional): LLM compression if still over budget.
    """
    if not chunks:
        return []

    ranked = sorted(
        chunks,
        key=lambda c: c.get("hybrid_score", c.get("score", 0)),
        reverse=True,
    )

    selected, used = [], 0
    for chunk in ranked:
        t = estimate_tokens(chunk["text"])
        if used + t > budget_tokens:
            break
        selected.append(chunk)
        used += t

    logger.info(
        f"[ContextCompressor] {len(chunks)} chunks → {len(selected)} selected "
        f"({used}/{budget_tokens} tokens)"
    )

    if used <= budget_tokens:
        return selected

    if use_llm_compression and selected:
        return _llm_compress(selected, budget_tokens)

    return selected


def _llm_compress(chunks: List[Dict], budget_tokens: int) -> List[Dict]:
    from backend.services.llm.router import invoke_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You are a legal text compressor. "
        "Compress the following legal text to roughly half its length "
        "while preserving ALL legal facts, holdings, citations, dates, "
        "party names, and statutory references. "
        "Output ONLY the compressed text. No commentary."
    )

    compressed, used = [], 0
    for chunk in chunks:
        if used >= budget_tokens:
            break
        t         = estimate_tokens(chunk["text"])
        remaining = budget_tokens - used

        if t > remaining and t > 200:
            try:
                compressed_text = invoke_llm([
                    SystemMessage(content=system),
                    HumanMessage(content=chunk["text"]),
                ])
                new_chunk = dict(chunk)
                new_chunk["text"] = compressed_text.strip()
                new_t = estimate_tokens(new_chunk["text"])
                if used + new_t <= budget_tokens:
                    compressed.append(new_chunk)
                    used += new_t
            except Exception as e:
                logger.warning(f"[ContextCompressor] LLM compression failed: {e}")
                max_chars = remaining * 4
                new_chunk = dict(chunk)
                new_chunk["text"] = chunk["text"][:max_chars] + "…"
                compressed.append(new_chunk)
                used += estimate_tokens(new_chunk["text"])
        else:
            if used + t <= budget_tokens:
                compressed.append(chunk)
                used += t

    return compressed


def build_context_string(chunks: List[Dict]) -> str:
    """
    Convert chunk list into a numbered context string for the LLM prompt.
    UUID prefix is stripped from doc names for clean citations.
    Format: [1] DocName (p.X)\ntext\n\n---\n\n[2] ...
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta     = chunk.get("metadata", {})
        raw_name = meta.get("source_doc", "Document")
        src      = f"{_clean_doc_name(raw_name)} (p.{meta.get('page_nums', '?')})"
        parts.append(f"[{i}] {src}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
