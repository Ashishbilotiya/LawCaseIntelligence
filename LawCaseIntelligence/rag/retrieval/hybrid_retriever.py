"""
rag/retrieval/hybrid_retriever.py
Hybrid retrieval: semantic (vector) + keyword (BM25-style) with reranking.

Retrieves top 10 candidates, ranks by hybrid score.
Final context filtering is done by context_compressor using token budgets.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from rag.embeddings.embedding_factory import get_embeddings
from rag.vectordb.collection_manager import (
    project_collection_name,
    global_collection_name,
    get_or_create_collection,
)

logger = logging.getLogger(__name__)


def _keyword_score(query: str, text: str) -> float:
    """Simple TF-style keyword overlap score."""
    q_tokens = set(re.findall(r"\w+", query.lower()))
    t_tokens = re.findall(r"\w+", text.lower())
    if not q_tokens or not t_tokens:
        return 0.0
    matches = sum(1 for t in t_tokens if t in q_tokens)
    return matches / len(t_tokens)


def semantic_search(
    query: str,
    project_id: Optional[str] = None,
    top_k: int = 10,
    where: Optional[Dict] = None,
) -> List[Dict]:
    """Semantic search in ChromaDB. Returns {text, metadata, score} dicts."""
    emb       = get_embeddings()
    query_vec = emb.embed_query(query)

    col_name = project_collection_name(project_id) if project_id else global_collection_name()
    col      = get_or_create_collection(col_name)

    if col.count() == 0:
        logger.warning(f"Collection '{col_name}' is empty")
        return []

    kwargs: Dict = {
        "query_embeddings": [query_vec],
        "n_results":        min(top_k, col.count()),
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results   = col.query(**kwargs)
    docs      = results.get("documents",  [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]

    hits = []
    for doc, meta, dist in zip(docs, metas, distances):
        hits.append({"text": doc, "metadata": meta, "score": float(1 - dist)})
    return hits


def hybrid_search(
    query: str,
    project_id: Optional[str] = None,
    top_k: int = 10,
    alpha: float = 0.7,
) -> List[Dict]:
    """
    Hybrid retrieval: semantic + keyword reranking.
    Returns top_k candidates ranked by hybrid score.
    Token-based filtering happens downstream in context_compressor.
    """
    semantic_hits = semantic_search(query, project_id, top_k=top_k * 2)
    if not semantic_hits:
        return []

    for hit in semantic_hits:
        kw_score        = _keyword_score(query, hit["text"])
        hit["hybrid_score"] = alpha * hit["score"] + (1 - alpha) * kw_score

    semantic_hits.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return semantic_hits[:top_k]
