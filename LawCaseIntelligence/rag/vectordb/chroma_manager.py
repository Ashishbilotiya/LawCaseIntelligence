"""
rag/vectordb/chroma_manager.py
ChromaDB vector store manager for legal document embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    logger.info(f"ChromaDB client ready at: {settings.chroma_persist_dir}")
    return client


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection by name."""
    client = get_chroma_client()
    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(f"Collection '{collection_name}': {col.count()} docs")
    return col


def add_documents(
    collection_name: str,
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    ids: List[str],
) -> None:
    """Add documents with pre-computed embeddings to a ChromaDB collection."""
    col = get_or_create_collection(collection_name)
    # Batch in groups of 100 to avoid memory issues
    batch = 100
    for i in range(0, len(texts), batch):
        col.add(
            documents=texts[i:i+batch],        # FIX: was incorrectly passing embeddings here
            embeddings=embeddings[i:i+batch],
            metadatas=metadatas[i:i+batch],
            ids=ids[i:i+batch],
        )
    logger.info(f"Added {len(texts)} chunks to '{collection_name}'")


def query_collection(
    collection_name: str,
    query_embedding: List[float],
    n_results: int = 5,
    where: Optional[Dict] = None,
) -> Dict:
    """Query a ChromaDB collection with a pre-computed embedding."""
    col = get_or_create_collection(collection_name)
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, col.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return col.query(**kwargs)


def delete_collection(collection_name: str) -> None:
    """Delete a ChromaDB collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection '{collection_name}': {e}")


def list_collections() -> List[str]:
    """List all collection names."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def collection_count(collection_name: str) -> int:
    """Return number of documents in a collection."""
    try:
        col = get_or_create_collection(collection_name)
        return col.count()
    except Exception:
        return 0
