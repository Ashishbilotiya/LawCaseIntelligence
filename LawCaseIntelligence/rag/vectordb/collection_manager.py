"""
rag/vectordb/collection_manager.py
High-level collection management — per-project and global collections.

delete_document_chunks uses two strategies:
  1. Filter by doc_id metadata (for new documents ingested with doc_id)
  2. Filter by source_doc filename (for old documents ingested without doc_id)
  3. Delete from both project AND global collections
"""
from __future__ import annotations

import logging
from typing import List, Optional

from backend.config.settings import get_settings
from rag.vectordb.chroma_manager import (
    get_or_create_collection,
    delete_collection,
    collection_count,
    list_collections,
)

logger = logging.getLogger(__name__)


def project_collection_name(project_id: str) -> str:
    settings = get_settings()
    return f"{settings.chroma_collection_prefix}{project_id.replace('-', '_')}"


def global_collection_name() -> str:
    settings = get_settings()
    return f"{settings.chroma_collection_prefix}global"


def get_project_collection(project_id: str):
    return get_or_create_collection(project_collection_name(project_id))


def get_global_collection():
    return get_or_create_collection(global_collection_name())


def delete_project_collection(project_id: str) -> None:
    delete_collection(project_collection_name(project_id))


def _delete_from_collection(col, doc_id: str, doc_name: str) -> int:
    """
    Delete all chunks matching either doc_id or source_doc from one collection.
    Tries doc_id first (new ingestion), then source_doc (legacy ingestion).
    Returns total chunks deleted.
    """
    if col.count() == 0:
        return 0

    deleted = 0

    # Strategy 1: match on doc_id metadata (new documents)
    try:
        results = col.get(where={"doc_id": {"$eq": doc_id}})
        ids = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
            deleted += len(ids)
            logger.info(f"[CollectionManager] Deleted {len(ids)} chunks by doc_id={doc_id}")
    except Exception as e:
        logger.debug(f"[CollectionManager] doc_id filter failed: {e}")

    # Strategy 2: match on source_doc filename (legacy/old documents)
    if doc_name:
        try:
            results = col.get(where={"source_doc": {"$eq": doc_name}})
            ids = [i for i in results.get("ids", []) if i]
            if ids:
                col.delete(ids=ids)
                deleted += len(ids)
                logger.info(f"[CollectionManager] Deleted {len(ids)} chunks by source_doc={doc_name}")
        except Exception as e:
            logger.debug(f"[CollectionManager] source_doc filter failed: {e}")

    return deleted


def delete_document_chunks(doc_id: str, project_id: str, doc_name: str = "") -> int:
    """
    Delete all ChromaDB vectors for a document from BOTH project and global collections.

    Args:
        doc_id:     DB document UUID
        project_id: Project UUID
        doc_name:   Original filename (fallback for legacy chunks without doc_id metadata)

    Returns:
        Total number of chunks deleted across all collections.
    """
    total_deleted = 0

    # Delete from project collection
    try:
        proj_col = get_project_collection(project_id)
        total_deleted += _delete_from_collection(proj_col, doc_id, doc_name)
    except Exception as e:
        logger.warning(f"[CollectionManager] Project collection deletion failed: {e}")

    # Delete from global collection
    try:
        glob_col = get_global_collection()
        total_deleted += _delete_from_collection(glob_col, doc_id, doc_name)
    except Exception as e:
        logger.warning(f"[CollectionManager] Global collection deletion failed: {e}")

    logger.info(f"[CollectionManager] Total deleted: {total_deleted} chunks for doc_id={doc_id}")
    return total_deleted


def get_collection_stats() -> List[dict]:
    names = list_collections()
    return [{"name": n, "count": collection_count(n)} for n in names]
