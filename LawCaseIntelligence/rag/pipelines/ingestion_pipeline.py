"""
rag/pipelines/ingestion_pipeline.py
End-to-end ingestion: PDF → chunks → embeddings → ChromaDB.

IMPORTANT: doc_id is now stored in chunk metadata so vectors
can be reliably deleted when a document is removed.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from backend.config.settings import get_settings
from rag.ingestion.pdf_loader import (
    extract_text_from_pdf,
    get_pdf_page_count,
    stream_chunks,
)
from rag.ingestion.metadata_extractor import extract_court_metadata
from rag.chunking.semantic_chunker import chunk_text
from rag.embeddings.embedding_factory import get_embeddings
from rag.vectordb.collection_manager import (
    project_collection_name,
    global_collection_name,
    get_or_create_collection,
)

logger = logging.getLogger(__name__)


def ingest_document(
    pdf_path: str,
    project_id: str,
    doc_id: Optional[str] = None,          # ← NEW: pass the DB doc_id for deletion
    metadata: Optional[Dict] = None,
    batch_size: int = 32,
) -> int:
    """
    Ingest a PDF into ChromaDB for RAG retrieval.

    Args:
        pdf_path:   Absolute path to the PDF
        project_id: Project UUID
        doc_id:     DB document UUID — stored in metadata for reliable deletion
        metadata:   Pre-extracted metadata dict (court, case_number, etc.)
        batch_size: Embedding batch size

    Returns:
        Number of chunks indexed
    """
    settings   = get_settings()
    emb        = get_embeddings()
    doc_name   = Path(pdf_path).name
    page_count = get_pdf_page_count(pdf_path)
    # Use provided doc_id or derive from filename (fallback for old callers)
    effective_doc_id = doc_id or doc_name

    logger.info(f"Ingesting: {doc_name} ({page_count} pages)")

    # ── Extract text ─────────────────────────────────────────────
    if page_count >= settings.large_page_threshold:
        chunk_tuples = list(stream_chunks(pdf_path, settings.chunk_size, settings.chunk_overlap))
        chunks   = [c for c, _ in chunk_tuples]
        pg_lists = [p for _, p in chunk_tuples]
    else:
        full_text, _ = extract_text_from_pdf(pdf_path)
        chunks   = chunk_text(full_text, settings.chunk_size, settings.chunk_overlap)
        pg_lists = [[] for _ in chunks]

    if not chunks:
        logger.warning(f"No chunks extracted from {doc_name}")
        return 0

    # ── Extract metadata ─────────────────────────────────────────
    if not metadata:
        head_text, _ = extract_text_from_pdf(pdf_path) if page_count < 10 else ("", [])
        metadata = extract_court_metadata(head_text[:3000]) if head_text else {}

    # ── Build per-chunk metadata — include doc_id for deletion ───
    chunk_metas: List[Dict] = []
    chunk_ids:   List[str]  = []
    for i, (chunk, pages) in enumerate(zip(chunks, pg_lists)):
        chunk_metas.append({
            "source_doc":   doc_name,
            "doc_id":       effective_doc_id,   # ← stored for deletion queries
            "project_id":   project_id,
            "court":        metadata.get("court", ""),
            "case_number":  metadata.get("case_number", ""),
            "date":         metadata.get("date_of_judgment", ""),
            "chunk_index":  i,
            "page_nums":    str(pages),
        })
        # Stable ID: project + doc_id + index (enables prefix-based deletion)
        chunk_ids.append(f"{project_id}_{effective_doc_id}_{i}")

    # ── Embed in batches ─────────────────────────────────────────
    all_embeddings: List[List[float]] = []
    for b_start in range(0, len(chunks), batch_size):
        batch = chunks[b_start:b_start + batch_size]
        vecs  = emb.embed_documents(batch)
        all_embeddings.extend(vecs)
        logger.info(f"Embedded batch {b_start//batch_size + 1} / {-(-len(chunks)//batch_size)}")

    # ── Store in project and global collections ──────────────────
    for col_name in [project_collection_name(project_id), global_collection_name()]:
        col = get_or_create_collection(col_name)
        for b_start in range(0, len(chunks), 100):
            col.add(
                documents=chunks[b_start:b_start+100],
                embeddings=all_embeddings[b_start:b_start+100],
                metadatas=chunk_metas[b_start:b_start+100],
                ids=chunk_ids[b_start:b_start+100],
            )

    logger.info(f"✅ Ingested {len(chunks)} chunks from {doc_name} (doc_id={effective_doc_id})")
    return len(chunks)
