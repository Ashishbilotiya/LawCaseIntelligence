"""
rag/ingestion/document_processor.py
High-level document processor — orchestrates load → clean → chunk → embed → store.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def process_document(
    pdf_path: str,
    project_id: str,
    metadata: Optional[Dict] = None,
) -> Tuple[List[str], Dict]:
    """
    Process a PDF end-to-end:
    1. Extract text (streaming if large)
    2. Clean text
    3. Extract metadata
    4. Chunk text

    Returns (chunks, metadata_dict) — does NOT embed or store.
    Use ingestion_pipeline.ingest_document for full storage.
    """
    from backend.config.settings import get_settings
    from rag.ingestion.pdf_loader import (
        extract_text_from_pdf,
        get_pdf_page_count,
        stream_chunks,
    )
    from rag.ingestion.metadata_extractor import extract_court_metadata
    from rag.chunking.semantic_chunker import chunk_text
    from backend.utils.text_utils import clean_text

    settings   = get_settings()
    page_count = get_pdf_page_count(pdf_path)

    if page_count >= settings.large_page_threshold:
        logger.info(f"Large PDF ({page_count}p) — streaming chunks")
        chunks = [c for c, _ in stream_chunks(pdf_path, settings.chunk_size, settings.chunk_overlap)]
        header = chunks[0] if chunks else ""
    else:
        full_text, _ = extract_text_from_pdf(pdf_path)
        full_text    = clean_text(full_text)
        chunks       = chunk_text(full_text, settings.chunk_size, settings.chunk_overlap)
        header       = full_text[:3000]

    if not metadata:
        metadata = extract_court_metadata(header)

    metadata["source_doc"]  = Path(pdf_path).name
    metadata["project_id"]  = project_id
    metadata["page_count"]  = page_count

    logger.info(f"Processed {Path(pdf_path).name}: {len(chunks)} chunks")
    return chunks, metadata
