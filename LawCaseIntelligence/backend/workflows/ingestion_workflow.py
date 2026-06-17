"""
backend/workflows/ingestion_workflow.py
Document ingestion workflow: save PDF → register in DB → ingest into RAG.
Separate from the extraction pipeline — handles bulk RAG ingestion.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def ingest_pdf_to_rag(
    pdf_path: str,
    project_id: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Ingest a PDF into ChromaDB for RAG search without running agent extraction.
    Useful for bulk-indexing documents before extraction.

    Returns: {chunks_indexed, error}
    """
    try:
        from rag.pipelines.ingestion_pipeline import ingest_document
        count = ingest_document(
            pdf_path=pdf_path,
            project_id=project_id,
            metadata=metadata or {},
        )
        logger.info(f"RAG ingestion complete: {pdf_path} → {count} chunks")
        return {"chunks_indexed": count, "error": None}
    except Exception as e:
        logger.error(f"RAG ingestion failed for {pdf_path}: {e}")
        return {"chunks_indexed": 0, "error": str(e)}


def bulk_ingest_directory(
    directory: str,
    project_id: str,
) -> List[Dict]:
    """
    Ingest all PDFs in a directory into the RAG system.
    Returns list of per-file results.
    """
    from backend.utils.file_utils import list_pdfs
    paths   = list_pdfs(directory)
    results = []
    logger.info(f"Bulk ingesting {len(paths)} PDFs from {directory}")
    for path in paths:
        result = ingest_pdf_to_rag(path, project_id)
        result["pdf_path"] = path
        results.append(result)
    return results


def register_document(
    project_id: str,
    pdf_path: str,
    file_size: int = 0,
) -> str:
    """
    Register a PDF in the documents table. Returns document ID.
    """
    from database.repositories.case_repository import DocumentRepository
    dr  = DocumentRepository()
    doc = dr.create(
        project_id=project_id,
        document_name=Path(pdf_path).name,
        location=pdf_path,
        file_size=file_size,
    )
    return doc.id
