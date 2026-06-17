"""
rag/ingestion/pdf_loader.py
PDF loading utilities using PyMuPDF (fitz).
Supports streaming for large documents (100+ pages).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Tuple

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")


def get_pdf_page_count(pdf_path: str) -> int:
    """Return number of pages in a PDF without loading all content."""
    if not PYMUPDF_AVAILABLE:
        return 0
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[dict]]:
    """
    Extract full text and per-page metadata from a PDF.
    Returns (full_text, page_metadata_list).
    Suitable for PDFs up to ~100 pages.
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    full_text_parts: list = []
    page_metadata: list = []

    # FIX: capture page_count BEFORE closing the document
    total_pages = doc.page_count

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            full_text_parts.append(text)
            page_metadata.append({"page_num": page_num + 1, "char_count": len(text)})

    doc.close()

    full_text = "\n\n".join(full_text_parts)
    # FIX: use the pre-captured total_pages, not doc.page_count (doc is now closed)
    logger.info(f"Extracted {len(full_text)} chars from {total_pages} pages: {pdf_path}")
    return full_text, page_metadata


def stream_page_text(pdf_path: str) -> Generator[Tuple[int, str], None, None]:
    """
    Generator that yields (page_number, page_text) one page at a time.
    Memory-efficient for very large PDFs.
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF not installed.")

    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            yield page_num + 1, text
        page = None  # free page memory
    doc.close()


def stream_chunks(
    pdf_path: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> Generator[Tuple[str, List[int]], None, None]:
    """
    Stream text chunks from a PDF without loading the full document.
    Yields (chunk_text, [page_numbers_in_chunk]).
    """
    buffer = ""
    buffer_pages: list = []

    for page_num, page_text in stream_page_text(pdf_path):
        buffer += page_text + "\n"
        buffer_pages.append(page_num)

        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            yield chunk, list(buffer_pages)
            # Slide window with overlap
            buffer = buffer[chunk_size - chunk_overlap:]
            # Keep only pages that could still be in buffer
            buffer_pages = [buffer_pages[-1]] if buffer_pages else []

    # Yield remaining buffer
    if buffer.strip():
        yield buffer.strip(), buffer_pages
