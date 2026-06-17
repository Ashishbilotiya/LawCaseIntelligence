"""
rag/chunking/semantic_chunker.py
Text chunking utilities with overlap support.
"""
from __future__ import annotations

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> List[str]:
    """
    Split text into overlapping chunks, preferring sentence boundaries.
    """
    if not text or not text.strip():
        return []

    # Prefer splitting at paragraph/sentence breaks
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            if current and chunk_overlap > 0:
                overlap_start = max(0, len(current) - chunk_overlap)
                current = current[overlap_start:] + " " + sentence
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    # Filter tiny chunks
    chunks = [c for c in chunks if len(c.strip()) > 100]
    return chunks


def split_by_size(text: str, size: int = 3000) -> List[str]:
    """Simple fixed-size character splitter — no overlap."""
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]
