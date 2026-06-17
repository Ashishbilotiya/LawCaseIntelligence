"""
backend/utils/text_utils.py
Text processing utilities for legal document handling.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List


def clean_text(text: str) -> str:
    """
    Clean raw PDF-extracted text:
    - Normalize unicode
    - Remove null bytes and control chars
    - Collapse excessive whitespace
    - Fix common OCR artefacts
    """
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove null bytes and non-printable control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Remove page-break markers
    text = re.sub(r"\f", "\n", text)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but preserve leading indent on lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def truncate_text(text: str, max_chars: int = 3000) -> str:
    """Truncate text to max_chars, breaking at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


def extract_section_numbers(text: str) -> List[str]:
    """Extract all Section/Article references from text."""
    patterns = [
        r"[Ss]ection\s+\d+[A-Za-z]?(?:\(\d+\))?(?:\([a-z]\))?",
        r"[Aa]rticle\s+\d+[A-Za-z]?",
        r"[Ss]ub-[Ss]ection\s+\(\d+\)",
        r"[Cc]lause\s+\([a-z\d]+\)",
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text))
    return list(dict.fromkeys(found))


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple heuristics."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r"\b\w+\b", text))


def normalize_party_name(name: str) -> str:
    """Normalize party name — title case, strip extra whitespace."""
    return re.sub(r"\s+", " ", name.strip()).title()


def extract_years(text: str) -> List[str]:
    """Extract 4-digit years (1900–2030) from text."""
    return re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", text)
