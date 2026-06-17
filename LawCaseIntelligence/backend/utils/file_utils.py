"""
backend/utils/file_utils.py
File handling utilities for PDF uploads and exports.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist. Return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size_mb(path: str) -> float:
    """Return file size in MB."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def get_file_hash(path: str) -> str:
    """Return SHA-256 hash of file contents for deduplication."""
    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as e:
        logger.warning(f"Could not hash file {path}: {e}")
        return ""


def safe_filename(name: str) -> str:
    """Sanitize a filename — strip special chars, limit length."""
    import re
    name = re.sub(r"[^\w\s\-.]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:200]


def list_pdfs(directory: str) -> List[str]:
    """Return sorted list of PDF paths in a directory."""
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*.pdf"))


def delete_file(path: str) -> bool:
    """Delete a file. Returns True on success."""
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.warning(f"Could not delete {path}: {e}")
        return False


def copy_file(src: str, dst: str) -> str:
    """Copy file from src to dst. Return dst path."""
    ensure_dir(str(Path(dst).parent))
    shutil.copy2(src, dst)
    return dst


def write_text_file(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write string content to a text file."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """Read and return text file contents."""
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except OSError as e:
        logger.warning(f"Could not read {path}: {e}")
        return ""
