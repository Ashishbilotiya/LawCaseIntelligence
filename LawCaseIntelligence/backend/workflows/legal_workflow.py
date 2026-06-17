"""
backend/workflows/legal_workflow.py
High-level workflow coordinator that wraps the LangGraph pipeline.
Provides sync and async entry points, progress callbacks, and batch processing.
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_single(
    pdf_path: str,
    project_id: str,
    document_name: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Run the full 6-agent pipeline on a single PDF.

    Args:
        pdf_path:     Absolute path to PDF
        project_id:   Project UUID
        document_name: Override display name
        on_progress:  Optional callback(message) for progress updates

    Returns:
        Pipeline result dict with final_json, db_record_id, error
    """
    from agents.graph.graph_builder import process_judgment

    name = document_name or Path(pdf_path).name
    if on_progress:
        on_progress(f"Starting pipeline for: {name}")

    try:
        result = process_judgment(
            pdf_path=pdf_path,
            project_id=project_id,
            document_name=name,
        )
        if on_progress:
            on_progress(f"✅ Complete: {name}")
        return result
    except Exception as e:
        logger.error(f"Pipeline failed for {name}: {e}")
        if on_progress:
            on_progress(f"❌ Failed: {name} — {e}")
        return {"error": str(e), "document_name": name}


def run_batch(
    pdf_paths: List[str],
    project_id: str,
    max_workers: int = 2,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    Run the pipeline on multiple PDFs using a thread pool.
    max_workers=2 to respect LLM rate limits.

    Returns list of result dicts in completion order.
    """
    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single, path, project_id, None, on_progress): path
            for path in pdf_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item failed {path}: {e}")
                results.append({"error": str(e), "pdf_path": path})

    logger.info(f"Batch complete: {len(results)}/{len(pdf_paths)} processed")
    return results


def run_async_background(
    pdf_path: str,
    project_id: str,
    document_name: Optional[str] = None,
) -> threading.Thread:
    """
    Launch pipeline in a background daemon thread.
    Returns the thread (already started).
    """
    t = threading.Thread(
        target=run_single,
        args=(pdf_path, project_id, document_name),
        daemon=True,
    )
    t.start()
    logger.info(f"Background pipeline started for: {document_name or pdf_path}")
    return t
