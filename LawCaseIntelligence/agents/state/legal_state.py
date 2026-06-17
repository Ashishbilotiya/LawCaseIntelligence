"""
agents/state/legal_state.py
LangGraph TypedDict state shared across all agents in the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class LegalState(TypedDict):

    # ── Input ─────────────────────────────────────────────────────
    pdf_path:      str
    project_id:    str
    document_name: str
    socket_room:   Optional[str]

    # ── Extraction ────────────────────────────────────────────────
    full_text:    str
    chunks:       List[str]
    trimmed_text: str
    page_count:   int
    metadata:     Dict[str, str]

    # ── UCE Results (one dict per chunk, from Universal Chunk Extractor)
    uce_results: List[Dict]

    # ── Agent Outputs (populated by supervisor agents) ────────────
    issue_output:     Optional[Dict]
    argument_output:  Optional[Dict]
    statute_output:   Optional[Dict]
    precedent_output: Optional[Dict]
    reasoning_output: Optional[Dict]

    # ── Master Supervisor Output ───────────────────────────────────
    final_json:    Optional[Dict]
    case_summary:  str
    headnotes:     List[str]
    case_category: str
    win_indicator: str

    # ── Validation & Retry ────────────────────────────────────────
    validation_errors: List[str]
    retry_count:       int

    # ── Database & RAG ────────────────────────────────────────────
    db_record_id: Optional[str]
    rag_ingested: bool

    # ── Output ────────────────────────────────────────────────────
    output_document: Optional[str]

    # ── Error Tracking ────────────────────────────────────────────
    error:          Optional[str]
    processing_log: List[str]
