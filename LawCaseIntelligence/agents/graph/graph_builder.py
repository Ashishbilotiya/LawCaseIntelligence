"""
agents/graph/graph_builder.py
LangGraph pipeline — optimized with UCE + parallel supervisor agents + chunk ranking.

New flow (~3-8 min vs 15-25 min):
  PDF Extractor
      ↓
  ChunkRanker (17 → ~12 chunks)
      ↓
  UCE Node (~12 LLM calls, one per chunk)
      ↓
  Parallel Supervisor Agents (4 concurrent × 1 call = 4 LLM calls)
      ↓
  Reasoning Agent (1 LLM call)
      ↓
  Master Agent (1 LLM call)
      ↓
  DB Write + RAG Ingestion

Total: ~18-22 LLM calls vs 85+ previously.
"""
from __future__ import annotations

import concurrent.futures
import gc
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from agents.state.legal_state import LegalState
from agents.nodes.issue_agent import issue_agent_node
from agents.nodes.argument_agent import argument_agent_node
from agents.nodes.law_statute_agent import law_statute_agent_node
from agents.nodes.precedent_agent import precedent_agent_node
from agents.nodes.reasoning_verdict_agent import reasoning_verdict_agent_node
from agents.nodes.master_supervisor_agent import (
    master_supervisor_node, validation_router, increment_retry_node,
)
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()
TOTAL_STEPS = _settings.pipeline_total_steps


def _emit(room: Optional[str], event: str, data: dict) -> None:
    if not room:
        return
    try:
        from frontend.flask_app import socketio
        socketio.emit(event, data, room=room)
    except Exception as e:
        logger.debug(f"Socket emit failed: {e}")


def _progress(room, step, label, status, info=""):
    _emit(room, "agent_progress", {
        "step": step, "total": TOTAL_STEPS,
        "agent": label, "status": status, "info": info,
    })


# ── Node 1: PDF Extractor ─────────────────────────────────────────

def pdf_extractor_node(state: LegalState) -> Dict[str, Any]:
    from rag.ingestion.pdf_loader import extract_text_from_pdf, get_pdf_page_count, stream_chunks
    from rag.ingestion.metadata_extractor import extract_court_metadata
    from rag.chunking.semantic_chunker import chunk_text

    _progress(state.get("socket_room"), 1, "PDF Extractor", "running")
    settings = get_settings()
    pdf_path = state["pdf_path"]
    logger.info(f"▶ PDF Extractor: {pdf_path}")

    try:
        page_count = get_pdf_page_count(pdf_path)
        is_large   = page_count >= settings.large_page_threshold

        if is_large:
            chunk_texts = [c for c, _ in stream_chunks(pdf_path, settings.chunk_size)]
            full_text   = ""
            meta        = {}
            gc.collect()
        else:
            full_text, _ = extract_text_from_pdf(pdf_path)
            chunk_texts  = chunk_text(full_text, settings.chunk_size, settings.chunk_overlap)
            meta         = extract_court_metadata(full_text[:3000])

        logger.info(f"✅ PDF extracted: {page_count} pages → {len(chunk_texts)} chunks")
        _progress(state.get("socket_room"), 1, "PDF Extractor", "done",
                  f"{page_count} pages, {len(chunk_texts)} chunks")

        return {
            "full_text":    full_text,
            "chunks":       chunk_texts,
            "trimmed_text": (full_text or "")[:settings.chunk_size],
            "page_count":   page_count,
            "metadata":     meta,
            "uce_results":  [],
            "retry_count":  0,
            "validation_errors": [],
            "processing_log": [f"PDF: {page_count} pages, {len(chunk_texts)} chunks"],
            "rag_ingested": False,
            "error":        None,
        }
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        _progress(state.get("socket_room"), 1, "PDF Extractor", "error", str(e))
        return {
            "full_text": "", "chunks": [], "trimmed_text": "",
            "page_count": 0, "metadata": {}, "uce_results": [],
            "retry_count": 0, "validation_errors": [],
            "processing_log": [f"PDF error: {e}"],
            "rag_ingested": False, "error": str(e),
        }


# ── Node 2: UCE (with ChunkRanker) ───────────────────────────────

def uce_node(state: LegalState) -> Dict[str, Any]:
    """
    UCE extracts all fields from each chunk in ONE LLM call per chunk.
    Now processes all chunks to ensure absolute legal coverage.
    """
    from agents.universal_chunk_extractor import extract_from_chunks

    chunks = state.get("chunks", [])
    room   = state.get("socket_room")

    _progress(room, 2, "Universal Extraction", "running", f"processing {len(chunks)} chunks")

    if not chunks:
        logger.warning("[UCE] No chunks")
        _progress(room, 2, "Universal Extraction", "done", "no chunks")
        return {"uce_results": []}

    # Process all chunks without skipping or ranking to ensure no critical evidence is missed
    uce_results = extract_from_chunks(chunks, agent_name="UCE")
    succeeded   = sum(1 for r in uce_results if r)

    logger.info(f"[UCE] ✅ {succeeded}/{len(chunks)} chunks extracted")
    _progress(room, 2, "Universal Extraction", "done", f"{succeeded}/{len(chunks)} chunks")

    return {
        "uce_results":    uce_results,
        "processing_log": state.get("processing_log", []) + [
            f"UCE: {succeeded}/{len(chunks)} chunks processed"
        ],
    }


# ── Node 3: Parallel Supervisors ──────────────────────────────────

def parallel_supervisors_node(state: LegalState) -> Dict[str, Any]:
    """
    Run Issue, Argument, Law, Precedent agents concurrently.
    Each makes exactly ONE LLM call on aggregated UCE data.
    4 calls in parallel ≈ time of 1 sequential call.
    """
    room = state.get("socket_room")
    _progress(room, 3, "Supervisor Agents", "running", "4 agents running in parallel")

    results: Dict[str, Any] = {}

    def run(name, fn, s):
        try:
            return name, fn(s)
        except Exception as e:
            logger.error(f"[Parallel] {name} error: {e}")
            return name, {}

    agents = [
        ("issue",     issue_agent_node),
        ("argument",  argument_agent_node),
        ("statute",   law_statute_agent_node),
        ("precedent", precedent_agent_node),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run, name, fn, state): name for name, fn in agents}
        for f in concurrent.futures.as_completed(futures):
            _, result = f.result()
            results.update(result)

    _progress(room, 3, "Supervisor Agents", "done", "4 agents complete")
    log = state.get("processing_log", []) + ["ParallelSupervisors: 4 agents complete"]
    return {**results, "processing_log": log}


# ── Node 4: Reasoning ─────────────────────────────────────────────

def reasoning_node(state: LegalState) -> Dict[str, Any]:
    room = state.get("socket_room")
    _progress(room, 4, "Reasoning & Verdict", "running")
    result = reasoning_verdict_agent_node(state)
    outcome = (result.get("reasoning_output") or {}).get("outcome", "")
    _progress(room, 4, "Reasoning & Verdict", "done", outcome)
    return result


# ── Node 5: Master ────────────────────────────────────────────────

def master_node(state: LegalState) -> Dict[str, Any]:
    room = state.get("socket_room")
    _progress(room, 5, "Master Supervisor", "running")
    result = master_supervisor_node(state)
    _progress(room, 5, "Master Supervisor", "done",
              f"category={result.get('case_category','')} win={result.get('win_indicator','')}")
    return result


# ── Node 6: DB Write ──────────────────────────────────────────────

def db_write_node(state: LegalState) -> Dict[str, Any]:
    from database.database import get_session
    from database.models import ProcessedJudgment, DocumentInProject

    logger.info("▶ DB Write Node")
    final  = state.get("final_json") or {}
    trends = final.get("trends")     or {}
    meta   = state.get("metadata")   or {}

    try:
        s   = get_session()
        doc = s.query(DocumentInProject).filter_by(
            project_id=state["project_id"], document_name=state["document_name"]
        ).first()
        if doc:
            doc.status = "done"

        record = ProcessedJudgment(
            id=final.get("document_id", str(uuid.uuid4())),
            project_id=state["project_id"],
            document_name=state["document_name"],
            court=meta.get("court", ""),
            case_number=meta.get("case_number", ""),
            date_of_judgment=meta.get("date_of_judgment", ""),
            case_category=final.get("case_category", "General"),
            win_indicator=final.get("win_indicator", "Neutral"),
            outcome=final.get("outcome", ""),
            issue_json=final.get("issue"),
            petitioner_json=final.get("petitioner_arguments"),
            respondent_json=final.get("respondent_arguments"),
            statutes_json=final.get("statutes"),
            precedents_json=final.get("precedents"),
            reasoning_json=final.get("reasoning"),
            trends_json=final.get("trends"),
            case_summary=final.get("case_summary", ""),
            frequently_cited_sections=trends.get("frequently_cited_sections", []),
            processing_log=final.get("processing_log", []),
        )
        s.add(record); s.commit()
        db_id = record.id; s.close()
        logger.info(f"✅ DB write: {db_id}")
        _progress(state.get("socket_room"), 6, "DB Write", "done")
        return {"db_record_id": db_id}
    except Exception as e:
        logger.error(f"DB write failed: {e}")
        return {"db_record_id": None, "error": str(e)}


# ── Node 7: RAG Ingestion ─────────────────────────────────────────

def rag_ingestion_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ RAG Ingestion Node")
    try:
        from rag.pipelines.ingestion_pipeline import ingest_document
        from database.database import get_session
        from database.models import DocumentInProject

        doc_id = None
        try:
            s   = get_session()
            rec = s.query(DocumentInProject).filter_by(
                project_id=state["project_id"], document_name=state["document_name"]
            ).first()
            if rec:
                doc_id = rec.id
            s.close()
        except Exception as e:
            logger.warning(f"Could not resolve doc_id: {e}")

        ingest_document(
            pdf_path=state["pdf_path"],
            project_id=state["project_id"],
            doc_id=doc_id,
            metadata=state.get("metadata") or {},
        )
        logger.info("✅ RAG ingestion complete")
        _progress(state.get("socket_room"), 7, "RAG Ingestion", "done")
        return {"rag_ingested": True}
    except Exception as e:
        logger.warning(f"RAG ingestion failed: {e}")
        return {"rag_ingested": False}


# ── Node 8: Doc Builder ───────────────────────────────────────────

def doc_builder_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ Document Builder Node")
    final  = state.get("final_json") or {}
    issue  = final.get("issue")                or {}
    pet    = final.get("petitioner_arguments") or {}
    res    = final.get("respondent_arguments") or {}
    stat   = final.get("statutes")             or {}
    prec   = final.get("precedents")           or {}
    reason = final.get("reasoning")            or {}
    trends = final.get("trends")               or {}

    def bullet(items):
        return "\n".join(f"  - {i}" for i in (items or [])) or "  - N/A"

    statutes_text = "\n".join(
        f"  - {s.get('act','')} § {s.get('section','')} — {s.get('description','')}"
        for s in stat.get("statutes", [])
    ) or "  - N/A"

    prec_text = "\n".join(
        f"  - {p.get('case_name','')} ({p.get('year','')}) — {p.get('relevance','')}"
        for p in prec.get("precedents", [])
    ) or "  - N/A"

    doc = f"""# ⚖️ Case Brief — {state.get('document_name', 'Unknown')}

**Court:** {final.get('court','N/A')} | **Case:** {final.get('case_number','N/A')} | **Date:** {final.get('date_of_judgment','N/A')}
**Category:** {trends.get('case_category','N/A')} | **Win Indicator:** {trends.get('win_indicator','N/A')}
**Outcome:** {final.get('outcome','N/A')}

---

## 1. Legal Issue
**Primary:** {issue.get('primary_issue','N/A')}

## 2. Petitioner Arguments
{bullet(pet.get('key_arguments',[]))}

## 3. Respondent Arguments
{bullet(res.get('key_arguments',[]))}

## 4. Laws & Statutes Applied
{statutes_text}

## 5. Precedents Cited
{prec_text}

## 6. Judicial Reasoning
{reason.get('judicial_reasoning','N/A')}

## 7. Final Decision
**Outcome:** {reason.get('outcome','N/A')}
**Relief Granted:** {reason.get('relief_granted','N/A')}

## 8. Plain-Language Summary
{final.get('case_summary','N/A')}

---
*Generated by LawCaseIntelligence — UCE + Parallel Supervisor Pipeline*
"""
    return {"output_document": doc}


# ── Build Graph ───────────────────────────────────────────────────

def build_legal_graph():
    from database.database import init_db
    init_db()

    g = StateGraph(LegalState)
    g.add_node("pdf_extractor",   pdf_extractor_node)
    g.add_node("uce_node",        uce_node)
    g.add_node("parallel_sup",    parallel_supervisors_node)
    g.add_node("reasoning",       reasoning_node)
    g.add_node("master",          master_node)
    g.add_node("increment_retry", increment_retry_node)
    g.add_node("db_write",        db_write_node)
    g.add_node("rag_ingestion",   rag_ingestion_node)
    g.add_node("doc_builder",     doc_builder_node)

    g.set_entry_point("pdf_extractor")
    g.add_edge("pdf_extractor", "uce_node")
    g.add_edge("uce_node",      "parallel_sup")
    g.add_edge("parallel_sup",  "reasoning")
    g.add_edge("reasoning",     "master")
    g.add_conditional_edges(
        "master", validation_router,
        {"db_write": "db_write", "retry": "increment_retry"},
    )
    g.add_edge("increment_retry", "master")
    g.add_edge("db_write",        "rag_ingestion")
    g.add_edge("rag_ingestion",   "doc_builder")
    g.add_edge("doc_builder",     END)

    return g.compile()


# ── Public entry point ────────────────────────────────────────────

def process_judgment(
    pdf_path: str,
    project_id: str,
    document_name: str | None = None,
    socket_room:   str | None = None,
) -> Dict[str, Any]:
    from database.database import init_db
    init_db()

    name  = document_name or Path(pdf_path).name
    graph = build_legal_graph()

    initial: LegalState = {
        "pdf_path":         pdf_path,
        "project_id":       project_id,
        "document_name":    name,
        "socket_room":      socket_room,
        "full_text":        "",
        "chunks":           [],
        "trimmed_text":     "",
        "page_count":       0,
        "metadata":         {},
        "uce_results":      [],
        "issue_output":     None,
        "argument_output":  None,
        "statute_output":   None,
        "precedent_output": None,
        "reasoning_output": None,
        "final_json":       None,
        "case_summary":     "",
        "headnotes":        [],
        "case_category":    "",
        "win_indicator":    "",
        "validation_errors": [],
        "retry_count":      0,
        "db_record_id":     None,
        "rag_ingested":     False,
        "output_document":  None,
        "error":            None,
        "processing_log":   [],
    }

    logger.info(f"🚀 Pipeline: {name} (project={project_id}, room={socket_room})")
    result = graph.invoke(initial)
    logger.info(f"✅ Complete. DB ID: {result.get('db_record_id')}")

    return {
        "final_json":      result.get("final_json"),
        "output_document": result.get("output_document"),
        "db_record_id":    result.get("db_record_id"),
        "rag_ingested":    result.get("rag_ingested"),
        "error":           result.get("error"),
        "processing_log":  result.get("processing_log", []),
    }
