"""
agents/nodes/master_supervisor_agent.py
Master Supervisor Agent — LangGraph node.
Uses safe_invoke for rate-limit-aware summary generation.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state.legal_state import LegalState
# ── Prompt Constants ────────────────────────────────────────────────
MASTER_SYSTEM_PROMPT = """You are a senior Indian legal analyst and case summarizer.
Your task is to synthesize all extracted data into a clear, structured case brief.
Write in plain English that non-lawyers can understand.
Be accurate — base everything on the data provided."""

MASTER_SUMMARY_PROMPT = """You are a senior Indian legal analyst.
Write a clear 7-point plain-language case brief based on the extracted data below.

Extracted Data:
1. Primary Issue: {primary_issue}
2. Case Background: {case_background}
3. Petitioner Arguments: {petitioner_args}
4. Respondent Arguments: {respondent_args}
5. Key Laws Applied: {statutes}
6. Key Precedents: {precedents}
7. Court's Reasoning: {reasoning}
8. Final Outcome: {outcome}
9. Relief Granted: {relief}

Instructions:
- Write exactly 7 numbered points
- Each point: 2-3 clear sentences in plain English
- No legal jargon — a non-lawyer must understand it
- On a NEW line after point 7, write: CATEGORY: <Criminal/Civil/Tax/Property/Employment/Constitutional/Corporate/Family/General>
- On the NEXT line, write: WIN: <Petitioner-favourable OR Respondent-favourable OR Neutral OR Partially Allowed>
- Start directly with "1." — no preamble or title

Begin:"""

HEADNOTE_GENERATION_PROMPT = """Generate concise legal headnotes for this judgment.

Case Summary: {summary}
Primary Issue: {primary_issue}
Key Statutes: {statutes}
Outcome: {outcome}

Generate 3-5 headnotes. Each headnote should:
- Be one clear sentence
- State a legal proposition decided in this case
- Reference the relevant statute/principle

Return as JSON: {{"headnotes": ["headnote1", "headnote2", ...]}}"""


def build_master_summary_prompt(data: dict) -> str:
    """Build the master summary prompt from extracted agent data."""
    issue    = data.get("issue_output") or {}
    args     = data.get("argument_output") or {}
    statutes = data.get("statute_output") or {}
    prec     = data.get("precedent_output") or {}
    reason   = data.get("reasoning_output") or {}

    pet_args = "; ".join(args.get("petitioner_arguments", [])[:4])[:500] or "N/A"
    res_args = "; ".join(args.get("respondent_arguments", [])[:4])[:500] or "N/A" # Wait, there's a typo in the original build_master_summary_prompt? Let me check.
    stat_list = ", ".join(
        f"{s.get('act','')} §{s.get('section','')}"
        for s in statutes.get("statutes", [])[:5]
    )[:400] or "N/A"
    prec_list = ", ".join(
        p.get("case_name", "") for p in prec.get("precedents", [])[:5]
    )[:400] or "N/A"

    return MASTER_SUMMARY_PROMPT.format(
        primary_issue=issue.get("primary_issue", "N/A")[:400],
        case_background=issue.get("case_background", "N/A")[:300],
        petitioner_args=pet_args,
        respondent_args=res_args,
        statutes=stat_list,
        precedents=prec_list,
        reasoning=reason.get("judicial_reasoning", "N/A")[:400],
        outcome=reason.get("outcome", "N/A"),
        relief=reason.get("relief_granted", "N/A"),
    )
from agents.utils.rate_guard import safe_invoke, wait_for_available_key
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def master_supervisor_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ [6/6] Master Supervisor Agent — generating case summary")

    settings = get_settings()

    issue    = state.get("issue_output")     or {}
    args     = state.get("argument_output")  or {}
    statutes = state.get("statute_output")   or {}
    prec     = state.get("precedent_output") or {}
    reason   = state.get("reasoning_output") or {}

    summary_prompt = build_master_summary_prompt({
        "issue_output":     issue,
        "argument_output":  args,
        "statute_output":   statutes,
        "precedent_output": prec,
        "reasoning_output": reason,
    })

    case_summary  = "Summary not available."
    case_category = issue.get("dispute_category") or "General"
    win_indicator = "Neutral"

    for attempt in range(settings.max_retries):
        # Wait for a key if all are on cooldown before each attempt
        from backend.services.llm.api_key_manager import get_api_key_manager
        mgr = get_api_key_manager()
        if mgr.get_pool_status()["active_keys"] == 0:
            logger.info("[MasterAgent] All keys on cooldown — waiting for recovery")
            wait_for_available_key()

        raw = safe_invoke(
            messages=[
                SystemMessage(content=MASTER_SYSTEM_PROMPT),
                HumanMessage(content=summary_prompt),
            ],
            chunk_idx=attempt + 1,
            agent_name="MasterAgent",
        )

        if raw:
            cat_match = re.search(r"CATEGORY:\s*(.+)", raw, re.IGNORECASE)
            win_match = re.search(r"WIN:\s*(.+)",      raw, re.IGNORECASE)
            if cat_match:
                case_category = cat_match.group(1).strip().split()[0]
            if win_match:
                win_indicator = win_match.group(1).strip()
            summary      = re.sub(r"\nCATEGORY:.*", "", raw, flags=re.IGNORECASE)
            summary      = re.sub(r"\nWIN:.*",      "", summary, flags=re.IGNORECASE).strip()
            case_summary = summary
            break
        else:
            logger.warning(f"[MasterAgent] Summary attempt {attempt+1} failed")
            if attempt == settings.max_retries - 1:
                logger.error("[MasterAgent] All retries exhausted for summary")

    doc_id = str(uuid.uuid4())
    meta   = state.get("metadata") or {}

    final_json: Dict[str, Any] = {
        "document_id":      doc_id,
        "project_id":       state.get("project_id", ""),
        "document_name":    state.get("document_name", ""),
        "court":            meta.get("court", ""),
        "case_number":      meta.get("case_number", ""),
        "date_of_judgment": meta.get("date_of_judgment", ""),
        "case_category":    case_category,
        "win_indicator":    win_indicator,
        "outcome":          (reason or {}).get("outcome", ""),
        "issue":            issue,
        "petitioner_arguments": {
            "party":         "Petitioner",
            "key_arguments": args.get("petitioner_arguments", []),
            "citations":     args.get("petitioner_citations", []),
        },
        "respondent_arguments": {
            "party":         "Respondent",
            "key_arguments": args.get("respondent_arguments", []),
            "citations":     args.get("respondent_citations", []),
        },
        "statutes":   statutes,
        "precedents": prec,
        "reasoning":  reason,
        "trends": {
            "case_category":              case_category,
            "win_indicator":              win_indicator,
            "frequently_cited_sections":  [s.get("section", "") for s in statutes.get("statutes", [])],
            "top_precedents":             [p.get("case_name", "") for p in prec.get("precedents", [])[:5]],
        },
        "case_summary":   case_summary,
        "processing_log": state.get("processing_log", []),
    }

    logger.info(f"[MasterAgent] ✅ final_json built. category={case_category}, win={win_indicator}")
    return {
        "final_json":        final_json,
        "case_summary":      case_summary,
        "case_category":     case_category,
        "win_indicator":     win_indicator,
        "headnotes":         [],
        "validation_errors": [],
    }


def validation_router(state: LegalState) -> str:
    final  = state.get("final_json") or {}
    retry  = state.get("retry_count", 0)
    errors = []
    required_keys = [
        "issue", "petitioner_arguments", "respondent_arguments",
        "statutes", "precedents", "reasoning", "case_summary",
    ]
    for key in required_keys:
        if not final.get(key):
            errors.append(f"Missing: {key}")
    if errors and retry < 2:
        logger.warning(f"[Validation] Failed (attempt {retry+1}): {errors}")
        return "retry"
    if errors:
        logger.warning(f"[Validation] Proceeding despite errors after {retry} retries")
    return "db_write"


def increment_retry_node(state: LegalState) -> Dict[str, Any]:
    retry = state.get("retry_count", 0) + 1
    logger.info(f"[Retry] Incrementing retry count to {retry}")
    return {"retry_count": retry, "validation_errors": []}
