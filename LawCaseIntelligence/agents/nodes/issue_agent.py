"""
agents/nodes/issue_agent.py
Issue Supervisor Agent — receives UCE-extracted issues and synthesizes final output.

New role:
  INPUT:  List of issue extractions from UCE (one per chunk)
  OUTPUT: Deduplicated, prioritized, final issue output

No longer calls LLM per-chunk. Makes ONE synthesis call on aggregated data.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state.legal_state import LegalState
from agents.utils.rate_guard import safe_invoke
from backend.models.issue_models import IssueOutput

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """\
You are an expert Indian legal analyst.
You will receive extracted legal issue snippets from multiple sections of a judgment.
Your task: deduplicate, prioritize, and identify the primary legal issue.
Respond with valid JSON only. No markdown. No explanation."""

SUPERVISOR_SCHEMA = """{
  "primary_issue": "string — the single central legal question",
  "sub_issues": ["string"],
  "case_background": "string — brief factual background",
  "dispute_category": "string — Civil/Criminal/Constitutional/Employment/Property/Tax/Family/General",
  "source_pages": [],
  "headnote": "string — one-sentence headnote"
}"""

SUPERVISOR_PROMPT = """\
These issue snippets were extracted from different sections of a court judgment:

{extracted}

Synthesize into a single coherent issue analysis.
Identify the ONE primary legal issue.
Remove duplicates. Prioritize by importance.

Return JSON matching exactly:
{schema}"""


def _parse_json(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[IssueAgent] Parse error: {e}")
        return {}


def _aggregate_from_uce(uce_results: List[Dict]) -> List[str]:
    """Collect all issue snippets from UCE results."""
    issues = []
    seen: set = set()
    for r in uce_results:
        for item in r.get("issues", []):
            k = str(item).strip().lower()[:100]
            if k and k not in seen:
                seen.add(k)
                issues.append(str(item))
    return issues


def _fallback_output(uce_results: List[Dict]) -> Dict:
    """Build basic output from UCE data without LLM synthesis."""
    issues   = _aggregate_from_uce(uce_results)
    category = next(
        (r.get("dispute_category") for r in uce_results if r.get("dispute_category")), ""
    )
    return {
        "primary_issue":    issues[0] if issues else "",
        "sub_issues":       issues[1:5] if len(issues) > 1 else [],
        "case_background":  "",
        "dispute_category": category,
        "source_pages":     [],
        "headnote":         "",
    }


def issue_agent_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ [1/5] Issue Supervisor Agent")
    uce_results: List[Dict] = state.get("uce_results", [])

    if not uce_results:
        logger.warning("[IssueAgent] No UCE results — using empty output")
        return {
            "issue_output":   IssueOutput().model_dump(),
            "processing_log": state.get("processing_log", []) + ["IssueAgent: no UCE data"],
        }

    issues = _aggregate_from_uce(uce_results)
    if not issues:
        logger.warning("[IssueAgent] No issues found in UCE results")
        return {
            "issue_output":   IssueOutput().model_dump(),
            "processing_log": state.get("processing_log", []) + ["IssueAgent: no issues found"],
        }

    extracted_text = "\n".join(f"- {i}" for i in issues[:30])  # cap at 30

    raw = safe_invoke(
        messages=[
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=SUPERVISOR_PROMPT.format(
                extracted=extracted_text,
                schema=SUPERVISOR_SCHEMA,
            )),
        ],
        chunk_idx=1,
        agent_name="IssueAgent",
    )

    if raw:
        parsed = _parse_json(raw)
        if parsed and parsed.get("primary_issue"):
            logger.info(f"[IssueAgent] ✅ primary_issue='{parsed['primary_issue'][:60]}'")
            return {
                "issue_output":   parsed,
                "processing_log": state.get("processing_log", []) + ["IssueAgent: synthesis done"],
            }

    # Fallback: build from UCE data without LLM
    logger.warning("[IssueAgent] LLM synthesis failed — using fallback")
    return {
        "issue_output":   _fallback_output(uce_results),
        "processing_log": state.get("processing_log", []) + ["IssueAgent: fallback used"],
    }
