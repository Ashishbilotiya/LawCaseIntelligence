"""
agents/nodes/law_statute_agent.py
Law & Statute Supervisor Agent — receives UCE-extracted statutes and deduplicates.

New role:
  INPUT:  UCE results with statutes[]
  OUTPUT: Deduplicated, grouped, ranked statute output

ONE synthesis LLM call (not 17).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state.legal_state import LegalState
from agents.utils.rate_guard import safe_invoke
from backend.models.statute_models import StatuteOutput

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """\
You are an expert Indian legal analyst specializing in statutory law.
You will receive statute snippets from multiple sections of a judgment.
Deduplicate, group related sections, and rank by importance.
Respond with valid JSON only. No markdown. No explanation."""

SUPERVISOR_SCHEMA = """{
  "statutes": [{"act": "", "section": "", "description": "", "relevance": ""}],
  "constitutional_provisions": ["string"],
  "regulatory_references": ["string"],
  "source_pages": [],
  "headnote": "string"
}"""

SUPERVISOR_PROMPT = """\
Statutes extracted from judgment:
{statutes}

Constitutional provisions found:
{constitutional}

Deduplicate, group by Act, rank by relevance.
Return JSON matching exactly:
{schema}"""


def _aggregate_statutes(uce_results: List[Dict]) -> tuple:
    seen, statutes = set(), []
    for r in uce_results:
        for s in r.get("statutes", []):
            key = f"{s.get('act','').lower()}|{s.get('section','').lower()}"
            if key != "|" and key not in seen:
                seen.add(key); statutes.append(s)
    const_seen, constitutional = set(), []
    for r in uce_results:
        for c in r.get("page_refs", []):
            if isinstance(c, str) and "article" in c.lower():
                k = c.lower().strip()
                if k not in const_seen:
                    const_seen.add(k); constitutional.append(c)
    return statutes, constitutional


def _parse_json(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if m: raw = m.group(0)
        data = json.loads(raw)
        return StatuteOutput(**data).model_dump()
    except Exception as e:
        logger.warning(f"[LawAgent] Parse error: {e}")
        return {}


def law_statute_agent_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ [3/5] Law & Statute Supervisor Agent")
    uce_results: List[Dict] = state.get("uce_results", [])

    if not uce_results:
        return {"statute_output": StatuteOutput().model_dump()}

    statutes, constitutional = _aggregate_statutes(uce_results)

    if not statutes and not constitutional:
        return {
            "statute_output": StatuteOutput().model_dump(),
            "processing_log": state.get("processing_log", []) + ["LawAgent: no statutes found"],
        }

    statutes_text = "\n".join(
        f"- {s.get('act','')} § {s.get('section','')} — {s.get('description','')}"
        for s in statutes[:25]
    )

    raw = safe_invoke(
        messages=[
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=SUPERVISOR_PROMPT.format(
                statutes=statutes_text or "None found",
                constitutional="\n".join(f"- {c}" for c in constitutional) or "None found",
                schema=SUPERVISOR_SCHEMA,
            )),
        ],
        chunk_idx=1,
        agent_name="LawAgent",
    )

    if raw:
        parsed = _parse_json(raw)
        if parsed:
            logger.info(f"[LawAgent] ✅ {len(parsed.get('statutes', []))} statutes")
            return {
                "statute_output": parsed,
                "processing_log": state.get("processing_log", []) + [
                    f"LawAgent: {len(parsed.get('statutes', []))} statutes"
                ],
            }

    # Fallback
    logger.warning("[LawAgent] LLM failed — using fallback")
    return {
        "statute_output": {
            "statutes": statutes,
            "constitutional_provisions": constitutional,
            "regulatory_references": [],
            "source_pages": [],
            "headnote": "",
        },
        "processing_log": state.get("processing_log", []) + ["LawAgent: fallback used"],
    }
