"""
agents/nodes/precedent_agent.py
Precedent Supervisor Agent — receives UCE-extracted precedents and ranks them.

New role:
  INPUT:  UCE results with precedents[]
  OUTPUT: Deduplicated, scored, ranked precedent output

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
from backend.models.precedent_models import PrecedentOutput

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """\
You are an expert Indian legal analyst specializing in case law.
You will receive precedent citations extracted from multiple sections of a judgment.
Deduplicate, score importance, and rank precedents.
Respond with valid JSON only. No markdown. No explanation."""

SUPERVISOR_SCHEMA = """{
  "precedents": [
    {
      "case_name": "",
      "citation": "",
      "court": "",
      "year": "",
      "relevance": "",
      "distinguished_or_followed": "",
      "importance_score": 3
    }
  ],
  "total_citations": 0,
  "source_pages": [],
  "headnote": "string"
}"""

SUPERVISOR_PROMPT = """\
Precedents extracted from judgment sections:
{precedents}

Deduplicate by case name.
Score importance 1-5 (5=central precedent, 1=minor reference).
Rank by importance_score descending.

Return JSON matching exactly:
{schema}"""


def _aggregate_precedents(uce_results: List[Dict]) -> List[Dict]:
    seen, precs = set(), []
    for r in uce_results:
        for p in r.get("precedents", []):
            key = p.get("case_name", "").lower().strip()[:60]
            if key and key not in seen:
                seen.add(key); precs.append(p)
    return precs


def _parse_json(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if m: raw = m.group(0)
        data = json.loads(raw)
        return PrecedentOutput(**data).model_dump()
    except Exception as e:
        logger.warning(f"[PrecedentAgent] Parse error: {e}")
        return {}


def precedent_agent_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ [4/5] Precedent Supervisor Agent")
    uce_results: List[Dict] = state.get("uce_results", [])

    if not uce_results:
        return {"precedent_output": PrecedentOutput().model_dump()}

    precedents = _aggregate_precedents(uce_results)

    if not precedents:
        return {
            "precedent_output": PrecedentOutput().model_dump(),
            "processing_log": state.get("processing_log", []) + ["PrecedentAgent: no precedents found"],
        }

    prec_text = "\n".join(
        f"- {p.get('case_name','')} | {p.get('citation','')} | {p.get('relevance','')}"
        for p in precedents[:30]
    )

    raw = safe_invoke(
        messages=[
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=SUPERVISOR_PROMPT.format(
                precedents=prec_text,
                schema=SUPERVISOR_SCHEMA,
            )),
        ],
        chunk_idx=1,
        agent_name="PrecedentAgent",
    )

    if raw:
        parsed = _parse_json(raw)
        if parsed:
            count = parsed.get("total_citations", len(parsed.get("precedents", [])))
            logger.info(f"[PrecedentAgent] ✅ {count} unique precedents")
            return {
                "precedent_output": parsed,
                "processing_log":   state.get("processing_log", []) + [
                    f"PrecedentAgent: {count} precedents"
                ],
            }

    # Fallback
    logger.warning("[PrecedentAgent] LLM failed — using fallback")
    fallback_precs = sorted(precedents, key=lambda p: p.get("importance_score", 1), reverse=True)
    return {
        "precedent_output": {
            "precedents":      fallback_precs,
            "total_citations": len(fallback_precs),
            "source_pages":    [],
            "headnote":        "",
        },
        "processing_log": state.get("processing_log", []) + ["PrecedentAgent: fallback used"],
    }
