"""
agents/nodes/argument_agent.py
Argument Supervisor Agent — receives UCE-extracted arguments and merges them.

New role:
  INPUT:  UCE results with petitioner/respondent arguments
  OUTPUT: Deduplicated, organized argument output

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
from backend.models.argument_models import ArgumentOutput

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """\
You are an expert Indian legal analyst specializing in argument analysis.
You will receive argument snippets extracted from multiple sections of a judgment.
Merge, deduplicate, and organize into petitioner and respondent arguments.
Respond with valid JSON only. No markdown. No explanation."""

SUPERVISOR_SCHEMA = """{
  "petitioner_arguments": ["string"],
  "petitioner_citations": ["string"],
  "respondent_arguments": ["string"],
  "respondent_citations": ["string"],
  "key_contentions": ["string"],
  "headnote": "string"
}"""

SUPERVISOR_PROMPT = """\
Petitioner arguments extracted:
{petitioner}

Respondent arguments extracted:
{respondent}

Merge, remove duplicates, organize clearly.
Return JSON matching exactly:
{schema}"""


def _dedup(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        k = str(item).strip().lower()[:100]
        if k and k not in seen:
            seen.add(k); out.append(str(item))
    return out


def _aggregate(uce_results: List[Dict]) -> tuple:
    pet_args = _dedup([a for r in uce_results for a in r.get("petitioner_arguments", [])])
    res_args = _dedup([a for r in uce_results for a in r.get("respondent_arguments", [])])
    return pet_args, res_args


def _parse_json(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if m: raw = m.group(0)
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"[ArgumentAgent] Parse error: {e}")
        return {}


def argument_agent_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ [2/5] Argument Supervisor Agent")
    uce_results: List[Dict] = state.get("uce_results", [])

    if not uce_results:
        return {"argument_output": ArgumentOutput().model_dump()}

    pet_args, res_args = _aggregate(uce_results)

    if not pet_args and not res_args:
        return {
            "argument_output": ArgumentOutput().model_dump(),
            "processing_log": state.get("processing_log", []) + ["ArgumentAgent: no args found"],
        }

    raw = safe_invoke(
        messages=[
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=SUPERVISOR_PROMPT.format(
                petitioner="\n".join(f"- {a}" for a in pet_args[:20]),
                respondent="\n".join(f"- {a}" for a in res_args[:20]),
                schema=SUPERVISOR_SCHEMA,
            )),
        ],
        chunk_idx=1,
        agent_name="ArgumentAgent",
    )

    if raw:
        parsed = _parse_json(raw)
        if parsed:
            logger.info(f"[ArgumentAgent] ✅ {len(parsed.get('petitioner_arguments',[]))} pet / "
                        f"{len(parsed.get('respondent_arguments',[]))} res args")
            return {
                "argument_output": parsed,
                "processing_log":  state.get("processing_log", []) + ["ArgumentAgent: synthesis done"],
            }

    # Fallback: use raw UCE data
    logger.warning("[ArgumentAgent] LLM synthesis failed — using fallback")
    return {
        "argument_output": {
            "petitioner_arguments": pet_args,
            "petitioner_citations": [],
            "respondent_arguments": res_args,
            "respondent_citations": [],
            "key_contentions":      [],
            "headnote":             "",
        },
        "processing_log": state.get("processing_log", []) + ["ArgumentAgent: fallback used"],
    }
