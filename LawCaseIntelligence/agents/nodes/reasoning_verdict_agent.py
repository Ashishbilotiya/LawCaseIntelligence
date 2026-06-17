"""
agents/nodes/reasoning_verdict_agent.py
Reasoning & Verdict Supervisor Agent — synthesizes reasoning from UCE snippets.

New role:
  INPUT:  UCE results with reasoning_snippets[] and outcome_clues[]
  OUTPUT: Synthesized judicial reasoning and verdict

ONE synthesis LLM call (not 17).
Prioritizes outcome_clues from last 40% of chunks (verdict is usually at end).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agents.state.legal_state import LegalState
from agents.utils.rate_guard import safe_invoke
from backend.models.reasoning_models import ReasoningOutput

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = """\
You are an expert Indian legal analyst specializing in judicial reasoning and verdicts.
You will receive reasoning snippets and outcome clues from a court judgment.
Synthesize the court's reasoning and determine the final verdict.
Respond with valid JSON only. No markdown. No explanation."""

SUPERVISOR_SCHEMA = """{
  "judicial_reasoning": "string — synthesized court reasoning",
  "key_findings": ["string"],
  "principles_applied": ["string"],
  "outcome": "string — Appeal Allowed/Dismissed/Petition Allowed/Dismissed/Partly Allowed/Remanded",
  "relief_granted": "string",
  "conditions": ["string"],
  "source_pages": [],
  "headnote": "string"
}"""

SUPERVISOR_PROMPT = """\
Reasoning snippets from judgment:
{reasoning}

Outcome/verdict clues (from end of document — most reliable):
{outcomes}

Synthesize the court's reasoning and determine the final outcome.
Return JSON matching exactly:
{schema}"""


def _aggregate_reasoning(uce_results: List[Dict]) -> tuple:
    total = len(uce_results)
    cutoff = max(0, total - max(1, total * 4 // 10))

    # Outcome clues prioritize end of document
    end_results  = uce_results[cutoff:]
    all_results  = uce_results

    reasoning_snippets = []
    seen_r: set = set()
    for r in all_results:
        for snippet in r.get("reasoning_snippets", []):
            k = str(snippet).strip().lower()[:80]
            if k and k not in seen_r:
                seen_r.add(k); reasoning_snippets.append(str(snippet))

    outcome_clues = []
    seen_o: set = set()
    for r in end_results:
        for clue in r.get("outcome_clues", []):
            k = str(clue).strip().lower()[:80]
            if k and k not in seen_o:
                seen_o.add(k); outcome_clues.append(str(clue))

    # Also include outcome clues from all results if end is sparse
    if len(outcome_clues) < 2:
        for r in all_results:
            for clue in r.get("outcome_clues", []):
                k = str(clue).strip().lower()[:80]
                if k and k not in seen_o:
                    seen_o.add(k); outcome_clues.append(str(clue))

    return reasoning_snippets, outcome_clues


def _parse_json(raw: str) -> Dict:
    if not raw:
        return {}
    try:
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        m   = re.search(r"\{.*\}", raw, re.DOTALL)
        if m: raw = m.group(0)
        data = json.loads(raw)
        return ReasoningOutput(**data).model_dump()
    except Exception as e:
        logger.warning(f"[ReasoningAgent] Parse error: {e}")
        return {}


def reasoning_verdict_agent_node(state: LegalState) -> Dict[str, Any]:
    logger.info("▶ [5/5] Reasoning & Verdict Supervisor Agent")
    uce_results: List[Dict] = state.get("uce_results", [])

    if not uce_results:
        return {"reasoning_output": ReasoningOutput().model_dump()}

    reasoning_snippets, outcome_clues = _aggregate_reasoning(uce_results)

    if not reasoning_snippets and not outcome_clues:
        return {
            "reasoning_output": ReasoningOutput().model_dump(),
            "processing_log": state.get("processing_log", []) + ["ReasoningAgent: no data found"],
        }

    raw = safe_invoke(
        messages=[
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=SUPERVISOR_PROMPT.format(
                reasoning="\n".join(f"- {s}" for s in reasoning_snippets[:20]),
                outcomes="\n".join(f"- {o}" for o in outcome_clues[:10]),
                schema=SUPERVISOR_SCHEMA,
            )),
        ],
        chunk_idx=1,
        agent_name="ReasoningAgent",
    )

    if raw:
        parsed = _parse_json(raw)
        if parsed:
            logger.info(f"[ReasoningAgent] ✅ outcome='{parsed.get('outcome','?')}'")
            return {
                "reasoning_output": parsed,
                "processing_log":   state.get("processing_log", []) + [
                    f"ReasoningAgent: outcome='{parsed.get('outcome','?')}'"
                ],
            }

    # Fallback
    logger.warning("[ReasoningAgent] LLM failed — using fallback")
    return {
        "reasoning_output": {
            "judicial_reasoning": " ".join(reasoning_snippets[:3]),
            "key_findings":       [],
            "principles_applied": [],
            "outcome":            outcome_clues[0] if outcome_clues else "",
            "relief_granted":     "",
            "conditions":         [],
            "source_pages":       [],
            "headnote":           "",
        },
        "processing_log": state.get("processing_log", []) + ["ReasoningAgent: fallback used"],
    }
