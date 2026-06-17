"""
agents/universal_chunk_extractor.py
Universal Chunk Extractor (UCE) — single LLM call per chunk.

Instead of 5 agents × 17 chunks = 85 calls,
UCE processes each chunk ONCE and extracts all fields simultaneously.

UCE is NOT a reasoning agent — it ONLY identifies and tags.
It does NOT summarize, infer, or generate conclusions.
Prompt is kept lightweight: 300–500 tokens.

Reliability guarantees
-----------------------
Every chunk passed in MUST end up with an entry in the returned list —
either a populated extraction dict, or (only after every retry has been
exhausted) an empty dict `{}`. Chunks are NEVER silently skipped.

  * Pass 1 — up to MAX_CHUNK_ATTEMPTS attempts per chunk, in order.
  * Pass 2 — any chunk still empty after Pass 1 gets FINAL_PASS_ATTEMPTS
             more attempts once the rest of the document has been
             processed (by then earlier chunks' token usage has aged
             out of the 60s TPM window, freeing up real headroom).

Token-budget guarantees
------------------------
UCE's output schema is small and fixed, so calls request a much smaller
`max_tokens` than the pipeline-wide default (8192). This keeps Groq's
"used + requested > limit" pre-flight TPM check from tripping on calls
that would never have actually consumed anywhere near that many tokens —
which previously caused frequent false-positive 429s and long forced
waits. If a response comes back truncated/invalid JSON, the retry for
that attempt uses a larger `UCE_RETRY_MAX_TOKENS` budget.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config.settings import get_settings
from backend.services.llm.token_scheduler import get_token_scheduler

logger = logging.getLogger(__name__)

_settings = get_settings()

# ── UCE System Prompt (lightweight — extraction only) ─────────────
UCE_SYSTEM = """\
You are a legal text tagger. Extract and tag legal information from the text.
DO NOT summarize. DO NOT infer. DO NOT generate conclusions.
ONLY identify and extract what is explicitly present.
Respond with valid JSON only. No markdown. No explanation."""

UCE_SCHEMA = """{
  "issues": ["exact issue text found"],
  "petitioner_arguments": ["exact argument text"],
  "respondent_arguments": ["exact argument text"],
  "statutes": [{"act": "", "section": "", "description": ""}],
  "precedents": [{"case_name": "", "citation": "", "relevance": ""}],
  "reasoning_snippets": ["exact reasoning text"],
  "outcome_clues": ["text indicating verdict/order"],
  "dispute_category": "",
  "page_refs": []
}"""

UCE_PROMPT = """\
Tag all legal information present in this text. Extract only what is explicitly stated.

Text:
{text}

Return JSON matching exactly:
{schema}

Use empty arrays for fields not found in this text."""


# ── Tunable constants ──────────────────────────────────────────────
# UCE only ever returns a small, fixed JSON tagging object — it never
# summarizes or reasons — so it does not need the pipeline-wide 8192
# max_tokens reservation. Reserving far more than is realistically used
# inflates the "requested" figure in Groq's TPM pre-flight check and
# triggers premature 429s well before the real per-minute budget is hit.
UCE_MAX_TOKENS       = _settings.uce_max_tokens         # normal extraction call
UCE_RETRY_MAX_TOKENS = _settings.uce_retry_max_tokens   # used if a previous attempt returned truncated/invalid JSON

MAX_CHUNK_ATTEMPTS   = _settings.uce_max_chunk_attempts        # in-line attempts per chunk before deferring to the final pass
FINAL_PASS_ATTEMPTS  = _settings.uce_final_pass_attempts       # extra attempts per chunk in the cleanup pass
KEY_WAIT_TIMEOUT     = _settings.uce_key_wait_timeout_seconds  # max seconds to wait for any key to free up

# Remaining-TPM thresholds for _dynamic_throttle (same units as groq_tpm_limit)
TPM_LOW_THRESHOLD  = _settings.uce_tpm_low_threshold
TPM_WARN_THRESHOLD = _settings.uce_tpm_warn_threshold


def _parse_uce_json(raw: str) -> Dict:
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
        logger.debug(f"[UCE] JSON parse failed: {e}")
        return {}


def _dynamic_throttle(scheduler, key_id: str) -> None:
    """
    Sleep only when necessary based on remaining TPM.
    Never sleep if TPM budget is healthy.
    """
    remaining = scheduler.get_remaining_tpm(key_id)
    if remaining is None:
        return
    if remaining < TPM_LOW_THRESHOLD:
        logger.info(f"[UCE] TPM low ({remaining}) on {key_id} — sleeping 8s")
        time.sleep(8)
    elif remaining < TPM_WARN_THRESHOLD:
        logger.debug(f"[UCE] TPM warning ({remaining}) — sleeping 2s")
        time.sleep(2)
    # else: healthy — no sleep


def _build_messages(chunk: str) -> List:
    return [
        SystemMessage(content=UCE_SYSTEM),
        HumanMessage(content=UCE_PROMPT.format(text=chunk, schema=UCE_SCHEMA)),
    ]


def _attempt_chunk(
    chunk: str,
    idx: int,
    total: int,
    attempt: int,
    scheduler,
    mgr,
) -> Optional[Dict]:
    """
    Single attempt to extract `chunk`.

    Returns the parsed dict on success, or None on failure — the caller
    decides whether to retry. Never raises.
    """
    from backend.services.llm.router import invoke_llm
    from agents.utils.rate_guard import wait_for_available_key

    # ── Make sure a key is actually available before spending a call ──
    status = mgr.get_pool_status()
    if status["active_keys"] == 0:
        logger.info(
            f"[UCE] Chunk {idx}/{total} attempt {attempt}: "
            f"all keys on cooldown — waiting for recovery"
        )
        wait_for_available_key(max_wait=KEY_WAIT_TIMEOUT)

    key_health = scheduler.get_best_key()
    if key_health is None:
        wait_s = min(5 * attempt, 20)
        logger.warning(
            f"[UCE] Chunk {idx}/{total} attempt {attempt}: "
            f"no key with sufficient TPM headroom — waiting {wait_s}s"
        )
        time.sleep(wait_s)
        return None

    # Use a larger budget on retries in case the first attempt was
    # truncated before it could close its JSON object.
    max_tokens = UCE_MAX_TOKENS if attempt == 1 else UCE_RETRY_MAX_TOKENS

    try:
        raw    = invoke_llm(_build_messages(chunk), max_tokens=max_tokens)
        parsed = _parse_uce_json(raw)

        if parsed:
            scheduler.record_success(
                key_health.key_id,
                estimated_tokens=len(chunk) // 4 + max_tokens,
            )
            _dynamic_throttle(scheduler, key_health.key_id)
            return parsed

        logger.warning(
            f"[UCE] Chunk {idx}/{total} attempt {attempt}: "
            f"LLM returned empty/invalid JSON (max_tokens={max_tokens}) — will retry"
        )
        return None

    except Exception as e:
        was_rate_limited = mgr.report_failure(key_health.key_id, e)
        if was_rate_limited:
            logger.warning(
                f"[UCE] Chunk {idx}/{total} attempt {attempt}: "
                f"{key_health.key_id} rate-limited — rotating keys"
            )
        else:
            logger.warning(
                f"[UCE] Chunk {idx}/{total} attempt {attempt} failed on "
                f"{key_health.key_id}: {e}"
            )
        return None


def extract_from_chunks(
    chunks: List[str],
    agent_name: str = "UCE",
) -> List[Dict[str, Any]]:
    """
    Run UCE on all chunks. Returns list of per-chunk extraction dicts —
    one entry per input chunk, in order. No chunk is ever skipped:
    every chunk gets up to MAX_CHUNK_ATTEMPTS attempts in the main pass,
    and any still-failing chunks get FINAL_PASS_ATTEMPTS more attempts
    in a cleanup pass once the rest of the document has been processed.

    Args:
        chunks:     List of text chunks from PDF
        agent_name: Label for logging

    Returns:
        List of dicts, one per chunk (empty dict only if ALL retries
        across both passes were exhausted)
    """
    from backend.services.llm.api_key_manager import get_api_key_manager
    from agents.utils.rate_guard import wait_for_available_key

    scheduler = get_token_scheduler()
    mgr       = get_api_key_manager()
    total     = len(chunks)
    results: List[Dict[str, Any]] = [{} for _ in range(total)]

    logger.info(
        f"[UCE] Processing {total} chunks "
        f"(max_tokens={UCE_MAX_TOKENS}, retry_max_tokens={UCE_RETRY_MAX_TOKENS})"
    )

    # ── Pass 1 — in-line attempts per chunk ───────────────────────────
    pending: List[int] = []
    for idx0, chunk in enumerate(chunks):
        idx = idx0 + 1
        logger.info(f"[UCE] Chunk {idx}/{total} ({len(chunk)} chars)")

        parsed = None
        for attempt in range(1, MAX_CHUNK_ATTEMPTS + 1):
            parsed = _attempt_chunk(chunk, idx, total, attempt, scheduler, mgr)
            if parsed:
                logger.info(f"[UCE] ✓ Chunk {idx}/{total} (attempt {attempt})")
                break

        if parsed:
            results[idx0] = parsed
        else:
            pending.append(idx0)
            logger.warning(
                f"[UCE] Chunk {idx}/{total}: no result after {MAX_CHUNK_ATTEMPTS} "
                f"attempts — queued for final retry pass"
            )

    # ── Pass 2 — final cleanup retry ──────────────────────────────────
    # By the time the rest of the document is done, earlier chunks'
    # token usage has aged out of the 60s sliding TPM window, so keys
    # that looked exhausted during Pass 1 often have headroom again.
    if pending:
        logger.info(
            f"[UCE] Final retry pass for {len(pending)} chunk(s): "
            f"{[i + 1 for i in pending]}"
        )
        for idx0 in pending:
            idx   = idx0 + 1
            chunk = chunks[idx0]

            parsed = None
            for attempt in range(1, FINAL_PASS_ATTEMPTS + 1):
                wait_for_available_key(max_wait=KEY_WAIT_TIMEOUT)
                parsed = _attempt_chunk(
                    chunk, idx, total, MAX_CHUNK_ATTEMPTS + attempt, scheduler, mgr,
                )
                if parsed:
                    logger.info(
                        f"[UCE] ✓ Chunk {idx}/{total} recovered on final pass "
                        f"(attempt {attempt})"
                    )
                    break

            if parsed:
                results[idx0] = parsed
            else:
                logger.error(
                    f"[UCE] ✗ Chunk {idx}/{total} FAILED after all retries "
                    f"(main + final pass) — contributing empty extraction"
                )

    succeeded = sum(1 for r in results if r)
    failed    = total - succeeded
    if failed:
        logger.warning(f"[UCE] ⚠️ {failed}/{total} chunk(s) ultimately returned empty extraction")
    logger.info(f"[UCE] ✅ Done — {succeeded}/{total} chunks extracted")
    return results
