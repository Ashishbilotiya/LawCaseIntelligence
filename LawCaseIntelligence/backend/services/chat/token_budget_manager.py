"""
backend/services/chat/token_budget_manager.py
Token-aware budget allocation and pre-request validation.

TOTAL_REQUEST_BUDGET = 7000 tokens
  ├── system_prompt   : 500
  ├── user_query      : 300
  ├── rag_context     : 2500
  ├── chat_history    : 1000
  └── output_reserved : 2500  (not sent, reserved for generation)

Input budget = 7000 - 2500 = 4500 tokens
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# ── Budget constants ──────────────────────────────────────────────
TOTAL_REQUEST_BUDGET  = _settings.chat_total_request_budget
OUTPUT_BUDGET         = _settings.chat_output_budget
INPUT_BUDGET          = TOTAL_REQUEST_BUDGET - OUTPUT_BUDGET   # 4500

SYSTEM_PROMPT_BUDGET  = _settings.chat_system_prompt_budget
USER_QUERY_BUDGET     = _settings.chat_user_query_budget
RAG_CONTEXT_BUDGET    = _settings.chat_rag_context_budget
CHAT_HISTORY_BUDGET   = _settings.chat_history_budget
SUMMARY_BUDGET        = _settings.chat_summary_budget

# tiktoken is optional — fallback to char/4 estimate
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(text: str) -> int:
        return len(_enc.encode(str(text)))
except Exception:
    def estimate_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(str(text)) // 4)


@dataclass
class BudgetAllocation:
    system_tokens:  int = 0
    query_tokens:   int = 0
    rag_tokens:     int = 0
    history_tokens: int = 0
    summary_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return (
            self.system_tokens + self.query_tokens +
            self.rag_tokens + self.history_tokens + self.summary_tokens
        )

    @property
    def fits_budget(self) -> bool:
        return self.total_input_tokens <= INPUT_BUDGET

    @property
    def remaining(self) -> int:
        return max(0, INPUT_BUDGET - self.total_input_tokens)


def validate_and_fit(
    system_prompt: str,
    user_query: str,
    rag_chunks: List[str],
    history_messages: List[str],
    summary: str = "",
) -> dict:
    """
    Validate that the assembled prompt fits INPUT_BUDGET.
    If it doesn't, prune in this order:
      1. oldest history messages
      2. compress summary
      3. trim rag chunks from the bottom (lowest ranked)

    Returns:
      {
        "rag_chunks": [...],        # possibly reduced
        "history_messages": [...],  # possibly pruned
        "summary": str,             # possibly compressed
        "allocation": BudgetAllocation,
        "was_pruned": bool,
      }
    """
    was_pruned = False

    system_toks  = estimate_tokens(system_prompt)
    query_toks   = estimate_tokens(user_query)
    summary_toks = estimate_tokens(summary)

    # Fixed costs
    fixed = system_toks + query_toks + summary_toks
    remaining_for_dynamic = INPUT_BUDGET - fixed

    # ── 1. Fit history into CHAT_HISTORY_BUDGET ───────────────────
    history_budget = min(CHAT_HISTORY_BUDGET, remaining_for_dynamic - RAG_CONTEXT_BUDGET)
    history_budget = max(0, history_budget)
    pruned_history, hist_toks = _fit_texts(history_messages, history_budget, from_start=True)
    if len(pruned_history) < len(history_messages):
        was_pruned = True

    # ── 2. Remaining budget goes to RAG ───────────────────────────
    rag_budget = min(RAG_CONTEXT_BUDGET, INPUT_BUDGET - fixed - hist_toks)
    rag_budget = max(0, rag_budget)
    pruned_rag, rag_toks = _fit_texts(rag_chunks, rag_budget, from_start=False)
    if len(pruned_rag) < len(rag_chunks):
        was_pruned = True

    # ── 3. If still over budget, compress summary ─────────────────
    total = fixed + hist_toks + rag_toks
    if total > INPUT_BUDGET and len(summary) > 100:
        max_summary_chars = max(50, (INPUT_BUDGET - system_toks - query_toks - hist_toks - rag_toks) * 4)
        summary = summary[:max_summary_chars] + "…"
        summary_toks = estimate_tokens(summary)
        was_pruned = True

    allocation = BudgetAllocation(
        system_tokens=system_toks,
        query_tokens=query_toks,
        rag_tokens=rag_toks,
        history_tokens=hist_toks,
        summary_tokens=summary_toks,
    )

    logger.info(
        f"[TokenBudget] sys={system_toks} query={query_toks} "
        f"rag={rag_toks} hist={hist_toks} sum={summary_toks} "
        f"total={allocation.total_input_tokens}/{INPUT_BUDGET} pruned={was_pruned}"
    )

    return {
        "rag_chunks":       pruned_rag,
        "history_messages": pruned_history,
        "summary":          summary,
        "allocation":       allocation,
        "was_pruned":       was_pruned,
    }


def fit_chunks_to_budget(chunks: List[str], budget_tokens: int = RAG_CONTEXT_BUDGET) -> List[str]:
    """
    Add chunks (highest-ranked first) until token budget is reached.
    Returns the subset of chunks that fits.
    """
    result, used = [], 0
    for chunk in chunks:
        t = estimate_tokens(chunk)
        if used + t > budget_tokens:
            break
        result.append(chunk)
        used += t
    return result


def _fit_texts(texts: List[str], budget: int, from_start: bool) -> tuple:
    """
    Fit texts into a token budget.
    from_start=True  → drop oldest (front) first  [history]
    from_start=False → drop lowest (back) first   [rag chunks]
    """
    if not texts or budget <= 0:
        return [], 0

    total = sum(estimate_tokens(t) for t in texts)
    if total <= budget:
        return list(texts), total

    result = list(texts)
    while result and sum(estimate_tokens(t) for t in result) > budget:
        if from_start:
            result.pop(0)   # remove oldest
        else:
            result.pop()    # remove lowest-ranked

    used = sum(estimate_tokens(t) for t in result)
    return result, used
