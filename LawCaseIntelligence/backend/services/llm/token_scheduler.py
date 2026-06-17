"""
backend/services/llm/token_scheduler.py
Score-based key selection — reads directly from TokenTracker (single source of truth).

Removed: duplicated TPM tracking (KeyTPMState with its own tokens_used).
Now: TokenScheduler.get_best_key() queries TokenTracker.get_tpm(key_id)
     for actual remaining-TPM data.

Score formula (weights configurable via backend.config.settings):
    score = remaining_tpm_pct * scheduler_weight_remaining_tpm  (default 0.70)
          + success_rate      * scheduler_weight_success_rate   (default 0.20)
          + idle_time_score   * scheduler_weight_idle_time       (default 0.10)

Highest score wins. Used by UCE + supervisor agents.
Chatbot continues using APIKeyManager.get_next_key() (round-robin) — unchanged.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from backend.config.settings import get_settings
from .api_key_manager import get_api_key_manager
from .token_tracker import get_token_tracker

logger = logging.getLogger(__name__)

_settings = get_settings()

# All tunables below come from backend.config.settings so the scheduler,
# diagnostics, and UCE throttling all agree on the same numbers — see
# Settings: groq_tpm_limit, scheduler_* fields.
#
# DEFAULT_TPM_LIMIT is the Groq llama-3.3-70b-versatile per-key TPM limit.
# CONFIRMED from live Groq 429 responses (RateLimitClassifier logs show
# "limit=12000" on every TPM error) — this is the ground truth.
DEFAULT_TPM_LIMIT        = _settings.groq_tpm_limit
SAFE_THRESHOLD           = _settings.scheduler_safe_threshold_tpm
IDLE_FULL_CREDIT_SECONDS = _settings.scheduler_idle_full_credit_seconds
WEIGHT_REMAINING_TPM     = _settings.scheduler_weight_remaining_tpm
WEIGHT_SUCCESS_RATE      = _settings.scheduler_weight_success_rate
WEIGHT_IDLE_TIME         = _settings.scheduler_weight_idle_time


class TokenScheduler:
    """
    Stateless scorer — all token data comes from TokenTracker.
    Selects the highest-scoring active, non-cooldown, non-daily-exhausted key.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────

    def get_best_key(self, tpm_limit: int = DEFAULT_TPM_LIMIT, reserve_tokens: Optional[int] = None):
        """
        Return the KeyHealth with the highest composite score.
        Returns None if no key has sufficient remaining TPM.

        If `reserve_tokens` is given, immediately reserves that many tokens
        against the chosen key in TokenTracker (before its LLM call even
        starts). This prevents the "thundering herd" race where several
        concurrent callers all see the same idle key (because none of
        their usage has been recorded yet) and all pile onto it. The
        caller MUST release this reservation via
        TokenTracker.release_reservation(key_id) once the call completes.
        """
        mgr     = get_api_key_manager()
        tracker = get_token_tracker()
        pool    = mgr.get_pool()   # already filters via try_recover internally on access

        candidates = [
            kh for kh in pool
            if kh.is_active and not kh.in_cooldown and not kh.daily_exhausted
        ]
        if not candidates:
            logger.info("[TokenScheduler] No active keys available")
            return None

        scored: List[tuple] = []
        for kh in candidates:
            metrics = tracker.get_metrics(kh.key_id)

            # Actual TPM usage (last 60s) PLUS any in-flight reservations
            # from calls that were just dispatched but haven't returned yet.
            tpm_used       = tracker.get_total_tpm(kh.key_id)
            remaining_tpm  = max(0, tpm_limit - tpm_used)
            remaining_pct  = remaining_tpm / tpm_limit if tpm_limit else 0.0

            success_rate = kh.success_rate  # 0..1

            if metrics and metrics.total_requests > 0:
                last_used_monotonic = metrics.last_activity_monotonic()
                idle_seconds = time.monotonic() - last_used_monotonic if last_used_monotonic else IDLE_FULL_CREDIT_SECONDS
            else:
                idle_seconds = IDLE_FULL_CREDIT_SECONDS  # never used → fully idle

            idle_score = min(1.0, idle_seconds / IDLE_FULL_CREDIT_SECONDS)

            score = (
                remaining_pct * WEIGHT_REMAINING_TPM +
                success_rate  * WEIGHT_SUCCESS_RATE +
                idle_score    * WEIGHT_IDLE_TIME
            )

            scored.append((score, kh, remaining_tpm, remaining_pct, success_rate, idle_score))

            logger.debug(
                f"[TokenScheduler] {kh.key_id}: score={score:.3f} "
                f"(remaining_tpm={remaining_tpm}/{tpm_limit} [{remaining_pct:.2f}], "
                f"success_rate={success_rate:.2f}, idle_score={idle_score:.2f})"
            )

            # Filter out keys with too little headroom
            if remaining_tpm < SAFE_THRESHOLD:
                logger.debug(
                    f"[TokenScheduler] {kh.key_id} below safe threshold "
                    f"({remaining_tpm} < {SAFE_THRESHOLD} TPM)"
                )

        # Only consider keys with sufficient headroom
        eligible = [s for s in scored if s[2] >= SAFE_THRESHOLD]

        if not eligible:
            best_remaining = max((s[2] for s in scored), default=0)
            logger.info(
                f"[TokenScheduler] All {len(scored)} keys below safe threshold "
                f"({SAFE_THRESHOLD} TPM). Best remaining={best_remaining}"
            )
            return None

        eligible.sort(key=lambda s: s[0], reverse=True)
        best_score, best_kh, best_remaining, best_pct, best_sr, best_idle = eligible[0]

        if reserve_tokens:
            tracker.reserve(best_kh.key_id, reserve_tokens)
            logger.info(
                f"[TokenScheduler] Selected {best_kh.key_id} "
                f"(score={best_score:.3f}, remaining_tpm={best_remaining}, "
                f"success_rate={best_sr:.2f}, idle_score={best_idle:.2f}, "
                f"reserved={reserve_tokens})"
            )
        else:
            logger.info(
                f"[TokenScheduler] Selected {best_kh.key_id} "
                f"(score={best_score:.3f}, remaining_tpm={best_remaining}, "
                f"success_rate={best_sr:.2f}, idle_score={best_idle:.2f})"
            )
        return best_kh

    def get_remaining_tpm(self, key_id: str, tpm_limit: int = DEFAULT_TPM_LIMIT) -> int:
        """Delegate to TokenTracker — single source of truth."""
        return get_token_tracker().get_remaining_tpm(key_id, tpm_limit)

    def record_success(
        self,
        key_id: str,
        estimated_tokens: int = 0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """
        Backward-compat hook for callers (e.g. UCE) that historically tracked
        their own per-key usage after a successful LLM call.

        IMPORTANT: This is intentionally a NO-OP for token accounting.
        GroqProvider.invoke()/stream() already records actual usage into
        TokenTracker via _record_usage() for the key that was *actually*
        used for the call. Recording again here — possibly for a different
        key_id than the one GroqProvider ended up using internally — would
        double-count tokens in TokenTracker and re-introduce exactly the
        "multiple TPM tracking systems" / false-cooldown problem this
        refactor removes.

        We still log it so the call site's intent is visible in diagnostics.
        """
        tracker = get_token_tracker()
        metrics = tracker.get_metrics(key_id)
        tpm_now = metrics.tpm_current if metrics else 0
        logger.debug(
            f"[TokenScheduler] record_success({key_id}) — no-op "
            f"(TokenTracker is single source of truth; current tpm={tpm_now})"
        )

    def get_status(self) -> List[dict]:
        """Return scoring breakdown for all keys (diagnostics)."""
        mgr     = get_api_key_manager()
        tracker = get_token_tracker()
        result  = []

        for kh in mgr.get_pool():
            metrics = tracker.get_metrics(kh.key_id)
            tpm_used  = metrics.tpm_current if metrics else 0
            reserved  = tracker.get_reserved_tpm(kh.key_id)
            remaining = max(0, DEFAULT_TPM_LIMIT - tpm_used - reserved)
            result.append({
                "key_id":        kh.key_id,
                "remaining_tpm": remaining,
                "tpm_limit":     DEFAULT_TPM_LIMIT,
                "tpm_used":      tpm_used,
                "reserved_tpm":  reserved,
                "success_rate":  round(kh.success_rate, 3),
                "is_active":     kh.is_active,
                "in_cooldown":   kh.in_cooldown,
                "daily_exhausted": kh.daily_exhausted,
            })
        return result


# ── Singleton ─────────────────────────────────────────────────────

_scheduler: Optional[TokenScheduler] = None
_sched_lock = threading.Lock()


def get_token_scheduler() -> TokenScheduler:
    global _scheduler
    if _scheduler is None:
        with _sched_lock:
            if _scheduler is None:
                _scheduler = TokenScheduler()
    return _scheduler
