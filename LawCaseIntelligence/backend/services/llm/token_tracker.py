"""
backend/services/llm/token_tracker.py
Single source of truth for all per-key token and request metrics.

Tracks (sliding windows):
  - TPM  tokens per minute   (60s window)
  - RPM  requests per minute (60s window)
  - TPD  tokens per day      (24h window)
  - RPD  requests per day    (24h window)

Supports ACTUAL token counts from provider responses
(prompt_tokens + completion_tokens) — not just estimates.

All other components (TokenScheduler, APIKeyManager diagnostics)
query THIS tracker. No separate counters elsewhere.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Optional, Tuple

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# TPM/RPM are defined by Groq over a rolling 60s window; TPD/RPD over a
# rolling 24h window. Centralized in settings.tpm_window_seconds /
# settings.tpd_window_seconds so no file declares its own copy.
MINUTE_WINDOW = _settings.tpm_window_seconds
DAY_WINDOW    = _settings.tpd_window_seconds

# How long an in-flight token reservation counts toward a key's TPM usage
# if it's never explicitly released (see TokenTracker.reserve/release_reservation).
RESERVATION_TTL = _settings.scheduler_reservation_ttl_seconds


@dataclass
class KeyTokenMetrics:
    """Sliding-window token + request metrics for a single API key."""
    key_id: str

    # Sliding windows: (monotonic_timestamp, token_count)
    _req_minute: Deque[float]              = field(default_factory=deque, init=False)
    _req_day:    Deque[float]              = field(default_factory=deque, init=False)
    _tok_minute: Deque[Tuple[float, int]]  = field(default_factory=deque, init=False)
    _tok_day:    Deque[Tuple[float, int]]  = field(default_factory=deque, init=False)

    # In-flight reservations: (monotonic_timestamp, estimated_tokens).
    # Added by TokenScheduler.get_best_key() the instant a key is chosen,
    # before the LLM call even starts — see reserved_tokens().
    _reservations: Deque[Tuple[float, int]] = field(default_factory=deque, init=False)

    # Lifetime totals
    total_requests:          int = 0
    total_tokens:            int = 0
    total_prompt_tokens:     int = 0
    total_completion_tokens: int = 0
    actual_token_records:    int = 0   # how many used real provider counts
    estimated_token_records: int = 0   # how many used estimates

    def record(
        self,
        tokens: int,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """
        Record one LLM call.
        If prompt_tokens / completion_tokens are provided (from response.usage),
        use them as the authoritative count. Otherwise use `tokens` as estimate.
        """
        now = time.monotonic()

        # Use actual counts if available
        if prompt_tokens is not None and completion_tokens is not None:
            actual_total = prompt_tokens + completion_tokens
            self.total_prompt_tokens     += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.actual_token_records    += 1
            record_tokens = actual_total
        else:
            self.estimated_token_records += 1
            record_tokens = tokens

        self._req_minute.append(now)
        self._req_day.append(now)
        self._tok_minute.append((now, record_tokens))
        self._tok_day.append((now, record_tokens))
        self.total_requests += 1
        self.total_tokens   += record_tokens
        self._prune(now)

    def _prune(self, now: float) -> None:
        minute_cutoff = now - MINUTE_WINDOW
        day_cutoff    = now - DAY_WINDOW
        while self._req_minute  and self._req_minute[0]  < minute_cutoff: self._req_minute.popleft()
        while self._req_day     and self._req_day[0]     < day_cutoff:    self._req_day.popleft()
        while self._tok_minute  and self._tok_minute[0][0] < minute_cutoff: self._tok_minute.popleft()
        while self._tok_day     and self._tok_day[0][0]    < day_cutoff:    self._tok_day.popleft()

    @property
    def rpm_current(self) -> int:
        self._prune(time.monotonic())
        return len(self._req_minute)

    @property
    def rpd_current(self) -> int:
        self._prune(time.monotonic())
        return len(self._req_day)

    @property
    def tpm_current(self) -> int:
        self._prune(time.monotonic())
        return sum(tk for _, tk in self._tok_minute)

    @property
    def tpd_current(self) -> int:
        self._prune(time.monotonic())
        return sum(tk for _, tk in self._tok_day)

    def last_activity_monotonic(self) -> Optional[float]:
        """
        Most recent monotonic timestamp of any request, after pruning.
        Used by TokenScheduler's idle-time scoring — exposed as a method
        so callers don't need to reach into the private _req_* deques.
        """
        self._prune(time.monotonic())
        if self._req_minute:
            return self._req_minute[-1]
        if self._req_day:
            return self._req_day[-1]
        return None

    # ── In-flight reservations ─────────────────────────────

    def add_reservation(self, tokens: int) -> None:
        """Reserve `tokens` against this key's TPM until released or expired."""
        self._reservations.append((time.monotonic(), tokens))

    def clear_oldest_reservation(self) -> None:
        """
        Release the oldest outstanding reservation (FIFO). Called once a
        reserved call completes — success, failure, or rotation away from
        the reserved key all release exactly one reservation.
        """
        if self._reservations:
            self._reservations.popleft()

    def reserved_tokens(self) -> int:
        """
        Sum of in-flight reservations not yet released, pruning any that
        exceeded RESERVATION_TTL (safety net for missed releases).
        """
        now = time.monotonic()
        cutoff = now - RESERVATION_TTL
        while self._reservations and self._reservations[0][0] < cutoff:
            self._reservations.popleft()
        return sum(tk for _, tk in self._reservations)

    def to_dict(self) -> dict:
        return {
            "key_id":                    self.key_id,
            "rpm_current":               self.rpm_current,
            "rpd_current":               self.rpd_current,
            "tpm_current":               self.tpm_current,
            "tpd_current":               self.tpd_current,
            "reserved_tpm":              self.reserved_tokens(),
            "total_requests":            self.total_requests,
            "total_tokens":              self.total_tokens,
            "total_prompt_tokens":       self.total_prompt_tokens,
            "total_completion_tokens":   self.total_completion_tokens,
            "actual_token_records":      self.actual_token_records,
            "estimated_token_records":   self.estimated_token_records,
            "accuracy":                  (
                f"{self.actual_token_records}/{self.total_requests} actual"
                if self.total_requests else "no requests"
            ),
        }


class TokenTracker:
    """
    Single source of truth for all per-key token/request metrics.
    Thread-safe. Used by TokenScheduler, GroqProvider, and Diagnostics.
    """

    def __init__(self) -> None:
        self._lock    = threading.RLock()
        self._metrics: Dict[str, KeyTokenMetrics] = {}

    def record(
        self,
        key_id: str,
        estimated_tokens: int,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """
        Record usage for a key.
        Prefer actual token counts from provider response when available.
        """
        with self._lock:
            if key_id not in self._metrics:
                self._metrics[key_id] = KeyTokenMetrics(key_id=key_id)
            self._metrics[key_id].record(
                tokens=estimated_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            m = self._metrics[key_id]
            logger.debug(
                f"[TokenTracker] {key_id}: "
                f"+{prompt_tokens or estimated_tokens} tokens "
                f"({'actual' if prompt_tokens else 'estimated'}) | "
                f"tpm={m.tpm_current} tpd={m.tpd_current}"
            )

    def get_metrics(self, key_id: str) -> Optional[KeyTokenMetrics]:
        with self._lock:
            return self._metrics.get(key_id)

    def get_tpm(self, key_id: str) -> int:
        with self._lock:
            m = self._metrics.get(key_id)
            return m.tpm_current if m else 0

    def get_reserved_tpm(self, key_id: str) -> int:
        """Sum of in-flight reservations for a key (see reserve())."""
        with self._lock:
            m = self._metrics.get(key_id)
            return m.reserved_tokens() if m else 0

    def get_total_tpm(self, key_id: str) -> int:
        """Actual TPM usage (last 60s) plus any in-flight reservations."""
        with self._lock:
            m = self._metrics.get(key_id)
            if not m:
                return 0
            return m.tpm_current + m.reserved_tokens()

    def reserve(self, key_id: str, tokens: int) -> None:
        """
        Reserve `tokens` against `key_id` immediately, before its LLM call
        starts. Call release_reservation(key_id) exactly once when that
        call completes (success or failure).

        This closes the "thundering herd" race: without it, several
        concurrent calls can all see the same key as idle (because none of
        their usage has been recorded yet) and all pile onto it, only to
        get a TPM 429 once Groq's real usage catches up.
        """
        if tokens <= 0:
            return
        with self._lock:
            if key_id not in self._metrics:
                self._metrics[key_id] = KeyTokenMetrics(key_id=key_id)
            self._metrics[key_id].add_reservation(tokens)
            logger.debug(
                f"[TokenTracker] {key_id}: reserved {tokens} tokens "
                f"(tpm_with_reservations={self.get_total_tpm(key_id)})"
            )

    def release_reservation(self, key_id: str) -> None:
        """Release one reservation previously made via reserve(key_id, ...)."""
        with self._lock:
            m = self._metrics.get(key_id)
            if m:
                m.clear_oldest_reservation()

    def get_remaining_tpm(self, key_id: str, tpm_limit: Optional[int] = None) -> int:
        """
        Remaining TPM headroom for a key, accounting for both actual usage
        and in-flight reservations.
        If `tpm_limit` is not given, uses settings.groq_tpm_limit
        (single source of truth for the provider's per-key TPM cap).
        """
        if tpm_limit is None:
            tpm_limit = _settings.groq_tpm_limit
        return max(0, tpm_limit - self.get_total_tpm(key_id))

    def get_all_metrics(self) -> Dict[str, dict]:
        with self._lock:
            return {kid: m.to_dict() for kid, m in self._metrics.items()}

    def get_summary(self) -> dict:
        with self._lock:
            all_m = list(self._metrics.values())
            return {
                "keys_tracked":   len(all_m),
                "total_requests": sum(m.total_requests for m in all_m),
                "total_tokens":   sum(m.total_tokens   for m in all_m),
                "actual_records": sum(m.actual_token_records   for m in all_m),
                "est_records":    sum(m.estimated_token_records for m in all_m),
                "per_key":        {m.key_id: m.to_dict() for m in all_m},
                "timestamp":      datetime.now(timezone.utc).isoformat(),
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()


# ── Singleton ─────────────────────────────────────────────────────
_token_tracker: Optional[TokenTracker] = None
_tt_lock = threading.Lock()


def get_token_tracker() -> TokenTracker:
    global _token_tracker
    if _token_tracker is None:
        with _tt_lock:
            if _token_tracker is None:
                _token_tracker = TokenTracker()
    return _token_tracker
