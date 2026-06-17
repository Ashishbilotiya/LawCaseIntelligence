"""
backend/services/llm/api_key_manager.py
API Key Pool Manager — structured rate-limit classification + persistent state.

Changes from previous version:
  - Removed _DAY_KEYWORDS / _MINUTE_KEYWORDS keyword matching entirely.
  - All error classification now goes through RateLimitClassifier.
  - Cooldown duration prefers retry_after from the actual error.
  - Daily-exhaustion state persists to data/system/key_state.json
    so it survives app restarts.
  - get_next_key() remains for backward compat (round-robin fallback),
    but TokenScheduler.get_best_key() is the preferred selection path
    for the agent pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from .rate_limit_classifier import RateLimitClassifier, RateLimitInfo
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# ── Cooldown durations (used only when retry_after is unavailable) ─
# Centralized in backend.config.settings (key_cooldown_seconds,
# key_max_consecutive_failures) so these stay in sync with the rest
# of the LLM infrastructure.
COOLDOWN_TPM_RPM   = _settings.key_cooldown_seconds  # past the 1-min TPM/RPM reset window
COOLDOWN_DEFAULT   = _settings.key_cooldown_seconds
MAX_CONSECUTIVE_FAILURES = _settings.key_max_consecutive_failures

# ── Persistent state file ──────────────────────────────────────────
# Path centralized in settings.key_state_file so APIKeyManager and
# Settings.ensure_dirs() agree on where persisted key state lives.
_STATE_FILE = Path(_settings.key_state_file)
_STATE_DIR  = _STATE_FILE.parent

# Minimum cooldown when computing "until midnight UTC" (settings.key_daily_cooldown_min_seconds)
_MIN_DAILY_COOLDOWN_SECONDS = _settings.key_daily_cooldown_min_seconds


def _seconds_until_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(_MIN_DAILY_COOLDOWN_SECONDS, int((midnight - now).total_seconds()))


def _cooldown_for(info: RateLimitInfo) -> int:
    """
    Determine cooldown duration in seconds.
      1. If retry_after provided by Groq → use it directly.
      2. Else: TPD/RPD → until midnight UTC; TPM/RPM → 80s.
    """
    if info.retry_after is not None:
        logger.info(f"[APIKeyManager] Using provider retry_after={info.retry_after}s")
        return max(1, info.retry_after)

    if info.is_daily:
        secs = _seconds_until_midnight_utc()
        logger.info(f"[APIKeyManager] {info.limit_type} (daily) → cooldown={secs}s (midnight UTC)")
        return secs

    logger.info(f"[APIKeyManager] {info.limit_type} (minute) → cooldown={COOLDOWN_TPM_RPM}s")
    return COOLDOWN_TPM_RPM


@dataclass
class KeyHealth:
    """Per-key health tracking."""
    key_id:  str
    api_key: str

    requests_count:       int = 0
    success_count:        int = 0
    failed_requests:      int = 0
    consecutive_failures: int = 0

    last_used:       Optional[datetime] = None
    last_failure:    Optional[datetime] = None
    cooldown_until:  Optional[datetime] = None
    cooldown_reason: str = ""

    daily_exhausted: bool = False
    daily_exhausted_date: str = ""   # ISO date string — auto-clears on new day

    is_active: bool = True

    @property
    def masked_key(self) -> str:
        return f"...{self.api_key[-6:]}" if len(self.api_key) > 6 else "***"

    @property
    def success_rate(self) -> float:
        return self.success_count / self.requests_count if self.requests_count else 1.0

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_until is not None and datetime.now(timezone.utc) < self.cooldown_until

    def mark_success(self) -> None:
        self.requests_count      += 1
        self.success_count       += 1
        self.consecutive_failures = 0
        self.last_used = datetime.now(timezone.utc)

    def _record_failure_counts(self) -> None:
        """
        Shared bookkeeping for any failure that counts toward the
        consecutive-failure cooldown threshold. Used by both the
        rate-limit path (mark_failure) and the transient-error path
        in APIKeyManager.report_failure().
        """
        now = datetime.now(timezone.utc)
        self.requests_count       += 1
        self.failed_requests      += 1
        self.consecutive_failures += 1
        self.last_failure = now
        self.last_used    = now

    def mark_failure(self, info: RateLimitInfo, cooldown_secs: int) -> None:
        """
        Apply a structured rate-limit failure: record counts, mark
        daily-exhaustion if applicable, and always disable the key
        for `cooldown_secs` (every caller of this method is on the
        confirmed-rate-limit path, so cooldown is never optional here).
        """
        self._record_failure_counts()

        if info.is_daily:
            self.daily_exhausted      = True
            self.daily_exhausted_date = datetime.now(timezone.utc).date().isoformat()

        self.disable(cooldown_secs=cooldown_secs, reason=info.limit_type)

    def disable(self, cooldown_secs: int, reason: str) -> None:
        self.is_active       = False
        self.cooldown_reason = reason
        self.cooldown_until  = datetime.fromtimestamp(time.time() + cooldown_secs, tz=timezone.utc)
        logger.warning(
            f"[APIKeyManager] {self.key_id} ({self.masked_key}) disabled. "
            f"reason={reason} cooldown={cooldown_secs}s "
            f"until={self.cooldown_until.strftime('%H:%M:%S UTC')}"
        )

    def try_recover(self) -> bool:
        # Auto-clear stale daily_exhausted flag if date has changed
        if self.daily_exhausted:
            today = datetime.now(timezone.utc).date().isoformat()
            if self.daily_exhausted_date and self.daily_exhausted_date != today:
                self.daily_exhausted      = False
                self.daily_exhausted_date = ""
                logger.info(f"[APIKeyManager] {self.key_id} daily_exhausted cleared (new day)")

        if not self.is_active and not self.in_cooldown:
            self.is_active            = True
            self.consecutive_failures = 0
            self.cooldown_until       = None
            prev_reason = self.cooldown_reason
            self.cooldown_reason      = ""
            logger.info(f"[APIKeyManager] {self.key_id} ({self.masked_key}) recovered from {prev_reason}.")
            return True
        return False

    def to_dict(self) -> dict:
        remaining = 0
        if self.in_cooldown and self.cooldown_until:
            remaining = int((self.cooldown_until - datetime.now(timezone.utc)).total_seconds())
        return {
            "key_id":               self.key_id,
            "masked_key":           self.masked_key,
            "is_active":            self.is_active,
            "in_cooldown":          self.in_cooldown,
            "cooldown_reason":      self.cooldown_reason,
            "cooldown_remaining_s": remaining,
            "daily_exhausted":      self.daily_exhausted,
            "requests_count":       self.requests_count,
            "success_count":        self.success_count,
            "failed_requests":      self.failed_requests,
            "consecutive_failures": self.consecutive_failures,
            "success_rate":         round(self.success_rate, 3),
            "last_used":            self.last_used.isoformat() if self.last_used else None,
            "cooldown_until":       self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    # ── Persistence helpers ────────────────────────────────────────
    def to_persist_dict(self) -> dict:
        return {
            "daily_exhausted":      self.daily_exhausted,
            "daily_exhausted_date": self.daily_exhausted_date,
            "cooldown_until":       self.cooldown_until.isoformat() if self.cooldown_until else None,
            "cooldown_reason":      self.cooldown_reason,
            "is_active":            self.is_active,
        }

    def load_persist_dict(self, data: dict) -> None:
        self.daily_exhausted      = data.get("daily_exhausted", False)
        self.daily_exhausted_date = data.get("daily_exhausted_date", "")
        cu = data.get("cooldown_until")
        if cu:
            try:
                self.cooldown_until = datetime.fromisoformat(cu)
            except ValueError:
                self.cooldown_until = None
        self.cooldown_reason = data.get("cooldown_reason", "")
        # is_active is re-derived via try_recover() on load


class APIKeyManager:
    """Thread-safe API key pool with structured rate-limit classification."""

    def __init__(self, keys: List[str]) -> None:
        if not keys:
            raise EnvironmentError(
                "No Groq API keys found. Set GROQ_API_KEY_1 … GROQ_API_KEY_4 in .env"
            )
        self._lock = threading.RLock()
        self._pool: List[KeyHealth] = [
            KeyHealth(key_id=f"key_{i+1}", api_key=k) for i, k in enumerate(keys)
        ]
        self._rr_index = 0

        # Check env var to force fresh state (useful after adding new keys or rate-limit reset)
        force_fresh = os.getenv("RESET_API_KEY_STATE", "false").lower() == "true"
        if force_fresh:
            logger.info("[APIKeyManager] RESET_API_KEY_STATE=true — ignoring persisted state")
            try:
                if _STATE_FILE.exists():
                    _STATE_FILE.unlink()
                    logger.info("[APIKeyManager] Deleted stale state file")
            except Exception as e:
                logger.warning(f"[APIKeyManager] Could not delete state file: {e}")
        else:
            self.load_state()
        self._try_recover_all()

        logger.info(
            f"[APIKeyManager] Initialised — {len(self._pool)} key(s): "
            + ", ".join(kh.key_id for kh in self._pool)
        )

    def force_reset_all_keys(self) -> None:
        """Manually reset all keys to active state (admin endpoint)."""
        with self._lock:
            for kh in self._pool:
                kh.is_active            = True
                kh.daily_exhausted      = False
                kh.daily_exhausted_date = ""
                kh.cooldown_until       = None
                kh.cooldown_reason      = ""
                kh.consecutive_failures = 0
            self.save_state()
            logger.info(f"[APIKeyManager] All {len(self._pool)} keys force-reset to active")

    # ── Public selection API ────────────────────────────────────────

    def get_next_key(self) -> KeyHealth:
        """Round-robin fallback selection (backward compat)."""
        with self._lock:
            self._try_recover_all()
            total = len(self._pool)
            for _ in range(total):
                idx            = self._rr_index % total
                self._rr_index = (idx + 1) % total
                kh             = self._pool[idx]
                if kh.is_active and not kh.in_cooldown and not kh.daily_exhausted:
                    logger.debug(f"[APIKeyManager] Assigned {kh.key_id} ({kh.masked_key}) [round-robin]")
                    return kh
            raise RuntimeError(
                f"[APIKeyManager] All {total} keys on cooldown. "
                f"Earliest recovery: {self._earliest_recovery_time()}"
            )

    def get_fallback_keys(self, exclude_key_id: str) -> List[KeyHealth]:
        with self._lock:
            self._try_recover_all()
            candidates = [
                kh for kh in self._pool
                if kh.key_id != exclude_key_id and kh.is_active
                and not kh.in_cooldown and not kh.daily_exhausted
            ]
            candidates.sort(key=lambda kh: kh.last_used or datetime.min.replace(tzinfo=timezone.utc))
            return candidates

    def get_pool(self) -> List[KeyHealth]:
        """Expose pool for TokenScheduler scoring (read-only usage)."""
        with self._lock:
            self._try_recover_all()
            return list(self._pool)

    # ── Reporting API ────────────────────────────────────────────────

    def report_success(self, key_id: str) -> None:
        with self._lock:
            kh = self._find(key_id)
            if kh:
                kh.mark_success()

    def report_failure(self, key_id: str, error: Exception) -> bool:
        """
        Classify error via RateLimitClassifier and apply cooldown.
        Returns True if the key was put on cooldown (rate limit).
        """
        with self._lock:
            kh = self._find(key_id)
            if not kh:
                return False

            if not RateLimitClassifier.is_rate_limit(error):
                if RateLimitClassifier.is_transient(error):
                    kh._record_failure_counts()
                    logger.warning(
                        f"[APIKeyManager] {key_id} transient error "
                        f"(consecutive={kh.consecutive_failures})"
                    )
                    if kh.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        kh.disable(cooldown_secs=COOLDOWN_DEFAULT, reason="transient")
                        return True
                    return False
                # Hard error (auth, bad request)
                kh.requests_count  += 1
                kh.failed_requests += 1
                logger.error(f"[APIKeyManager] {key_id} hard error: {error}")
                return False

            # ── Structured classification (no keyword matching) ──────
            info = RateLimitClassifier.classify(error)
            cooldown_secs = _cooldown_for(info)
            kh.mark_failure(info=info, cooldown_secs=cooldown_secs)

            logger.warning(
                f"[APIKeyManager] {key_id} {info.limit_type} "
                f"(limit={info.limit} used={info.used} requested={info.requested}) "
                f"→ cooldown {cooldown_secs}s"
            )

            self.save_state()
            return True

    # ── Status / diagnostics ─────────────────────────────────────────

    def all_keys_exhausted(self) -> bool:
        with self._lock:
            self._try_recover_all()
            return not any(
                kh.is_active and not kh.in_cooldown and not kh.daily_exhausted
                for kh in self._pool
            )

    def earliest_recovery_info(self) -> dict:
        with self._lock:
            times = [(kh.cooldown_until, kh.key_id) for kh in self._pool if kh.cooldown_until is not None]
            if not times:
                return {"recovery_time": "unknown", "wait_seconds": 0, "wait_human": "unknown", "key_id": ""}
            earliest_dt, key_id = min(times, key=lambda x: x[0])
            now = datetime.now(timezone.utc)
            wait_secs = max(0, int((earliest_dt - now).total_seconds()))
            mins, secs = divmod(wait_secs, 60)
            hours, mins = divmod(mins, 60)
            wait_str = f"{hours}h {mins}m" if hours else (f"{mins}m {secs}s" if mins else f"{secs}s")
            return {
                "recovery_time": earliest_dt.strftime("%H:%M:%S UTC"),
                "wait_seconds":  wait_secs,
                "wait_human":    wait_str,
                "key_id":        key_id,
            }

    def get_pool_status(self) -> dict:
        with self._lock:
            self._try_recover_all()
            return {
                "total_keys":    len(self._pool),
                "active_keys":   sum(1 for kh in self._pool if kh.is_active and not kh.in_cooldown and not kh.daily_exhausted),
                "cooldown_keys": sum(1 for kh in self._pool if kh.in_cooldown),
                "daily_exhausted_keys": sum(1 for kh in self._pool if kh.daily_exhausted),
                "all_exhausted": self.all_keys_exhausted(),
                "keys":          [kh.to_dict() for kh in self._pool],
            }

    # ── Persistence ────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist daily-exhaustion + cooldown state to data/system/key_state.json."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            state = {kh.key_id: kh.to_persist_dict() for kh in self._pool}
            tmp = _STATE_FILE.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            tmp.replace(_STATE_FILE)
            logger.debug(f"[APIKeyManager] State persisted to {_STATE_FILE}")
        except Exception as e:
            logger.warning(f"[APIKeyManager] save_state failed: {e}")

    def load_state(self) -> None:
        """Load persisted state on startup."""
        if not _STATE_FILE.exists():
            return
        try:
            with open(_STATE_FILE) as f:
                state = json.load(f)
            for kh in self._pool:
                if kh.key_id in state:
                    kh.load_persist_dict(state[kh.key_id])
            logger.info(f"[APIKeyManager] State loaded from {_STATE_FILE}")
        except Exception as e:
            logger.warning(f"[APIKeyManager] load_state failed: {e}")

    # ── Internal ──────────────────────────────────────────────────

    def _find(self, key_id: str) -> Optional[KeyHealth]:
        return next((kh for kh in self._pool if kh.key_id == key_id), None)

    def _try_recover_all(self) -> None:
        recovered_any = False
        for kh in self._pool:
            if kh.try_recover():
                recovered_any = True
        if recovered_any:
            self.save_state()

    def _earliest_recovery_time(self) -> str:
        times = [kh.cooldown_until for kh in self._pool if kh.cooldown_until]
        return min(times).strftime("%H:%M:%S UTC") if times else "unknown"


# ── Singleton ─────────────────────────────────────────────────────

_manager_instance: Optional[APIKeyManager] = None
_manager_lock = threading.Lock()


def _load_keys_from_env() -> List[str]:
    keys, seen = [], set()
    for i in range(1, 11):
        val = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if val and val not in seen:
            keys.append(val); seen.add(val)
    if not keys:
        val = os.environ.get("GROQ_API_KEY", "").strip()
        if val:
            keys.append(val)
    return keys


def get_api_key_manager() -> APIKeyManager:
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = APIKeyManager(_load_keys_from_env())
    return _manager_instance


def reset_api_key_manager() -> None:
    global _manager_instance
    with _manager_lock:
        _manager_instance = None
