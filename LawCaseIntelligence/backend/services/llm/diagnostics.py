"""
backend/services/llm/diagnostics.py
Full LLM infrastructure health report — single endpoint for debugging.

Aggregates data from:
  - APIKeyManager (pool status, cooldowns, daily exhaustion)
  - TokenTracker  (TPM/RPM/TPD/RPD per key — single source of truth)
  - TokenScheduler (current scoring breakdown)
  - rate_limit_classifier (no state, but version info)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, TypedDict

from .api_key_manager import get_api_key_manager
from .token_tracker import get_token_tracker
from .token_scheduler import get_token_scheduler
from .groq_provider import PRIMARY_MODEL

logger = logging.getLogger(__name__)


class KeyStatusDict(TypedDict):
    """Status of a single API key."""
    key_id: str
    masked_key: str
    is_active: bool
    in_cooldown: bool
    cooldown_reason: str
    cooldown_remaining_s: int
    daily_exhausted: bool
    requests_count: int
    success_count: int
    failed_requests: int
    consecutive_failures: int
    success_rate: float
    last_used: str | None
    cooldown_until: str | None


def get_llm_health_report() -> Dict[str, Any]:
    """
    Return a complete snapshot of LLM infrastructure health.

    Structure:
    {
      "timestamp": "...",
      "model": "llama-3.3-70b-versatile",
      "keys": [...],          # from APIKeyManager
      "token_usage": {...},   # from TokenTracker
      "scheduler": [...],     # from TokenScheduler
      "cooldowns": {...},     # derived summary
      "tracker": {...}        # TokenTracker summary
    }
    """
    mgr       = get_api_key_manager()
    tracker   = get_token_tracker()
    scheduler = get_token_scheduler()

    pool_status = mgr.get_pool_status()
    tracker_summary = tracker.get_summary()
    scheduler_status = scheduler.get_status()

    # ── Cooldown summary ───────────────────────────────────────────
    cooldowns = {}
    for k in pool_status["keys"]:
        if k["in_cooldown"] or k["daily_exhausted"]:
            cooldowns[k["key_id"]] = {
                "reason":           k["cooldown_reason"],
                "remaining_s":      k["cooldown_remaining_s"],
                "cooldown_until":   k["cooldown_until"],
                "daily_exhausted":  k["daily_exhausted"],
            }

    earliest = mgr.earliest_recovery_info() if pool_status["all_exhausted"] else None

    report = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "model":         PRIMARY_MODEL,
        "keys":          pool_status["keys"],
        "pool_summary": {
            "total_keys":           pool_status["total_keys"],
            "active_keys":          pool_status["active_keys"],
            "cooldown_keys":        pool_status["cooldown_keys"],
            "daily_exhausted_keys": pool_status["daily_exhausted_keys"],
            "all_exhausted":        pool_status["all_exhausted"],
        },
        "token_usage":   tracker_summary,
        "scheduler":     scheduler_status,
        "cooldowns":     cooldowns,
        "earliest_recovery": earliest,
        "tracker": {
            "keys_tracked":   tracker_summary["keys_tracked"],
            "total_requests": tracker_summary["total_requests"],
            "total_tokens":   tracker_summary["total_tokens"],
            "actual_records": tracker_summary["actual_records"],
            "est_records":    tracker_summary["est_records"],
        },
    }

    logger.info(
        f"[Diagnostics] health report: "
        f"{pool_status['active_keys']}/{pool_status['total_keys']} active, "
        f"{tracker_summary['total_requests']} total requests, "
        f"{tracker_summary['actual_records']} actual / "
        f"{tracker_summary['est_records']} estimated"
    )

    return report


def print_health_report() -> None:
    """Print a human-readable health report to console."""
    report = get_llm_health_report()

    print("\n" + "=" * 60)
    print("  LLM Infrastructure Health Report")
    print("=" * 60)
    print(f"  Model: {report['model']}")
    print(f"  Time:  {report['timestamp']}")
    print()

    print("  Key Pool:")
    for k in report["keys"]:
        status = "🟢 active"
        if k["daily_exhausted"]:
            status = "🔴 daily-exhausted"
        elif k["in_cooldown"]:
            status = f"🟡 cooldown ({k['cooldown_remaining_s']}s, {k['cooldown_reason']})"
        print(f"    {k['key_id']}: {status} | "
              f"req={k['requests_count']} success_rate={k['success_rate']}")

    print()
    print("  Token Scheduler Scores:")
    for s in report["scheduler"]:
        print(f"    {s['key_id']}: remaining_tpm={s['remaining_tpm']}/{s['tpm_limit']} "
              f"success_rate={s['success_rate']}")

    print()
    print(f"  Total requests: {report['tracker']['total_requests']}")
    print(f"  Total tokens:   {report['tracker']['total_tokens']}")
    print(f"  Actual/Est:     {report['tracker']['actual_records']}/{report['tracker']['est_records']}")

    if report["earliest_recovery"]:
        print()
        print(f"  ⚠️  ALL KEYS EXHAUSTED — recovery in {report['earliest_recovery']['wait_human']}")

    print("=" * 60 + "\n")