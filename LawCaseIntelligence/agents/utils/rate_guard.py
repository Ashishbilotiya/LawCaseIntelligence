"""
agents/utils/rate_guard.py
Centralized rate-limit protection for all agent chunk loops.

Problems solved:
  - 17 chunks fired simultaneously exhaust TPM across all keys instantly
  - Keys marked exhausted before they actually are (no delay between calls)

Solution:
  - INTER_CHUNK_DELAY: mandatory pause between every chunk call
  - wait_for_available_key(): sleep until earliest key recovers
  - safe_invoke(): drop-in replacement for invoke_llm in agent loops
"""
from __future__ import annotations

import logging
import time

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# ── Tunable constants (centralized in backend.config.settings) ────
INTER_CHUNK_DELAY      = _settings.inter_chunk_delay_seconds
                                 # e.g. 17 chunks × 6s = ~102s — well within TPM windows
COOLDOWN_POLL_INTERVAL = _settings.cooldown_poll_interval_seconds  # poll interval while waiting
MAX_COOLDOWN_WAIT      = _settings.max_cooldown_wait_seconds       # never wait longer than this
FAILURE_BACKOFF        = _settings.chunk_failure_backoff_seconds  # extra delay after a chunk failure


def inter_chunk_delay(chunk_idx: int, failed: bool = False) -> None:
    """Mandatory pause between chunk LLM calls to avoid TPM bursts."""
    delay = INTER_CHUNK_DELAY + (FAILURE_BACKOFF if failed else 0)
    logger.debug(f"[RateGuard] Chunk {chunk_idx} → sleeping {delay:.1f}s")
    time.sleep(delay)


def wait_for_available_key(max_wait: float = MAX_COOLDOWN_WAIT) -> bool:
    """
    Sleep in small intervals until at least one key exits cooldown.
    Returns True if a key became available, False if max_wait exceeded.
    """
    from backend.services.llm.api_key_manager import get_api_key_manager
    mgr      = get_api_key_manager()
    waited   = 0.0
    last_log = 0.0

    while waited < max_wait:
        status = mgr.get_pool_status()
        if status["active_keys"] > 0:
            logger.info(f"[RateGuard] Key available after {waited:.0f}s wait")
            return True

        if waited - last_log >= 15:
            all_keys = status.get("keys", [])
            times    = [k.get("cooldown_until") for k in all_keys if k.get("cooldown_until")]
            earliest = min(times) if times else "unknown"
            logger.info(
                f"[RateGuard] All keys on cooldown. "
                f"Earliest recovery: {earliest}. "
                f"Waited {waited:.0f}s / max {max_wait:.0f}s"
            )
            last_log = waited

        time.sleep(COOLDOWN_POLL_INTERVAL)
        waited += COOLDOWN_POLL_INTERVAL

    logger.warning(f"[RateGuard] Max wait {max_wait}s exceeded — proceeding")
    return False


def safe_invoke(messages: list, chunk_idx: int, agent_name: str) -> str | None:
    """
    Drop-in replacement for invoke_llm inside agent chunk loops.

    1. If all keys are on cooldown → waits until one recovers
    2. Calls invoke_llm
    3. Sleeps inter_chunk_delay after every call
    4. Returns None on failure so caller can skip the chunk gracefully

    Args:
        messages:   LangChain message list
        chunk_idx:  1-based chunk index (for logging)
        agent_name: e.g. "IssueAgent"
    """
    from backend.services.llm.router import invoke_llm
    from backend.services.llm.api_key_manager import get_api_key_manager

    # Wait for a key if all are on cooldown
    mgr    = get_api_key_manager()
    status = mgr.get_pool_status()
    if status["active_keys"] == 0:
        logger.info(
            f"[RateGuard] [{agent_name}] Chunk {chunk_idx}: "
            f"all keys on cooldown — waiting for recovery"
        )
        available = wait_for_available_key()
        if not available:
            logger.warning(
                f"[RateGuard] [{agent_name}] Chunk {chunk_idx}: "
                f"skipping — keys still exhausted after max wait"
            )
            inter_chunk_delay(chunk_idx, failed=True)
            return None

    try:
        result = invoke_llm(messages)
        inter_chunk_delay(chunk_idx, failed=False)
        return result
    except Exception as e:
        logger.warning(f"[RateGuard] [{agent_name}] Chunk {chunk_idx} failed: {e}")
        inter_chunk_delay(chunk_idx, failed=True)
        return None
