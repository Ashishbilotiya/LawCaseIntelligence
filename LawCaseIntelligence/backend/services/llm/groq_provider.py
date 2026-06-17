"""
backend/services/llm/groq_provider.py
Groq LLM provider — fixed model, structured rate-limit handling, actual token tracking.

Changes from previous version:
  - Extracts response.usage_metadata (prompt_tokens, completion_tokens) when available
  - Falls back to estimation only when usage metadata is missing
  - Persists key state via APIKeyManager.save_state() after rate-limit events
  - Detailed [GroqProvider] logs: selected key, actual usage, classification

Backward compatible:
  - invoke() signature unchanged
  - get_groq_provider() singleton unchanged
  - stream()/invoke_streaming_joined() removed (unused — see base.py)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from .base import BaseLLMProvider
from .api_key_manager import APIKeyManager, KeyHealth, get_api_key_manager
from .token_tracker import get_token_tracker
from .token_scheduler import get_token_scheduler
from .rate_limit_classifier import RateLimitClassifier
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# Centralized in backend.config.settings — single source of truth so
# diagnostics, the scheduler, and the provider never disagree.
PRIMARY_MODEL  = _settings.groq_primary_model
CONTEXT_WINDOW = _settings.groq_context_window


def _estimate_tokens(messages: List[BaseMessage], max_response_tokens: int) -> int:
    """Fallback estimate when actual usage metadata is unavailable.

    Improved estimation for legal text: uses 3.8 chars/token (better than 4)
    for more accurate TPM prediction.
    """
    input_chars = sum(len(str(m.content)) for m in messages)
    # Legal text often has slightly longer words - 3.8 chars/token is more accurate
    estimated_input_tokens = max(1, int(input_chars / 3.8))
    return estimated_input_tokens + max_response_tokens


def _extract_usage(response) -> tuple[Optional[int], Optional[int]]:
    """
    Extract (prompt_tokens, completion_tokens) from a LangChain ChatGroq response.
    Returns (None, None) if usage metadata is unavailable.
    """
    # LangChain AIMessage exposes usage_metadata: {"input_tokens", "output_tokens", "total_tokens"}
    usage = getattr(response, "usage_metadata", None)
    if usage:
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        if prompt is not None and completion is not None:
            return int(prompt), int(completion)

    # Fallback: response_metadata.token_usage (older format)
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}
    prompt     = token_usage.get("prompt_tokens")
    completion = token_usage.get("completion_tokens")
    if prompt is not None and completion is not None:
        return int(prompt), int(completion)

    return None, None


def _make_chat_groq(api_key: str, temperature: float, max_tokens: int) -> ChatGroq:
    """
    Build a ChatGroq client for one call.

    streaming=True is intentional even though invoke() is non-streaming —
    LangChain still returns a single aggregated AIMessage from .invoke(),
    but Groq's streaming response path is what populates usage_metadata
    reliably (see _extract_usage). max_retries=0 because GroqProvider
    handles retries/backoff itself (see invoke()).
    """
    return ChatGroq(
        model=PRIMARY_MODEL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        max_retries=0,
    )


class GroqProvider(BaseLLMProvider):
    """
    Groq provider — fixed model, structured rate-limit handling,
    actual token usage tracking via TokenTracker (single source of truth).
    """

    provider_name = "groq"

    def __init__(
        self,
        model_name: Optional[str] = None,    # accepted but ignored — model is fixed (see router.py)
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._temperature = temperature if temperature is not None else _settings.groq_temperature
        self._max_tokens  = max_tokens if max_tokens is not None else _settings.groq_max_tokens
        self._key_manager: APIKeyManager = get_api_key_manager()
        self._token_tracker = get_token_tracker()

        _ = self._key_manager.get_next_key()
        logger.info(
            f"[GroqProvider] Ready — model={PRIMARY_MODEL}, "
            f"temp={self._temperature}, max_tokens={self._max_tokens}, "
            f"keys={self._key_manager.get_pool_status()['total_keys']}"
        )

    @property
    def model_name(self) -> str:
        return PRIMARY_MODEL

    @property
    def context_window(self) -> int:
        return CONTEXT_WINDOW

    def _record_usage(self, key_id: str, response, messages: List[BaseMessage], max_tokens: int) -> None:
        """Record actual usage if available, else estimate using this call's max_tokens."""
        prompt_tok, completion_tok = _extract_usage(response)
        estimated = _estimate_tokens(messages, max_response_tokens=max_tokens)

        if prompt_tok is not None:
            logger.info(
                f"[GroqProvider] {key_id} actual usage: "
                f"prompt={prompt_tok} completion={completion_tok} "
                f"total={prompt_tok + completion_tok}"
            )
            self._token_tracker.record(
                key_id, estimated,
                prompt_tokens=prompt_tok, completion_tokens=completion_tok,
            )
        else:
            logger.debug(f"[GroqProvider] {key_id} no usage metadata — estimating {estimated} tokens")
            self._token_tracker.record(key_id, estimated)

    # ── Key selection (with scheduler) ────────────────────────

    def _select_keys(self, reserve_tokens: Optional[int] = None) -> List[KeyHealth]:
        """
        Build ordered list of keys to try:
          1. Best key per TokenScheduler (score-based)
          2. Fallback keys (round-robin order)
        """
        try:
            best = get_token_scheduler().get_best_key(reserve_tokens=reserve_tokens)
        except Exception as e:
            logger.debug(f"[GroqProvider] Scheduler unavailable, using round-robin: {e}")
            best = None
        except Exception as e:
            logger.debug(f"[GroqProvider] Scheduler unavailable, using round-robin: {e}")
            best = None

        if best is not None:
            fallbacks = self._key_manager.get_fallback_keys(exclude_key_id=best.key_id)
            return [best] + fallbacks

        # Scheduler returned None — try round-robin
        try:
            start = self._key_manager.get_next_key()
        except RuntimeError as e:
            raise RuntimeError(str(e)) from e
        return [start] + self._key_manager.get_fallback_keys(exclude_key_id=start.key_id)

    # ── Invoke (non-streaming) ─────────────────────────────────────

    def invoke(self, messages: List[BaseMessage], max_tokens: Optional[int] = None, **kwargs) -> str:
        last_err: Exception = RuntimeError("No keys available")

        # Per-call override (e.g. UCE requests a much smaller reservation
        # than the default so it doesn't trip Groq's TPM pre-flight check).
        effective_max_tokens = max_tokens if max_tokens is not None else self._max_tokens

        keys_to_try = self._select_keys(reserve_tokens=effective_max_tokens)
        best_key = keys_to_try[0] if keys_to_try else None

        try:
            for key_health in keys_to_try:
                logger.info(
                    f"[GroqProvider] Trying {key_health.key_id} ({key_health.masked_key}) "
                    f"max_tokens={effective_max_tokens}"
                )

                for attempt in range(_settings.max_retries):
                    try:
                        llm = _make_chat_groq(
                            api_key=key_health.api_key,
                            temperature=self._temperature,
                            max_tokens=effective_max_tokens,
                        )
                        response = llm.invoke(messages)
                        text = response.content.strip()

                        self._key_manager.report_success(key_health.key_id)
                        self._record_usage(key_health.key_id, response, messages, effective_max_tokens)
                        return text

                    except Exception as e:
                        last_err = e
                        was_rate_limited = self._key_manager.report_failure(key_health.key_id, e)

                        if was_rate_limited:
                            logger.warning(
                                f"[GroqProvider] {key_health.key_id} rate-limited, rotating to next key"
                            )
                            break

                        if RateLimitClassifier.is_transient(e) and attempt < _settings.max_retries - 1:
                            wait = _settings.backoff_base ** (attempt + 1) * _settings.retry_backoff_multiplier_seconds
                            logger.warning(
                                f"[GroqProvider] {key_health.key_id} transient error "
                                f"(attempt {attempt+1}), retrying in {wait:.0f}s"
                            )
                            time.sleep(wait)
                            continue

                        logger.error(f"[GroqProvider] {key_health.key_id} failed (attempt {attempt+1}): {e}")
                        break

            raise RuntimeError(f"[GroqProvider] All keys exhausted. Last error: {last_err}")
        finally:
            if best_key:
                self._token_tracker.release_reservation(best_key.key_id)


# ── Singleton ─────────────────────────────────────────────────────

_provider_instance: Optional[GroqProvider] = None
_provider_lock = threading.Lock()


def get_groq_provider(
    model_name: str | None = None,    # accepted but ignored — model is fixed (see router.py)
    temperature: float | None = None,
) -> GroqProvider:
    global _provider_instance
    if _provider_instance is None:
        with _provider_lock:
            if _provider_instance is None:
                _provider_instance = GroqProvider(temperature=temperature)
    return _provider_instance