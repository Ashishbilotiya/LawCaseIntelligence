"""
backend/services/llm/router.py
LLM router — single fixed model (llama-3.3-70b-versatile), API key rotation.

All calls use one model. Resilience comes from rotating API keys,
NOT from switching models.

NOTE: stream_llm() was removed — nothing in the codebase streams tokens
(chat UI uses a plain request/response fetch). See base.py for details.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


def get_llm_provider(
    model_name: Optional[str] = None,   # accepted but ignored — model is fixed
    temperature: Optional[float] = None,
) -> BaseLLMProvider:
    """Return the singleton GroqProvider (fixed model, key-rotation)."""
    from .groq_provider import get_groq_provider
    return get_groq_provider(temperature=temperature)


def invoke_llm(
    messages: List[BaseMessage],
    model_name: Optional[str] = None,   # accepted but ignored
    max_tokens: Optional[int] = None,
) -> str:
    """
    Invoke LLM with automatic API key rotation.
    Model is always llama-3.3-70b-versatile.
    Key rotation is handled inside GroqProvider.

    Args:
        max_tokens: Optional per-call override of the completion token
                     reservation. Pass a small value (e.g. for UCE's
                     lightweight tagging calls) so the request doesn't
                     reserve far more tokens than it needs against the
                     provider's TPM budget.
    """
    provider = get_llm_provider()
    return provider.invoke(messages, max_tokens=max_tokens)


def quick_invoke(system_prompt: str, user_prompt: str) -> str:
    """Convenience wrapper: build messages and invoke."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    return invoke_llm(messages)
