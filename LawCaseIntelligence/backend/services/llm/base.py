"""
backend/services/llm/base.py
Abstract base class for all LLM providers.
Groq-only implementation.

Streaming (stream() / invoke_streaming_joined()) and invoke_with_fallback()
were removed — the chat UI uses a plain request/response fetch (no
SSE/websocket token streaming), and no agent, UCE, or chat-router code path
used either. If needed again later, re-introduce as optional (non-abstract)
methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import BaseMessage


class BaseLLMProvider(ABC):
    """Abstract base for Groq LLM provider."""

    provider_name: str = "base"

    @abstractmethod
    def invoke(self, messages: List[BaseMessage], **kwargs) -> str:
        """Send messages and return full response string."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return active model name string."""
        ...

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Return context window in tokens."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
