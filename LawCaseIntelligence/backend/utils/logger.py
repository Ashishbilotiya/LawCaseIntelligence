"""
backend/utils/logger.py
Module-level logger factory and request ID context for tracing pipeline runs.
"""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Optional

_request_id_local = threading.local()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the LawCaseIntelligence namespace."""
    return logging.getLogger(f"LawCaseIntelligence.{name}")


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set a request ID for the current thread (pipeline run tracing)."""
    rid = request_id or str(uuid.uuid4())[:8]
    _request_id_local.value = rid
    return rid


def get_request_id() -> str:
    """Get the current thread's request ID."""
    return getattr(_request_id_local, "value", "no-rid")


class PipelineLogger:
    """
    Structured logger that prefixes all messages with [pipeline_id][agent_name].
    Use inside agent nodes for consistent log output.
    """

    def __init__(self, agent_name: str, pipeline_id: str = "") -> None:
        self._logger    = get_logger(agent_name)
        self._agent     = agent_name
        self._pipeline  = pipeline_id or get_request_id()

    def _prefix(self) -> str:
        return f"[{self._pipeline}][{self._agent}]"

    def info(self, msg: str) -> None:
        self._logger.info(f"{self._prefix()} {msg}")

    def warning(self, msg: str) -> None:
        self._logger.warning(f"{self._prefix()} {msg}")

    def error(self, msg: str) -> None:
        self._logger.error(f"{self._prefix()} {msg}")

    def debug(self, msg: str) -> None:
        self._logger.debug(f"{self._prefix()} {msg}")
