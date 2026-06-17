"""
backend/services/llm/rate_limit_classifier.py
Structured Groq error classification — NO keyword matching.

Parses actual error structure to extract:
  - limit_type: TPM / RPM / TPD / RPD / UNKNOWN
  - limit:      the numeric limit (e.g. 12000)
  - used:       tokens/requests already used
  - requested:  tokens/requests in the failed request
  - retry_after: seconds to wait (from error or header)

Why this eliminates false TPD detections:
  Old approach:  "daily" in error_str  → matched any error mentioning "daily"
  New approach:  Parse JSON error body → check actual limit type field from Groq
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()


@dataclass
class RateLimitInfo:
    limit_type:  str            # TPM | RPM | TPD | RPD | UNKNOWN
    limit:       int   = 0     # the actual numeric limit
    used:        int   = 0     # how many tokens/requests were used
    requested:   int   = 0     # how many were requested
    retry_after: Optional[int] = None   # seconds to wait (None = use default)
    raw_message: str  = ""

    @property
    def is_daily(self) -> bool:
        return self.limit_type in ("TPD", "RPD")

    @property
    def is_minute(self) -> bool:
        return self.limit_type in ("TPM", "RPM")


class RateLimitClassifier:
    """
    Classifies Groq rate limit errors by parsing the actual error structure.

    Groq error format (from API):
    {
      "error": {
        "message": "Rate limit reached for model ... on tokens per minute (TPM): \
                    Limit 12000, Used 5567, Requested 8901. \
                    Please try again in 19.37s.",
        "type": "tokens",
        "code": "rate_limit_exceeded"
      }
    }

    We parse:
      1. "tokens per minute" / "requests per minute" / etc. → limit_type
      2. "Limit N" → limit
      3. "Used N" → used
      4. "Requested N" → requested
      5. "try again in Xs" → retry_after
    """

    # ── Compiled patterns (fast, no string iteration) ─────────────
    _LIMIT_VALUE_RE = re.compile(r"limit\s+(\d+)", re.IGNORECASE)
    _USED_RE        = re.compile(r"used\s+(\d+)", re.IGNORECASE)
    _REQUESTED_RE   = re.compile(r"requested\s+(\d+)", re.IGNORECASE)
    _RETRY_RE       = re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE)
    _RETRY_HEADER   = re.compile(r"retry-after[:\s]+(\d+)", re.IGNORECASE)

    # Type detection — look for explicit parenthetical type label from Groq
    _TPM_RE = re.compile(r"tokens per minute\s*\(TPM\)", re.IGNORECASE)
    _RPM_RE = re.compile(r"requests per minute\s*\(RPM\)", re.IGNORECASE)
    _TPD_RE = re.compile(r"tokens per day\s*\(TPD\)", re.IGNORECASE)
    _RPD_RE = re.compile(r"requests per day\s*\(RPD\)", re.IGNORECASE)

    @classmethod
    def classify(cls, error: Exception) -> RateLimitInfo:
        """
        Classify a Groq exception into a structured RateLimitInfo.
        Returns RateLimitInfo(limit_type="UNKNOWN") if not a rate limit error.
        """
        err_str = str(error)

        # ── Try to extract JSON body first ────────────────────────
        message = cls._extract_message(err_str)
        if not message:
            message = err_str

        # ── Determine limit type from explicit Groq labels ────────
        limit_type = cls._classify_type(message)
        if limit_type == "UNKNOWN" and "429" not in err_str and "rate_limit" not in err_str.lower():
            # Not a rate limit error at all
            return RateLimitInfo(limit_type="UNKNOWN", raw_message=message[:300])

        # ── Extract numeric values ────────────────────────────────
        limit     = cls._extract_int(cls._LIMIT_VALUE_RE, message)
        used      = cls._extract_int(cls._USED_RE,        message)
        requested = cls._extract_int(cls._REQUESTED_RE,   message)

        # ── Extract retry_after ───────────────────────────────────
        retry_after = cls._extract_retry_after(message, err_str)

        result = RateLimitInfo(
            limit_type=limit_type,
            limit=limit,
            used=used,
            requested=requested,
            retry_after=retry_after,
            raw_message=message[:300],
        )

        logger.info(
            f"[RateLimitClassifier] type={result.limit_type} "
            f"limit={result.limit} used={result.used} "
            f"requested={result.requested} "
            f"retry_after={result.retry_after if result.retry_after is not None else 'N/A'}"
            f"{'s' if result.retry_after is not None else ''}"
        )
        return result

    @classmethod
    def is_rate_limit(cls, error: Exception) -> bool:
        """Quick check: is this a rate limit error at all?"""
        s = str(error)
        return "429" in s or "rate_limit_exceeded" in s or "rate limit" in s.lower()

    @classmethod
    def is_transient(cls, error: Exception) -> bool:
        """Is this a transient (non-rate-limit) error worth retrying?"""
        s = str(error).lower()
        return any(k in s for k in ("timeout", "timed out", "connection", "503", "502", "overloaded"))

    # ── Private helpers ───────────────────────────────────────────

    @classmethod
    def _extract_message(cls, err_str: str) -> str:
        """Try to extract the 'message' field from JSON error body."""
        # Groq errors often look like: "Error code: 429 - {'error': {'message': '...'}}"
        try:
            # Find first { ... } in the string
            start = err_str.find("{")
            if start == -1:
                return err_str
            body = json.loads(err_str[start:])
            # Navigate nested: body["error"]["message"]
            if isinstance(body, dict):
                error_obj = body.get("error", body)
                if isinstance(error_obj, dict):
                    return str(error_obj.get("message", err_str))
        except (json.JSONDecodeError, ValueError):
            pass
        return err_str

    @classmethod
    def _classify_type(cls, message: str) -> str:
        """Determine limit type from Groq's explicit parenthetical label."""
        if cls._TPD_RE.search(message):
            return "TPD"
        if cls._RPD_RE.search(message):
            return "RPD"
        if cls._TPM_RE.search(message):
            return "TPM"
        if cls._RPM_RE.search(message):
            return "RPM"
        # Fallback for older Groq error formats
        msg_lower = message.lower()
        if "per day" in msg_lower and "token" in msg_lower:
            return "TPD"
        if "per day" in msg_lower and "request" in msg_lower:
            return "RPD"
        if "per minute" in msg_lower and "token" in msg_lower:
            return "TPM"
        if "per minute" in msg_lower and "request" in msg_lower:
            return "RPM"
        if "429" in message or "rate_limit" in msg_lower:
            return "TPM"   # safe default for ambiguous 429
        return "UNKNOWN"

    @classmethod
    def _extract_int(cls, pattern: re.Pattern, text: str) -> int:
        m = pattern.search(text)
        return int(m.group(1)) if m else 0

    @classmethod
    def _extract_retry_after(cls, message: str, raw: str) -> Optional[int]:
        """Extract retry_after in seconds from message or headers."""
        m = cls._RETRY_RE.search(message)
        if m:
            return max(1, int(float(m.group(1))) + _settings.retry_after_buffer_seconds)
        m = cls._RETRY_HEADER.search(raw)
        if m:
            return int(m.group(1))
        return None


# ── Module-level convenience ──────────────────────────────────────

def classify_error(error: Exception) -> RateLimitInfo:
    return RateLimitClassifier.classify(error)
