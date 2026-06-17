# backend/services/llm/__init__.py
from .groq_provider import GroqProvider, get_groq_provider
from .router import get_llm_provider, invoke_llm, quick_invoke
from .api_key_manager import get_api_key_manager, APIKeyManager
from .token_tracker import get_token_tracker, TokenTracker
from .token_scheduler import get_token_scheduler, TokenScheduler
from .rate_limit_classifier import RateLimitClassifier, RateLimitInfo, classify_error
from .diagnostics import get_llm_health_report, print_health_report

__all__ = [
    # Provider
    "GroqProvider",
    "get_groq_provider",
    # Router
    "get_llm_provider",
    "invoke_llm",
    "quick_invoke",
    # Key management
    "APIKeyManager",
    "get_api_key_manager",
    # Token tracking (single source of truth for TPM/RPM/TPD/RPD)
    "TokenTracker",
    "get_token_tracker",
    # Score-based key scheduling
    "TokenScheduler",
    "get_token_scheduler",
    # Structured rate-limit classification
    "RateLimitClassifier",
    "RateLimitInfo",
    "classify_error",
    # Diagnostics
    "get_llm_health_report",
    "print_health_report",
]
