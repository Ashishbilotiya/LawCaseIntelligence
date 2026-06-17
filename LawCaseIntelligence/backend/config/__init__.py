# backend/config/__init__.py
from .settings import get_settings, Settings
from .logging_config import setup_logging, get_logger
from .constants import LEGAL_CATEGORIES, WIN_INDICATORS, OUTCOME_TYPES, AGENT_NAMES

__all__ = [
    "get_settings", "Settings",
    "setup_logging", "get_logger",
    "LEGAL_CATEGORIES", "WIN_INDICATORS", "OUTCOME_TYPES", "AGENT_NAMES",
]
