"""
backend/models/statute_models.py
Pydantic models for law and statute extraction.

source_pages accepts str | int — LLM sometimes returns URLs or text
instead of page numbers. We coerce safely instead of rejecting the whole result.
"""
from __future__ import annotations

from typing import Any, List, Union
from pydantic import BaseModel, Field, field_validator


class StatuteSection(BaseModel):
    act:         str = Field(default="", description="Name of the Act")
    section:     str = Field(default="", description="Section number(s)")
    description: str = Field(default="", description="Brief description of the section")
    relevance:   str = Field(default="", description="How this section was applied")


class StatuteOutput(BaseModel):
    statutes: List[StatuteSection] = Field(
        default_factory=list,
        description="All statutes and sections cited",
    )
    constitutional_provisions: List[str] = Field(
        default_factory=list,
        description="Constitutional articles cited",
    )
    regulatory_references: List[str] = Field(
        default_factory=list,
        description="Regulations, rules, notifications cited",
    )
    source_pages: List[Any] = Field(default_factory=list)
    headnote: str = Field(default="")

    @field_validator("source_pages", mode="before")
    @classmethod
    def coerce_source_pages(cls, v: Any) -> List[Any]:
        """
        Accept any value in source_pages.
        Convert integers where possible; keep strings as-is.
        Skip URLs and non-page values gracefully.
        """
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str):
                stripped = item.strip()
                # Skip URLs and empty strings
                if stripped.startswith("http") or not stripped:
                    continue
                # Try to parse as int
                try:
                    result.append(int(stripped))
                except ValueError:
                    # Keep as string (e.g. "p.5", "page 3")
                    result.append(stripped)
        return result
