"""
backend/models/precedent_models.py
Pydantic models for precedent analysis output.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class PrecedentCase(BaseModel):
    """A single cited precedent."""
    case_name: str = Field(default="")
    citation: str = Field(default="", description="AIR/SCC/SCR citation")
    court: str = Field(default="")
    year: str = Field(default="")
    relevance: str = Field(default="", description="How/why this case was cited")
    distinguished_or_followed: str = Field(
        default="",
        description="Whether court followed/distinguished/overruled this precedent"
    )
    importance_score: int = Field(
        default=1,
        ge=1, le=5,
        description="Citation importance 1-5"
    )


class PrecedentOutput(BaseModel):
    """Output from Precedent Analysis Agent."""
    precedents: List[PrecedentCase] = Field(default_factory=list)
    total_citations: int = Field(default=0)
    source_pages: List[int] = Field(default_factory=list)
    headnote: str = Field(default="")
