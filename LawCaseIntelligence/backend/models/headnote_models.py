"""
backend/models/headnote_models.py
Pydantic models for headnote generation.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class Headnote(BaseModel):
    """A single legal headnote."""
    text: str = Field(description="The headnote text")
    category: str = Field(default="", description="Legal category")
    relevant_section: str = Field(default="", description="Related act/section")


class HeadnoteCollection(BaseModel):
    """Collection of headnotes for a judgment."""
    headnotes: List[Headnote] = Field(default_factory=list)
    total: int = Field(default=0)
