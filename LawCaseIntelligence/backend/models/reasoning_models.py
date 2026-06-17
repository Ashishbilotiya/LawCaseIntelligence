"""
backend/models/reasoning_models.py
Pydantic models for judicial reasoning and verdict extraction.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class ReasoningOutput(BaseModel):
    """Output from Reasoning & Verdict Agent."""
    judicial_reasoning: str = Field(
        default="",
        description="Court's reasoning and analysis"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key factual and legal findings"
    )
    principles_applied: List[str] = Field(
        default_factory=list,
        description="Legal principles applied by the court"
    )
    outcome: str = Field(
        default="",
        description="Final outcome: Allowed/Dismissed/Partly Allowed/Remanded"
    )
    relief_granted: str = Field(
        default="",
        description="Specific relief granted"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions attached to the order"
    )
    source_pages: List[int] = Field(default_factory=list)
    headnote: str = Field(default="")


class HeadnoteOutput(BaseModel):
    """Generated headnotes for the judgment."""
    headnotes: List[str] = Field(
        default_factory=list,
        description="List of concise legal headnotes"
    )
    summary: str = Field(
        default="",
        description="Plain-language 7-point case summary"
    )
    case_category: str = Field(default="General")
    win_indicator: str = Field(default="Neutral")
