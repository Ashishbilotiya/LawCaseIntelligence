"""
backend/models/argument_models.py
Pydantic models for argument analysis output.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class ArgumentOutput(BaseModel):
    """Output from Argument Analysis Agent."""
    petitioner_arguments: List[str] = Field(
        default_factory=list,
        description="Key arguments made by petitioner/appellant"
    )
    petitioner_citations: List[str] = Field(
        default_factory=list,
        description="Cases/statutes cited by petitioner"
    )
    respondent_arguments: List[str] = Field(
        default_factory=list,
        description="Key arguments made by respondent"
    )
    respondent_citations: List[str] = Field(
        default_factory=list,
        description="Cases/statutes cited by respondent"
    )
    key_contentions: List[str] = Field(
        default_factory=list,
        description="Primary legal contentions in dispute"
    )
    headnote: str = Field(default="")
