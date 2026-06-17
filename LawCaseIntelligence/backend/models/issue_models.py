"""
backend/models/issue_models.py
Pydantic models for legal issue extraction.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class IssueOutput(BaseModel):
    """Output from Issue Extraction Agent."""
    primary_issue: str = Field(
        default="",
        description="Central legal question the court had to decide"
    )
    sub_issues: List[str] = Field(
        default_factory=list,
        description="Secondary legal questions or sub-issues"
    )
    case_background: str = Field(
        default="",
        description="Brief background of the case"
    )
    dispute_category: str = Field(
        default="",
        description="Category: Civil/Criminal/Constitutional/Tax/etc."
    )
    source_pages: List[int] = Field(
        default_factory=list,
        description="PDF page numbers where issues were found"
    )
    headnote: str = Field(
        default="",
        description="One-sentence headnote for this issue"
    )
