"""
backend/models/case_models.py
Pydantic models for case-level data.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class CaseMetadata(BaseModel):
    """Metadata extracted from judgment header."""
    court: str = Field(default="", description="Court name")
    case_number: str = Field(default="", description="Case number / citation")
    date_of_judgment: str = Field(default="", description="Date of judgment")
    judges: List[str] = Field(default_factory=list, description="List of judge names")
    petitioner_name: str = Field(default="", description="Petitioner/appellant name")
    respondent_name: str = Field(default="", description="Respondent name")
    case_type: str = Field(default="", description="Type: Civil/Criminal/Writ etc.")


class ProcessedCase(BaseModel):
    """Full structured output for a processed judgment."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(default="")
    document_name: str = Field(default="")
    metadata: CaseMetadata = Field(default_factory=CaseMetadata)
    case_category: str = Field(default="General")
    win_indicator: str = Field(default="Neutral")
    outcome: str = Field(default="")
    case_summary: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class CaseListItem(BaseModel):
    """Lightweight case item for list/dashboard views."""
    id: str
    document_name: str
    court: str = ""
    case_number: str = ""
    date_of_judgment: str = ""
    case_category: str = ""
    win_indicator: str = ""
    outcome: str = ""
    created_at: str = ""
