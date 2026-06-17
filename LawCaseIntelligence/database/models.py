"""
database/models.py
SQLAlchemy ORM models for LawCaseIntelligence.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, Boolean, JSON, Float, ForeignKey
)
from sqlalchemy.orm import relationship

from database.database import Base


def _now():
    return datetime.utcnow()


class Project(Base):
    __tablename__ = "projects"

    id                 = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name               = Column(String(255), nullable=False)
    description        = Column(Text, default="")
    our_argument_side  = Column(String(255), default="")   # FIX: was missing — used in templates
    created_at         = Column(DateTime, default=_now)
    updated_at         = Column(DateTime, default=_now, onupdate=_now)

    documents  = relationship("DocumentInProject", back_populates="project", cascade="all, delete-orphan")
    judgments  = relationship("ProcessedJudgment",  back_populates="project", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "our_argument_side": self.our_argument_side or "",   # FIX: exposed in dict
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            "document_count": len(self.documents),
        }


class DocumentInProject(Base):
    __tablename__ = "documents_in_project"

    id                = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id        = Column(String(36), ForeignKey("projects.id"), nullable=False)
    document_name     = Column(String(255), nullable=False)
    document_location = Column(Text, default="")
    file_size_bytes   = Column(Integer, default=0)
    page_count        = Column(Integer, default=0)
    status            = Column(String(32), default="pending")
    uploaded_at       = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="documents")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "project_id": self.project_id,
            "document_name": self.document_name, "status": self.status,
            "page_count": self.page_count, "uploaded_at": str(self.uploaded_at),
        }


class ProcessedJudgment(Base):
    __tablename__ = "processed_judgments"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id     = Column(String(36), ForeignKey("projects.id"), nullable=False)
    document_name  = Column(String(255), nullable=False)
    court          = Column(String(255), default="")
    case_number    = Column(String(255), default="")
    date_of_judgment = Column(String(64), default="")
    case_category  = Column(String(64), default="General")
    win_indicator  = Column(String(64), default="Neutral")
    outcome        = Column(String(255), default="")

    issue_json       = Column(JSON, default=dict)
    petitioner_json  = Column(JSON, default=dict)
    respondent_json  = Column(JSON, default=dict)
    statutes_json    = Column(JSON, default=dict)
    precedents_json  = Column(JSON, default=dict)
    reasoning_json   = Column(JSON, default=dict)
    trends_json      = Column(JSON, default=dict)

    case_summary              = Column(Text, default="")
    frequently_cited_sections = Column(JSON, default=list)
    processing_log            = Column(JSON, default=list)

    created_at = Column(DateTime, default=_now)
    project    = relationship("Project", back_populates="judgments")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "project_id": self.project_id,
            "document_name": self.document_name, "court": self.court,
            "case_number": self.case_number, "date_of_judgment": self.date_of_judgment,
            "case_category": self.case_category, "win_indicator": self.win_indicator,
            "outcome": self.outcome, "case_summary": self.case_summary,
            "issue": self.issue_json, "statutes": self.statutes_json,
            "precedents": self.precedents_json, "reasoning": self.reasoning_json,
            "petitioner_arguments": self.petitioner_json,
            "respondent_arguments": self.respondent_json,
            "trends": self.trends_json, "created_at": str(self.created_at),
        }
