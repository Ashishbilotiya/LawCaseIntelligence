"""
database/repositories/case_repository.py
Repository for Project and DocumentInProject CRUD operations.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from database.database import get_session
from database.models import Project, DocumentInProject


class ProjectRepository:
    def create(self, name: str, description: str = "") -> Project:
        session = get_session()
        try:
            p = Project(id=str(uuid.uuid4()), name=name, description=description)
            session.add(p); session.commit(); session.refresh(p)
            return p
        finally:
            session.close()

    def get(self, project_id: str) -> Optional[Project]:
        session = get_session()
        try:
            return session.query(Project).filter_by(id=project_id).first()
        finally:
            session.close()

    def list_all(self) -> List[dict]:
        session = get_session()
        try:
            return [p.to_dict() for p in session.query(Project).order_by(Project.created_at.desc()).all()]
        finally:
            session.close()

    def delete(self, project_id: str) -> bool:
        session = get_session()
        try:
            p = session.query(Project).filter_by(id=project_id).first()
            if p:
                session.delete(p); session.commit(); return True
            return False
        finally:
            session.close()


class DocumentRepository:
    def create(self, project_id: str, document_name: str, location: str = "",
               file_size: int = 0, page_count: int = 0) -> DocumentInProject:
        session = get_session()
        try:
            doc = DocumentInProject(
                id=str(uuid.uuid4()), project_id=project_id,
                document_name=document_name, document_location=location,
                file_size_bytes=file_size, page_count=page_count, status="pending",
            )
            session.add(doc); session.commit(); session.refresh(doc)
            return doc
        finally:
            session.close()

    def list_by_project(self, project_id: str) -> List[dict]:
        session = get_session()
        try:
            docs = session.query(DocumentInProject).filter_by(project_id=project_id).all()
            return [d.to_dict() for d in docs]
        finally:
            session.close()

    def update_status(self, doc_id: str, status: str) -> None:
        session = get_session()
        try:
            doc = session.query(DocumentInProject).filter_by(id=doc_id).first()
            if doc:
                doc.status = status; session.commit()
        finally:
            session.close()
