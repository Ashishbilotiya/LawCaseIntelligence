"""
database/repositories/judgment_repository.py
Repository for ProcessedJudgment CRUD and analytics queries.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from collections import Counter

from database.database import get_session
from database.models import ProcessedJudgment


class JudgmentRepository:
    def get(self, judgment_id: str) -> Optional[dict]:
        session = get_session()
        try:
            j = session.query(ProcessedJudgment).filter_by(id=judgment_id).first()
            return j.to_dict() if j else None
        finally:
            session.close()

    def list_by_project(self, project_id: str) -> List[dict]:
        session = get_session()
        try:
            rows = session.query(ProcessedJudgment).filter_by(project_id=project_id)\
                          .order_by(ProcessedJudgment.created_at.desc()).all()
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def list_all(self, limit: int = 100) -> List[dict]:
        session = get_session()
        try:
            rows = session.query(ProcessedJudgment)\
                          .order_by(ProcessedJudgment.created_at.desc()).limit(limit).all()
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def get_analytics(self, project_id: Optional[str] = None) -> Dict:
        session = get_session()
        try:
            q = session.query(ProcessedJudgment)
            if project_id:
                q = q.filter_by(project_id=project_id)
            rows = q.all()

            categories = Counter(r.case_category for r in rows)
            outcomes   = Counter(r.outcome        for r in rows)
            wins       = Counter(r.win_indicator  for r in rows)
            courts     = Counter(r.court          for r in rows)

            all_precedents: list = []
            all_sections: list   = []
            for r in rows:
                prec = r.precedents_json or {}
                for p in prec.get("precedents", []):
                    if p.get("case_name"):
                        all_precedents.append(p["case_name"])
                secs = r.frequently_cited_sections or []
                all_sections.extend(s for s in secs if s)

            top_precedents = Counter(all_precedents).most_common(10)
            top_sections   = Counter(all_sections).most_common(10)

            return {
                "total_cases":     len(rows),
                "categories":      dict(categories),
                "outcomes":        dict(outcomes),
                "win_indicators":  dict(wins),
                "courts":          dict(courts),
                "top_precedents":  [{"name": n, "count": c} for n, c in top_precedents],
                "top_sections":    [{"section": s, "count": c} for s, c in top_sections],
            }
        finally:
            session.close()

    def search(self, query: str, project_id: Optional[str] = None, limit: int = 20) -> List[dict]:
        session = get_session()
        try:
            q = session.query(ProcessedJudgment)
            if project_id:
                q = q.filter_by(project_id=project_id)
            q = q.filter(
                ProcessedJudgment.case_summary.ilike(f"%{query}%") |
                ProcessedJudgment.document_name.ilike(f"%{query}%") |
                ProcessedJudgment.case_number.ilike(f"%{query}%")
            ).limit(limit)
            return [r.to_dict() for r in q.all()]
        finally:
            session.close()
