"""
database/repositories/precedent_repository.py
Repository for precedent analytics across all judgments.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from database.database import get_session
from database.models import ProcessedJudgment


class PrecedentRepository:
    def get_top_cited(self, project_id: Optional[str] = None, top_n: int = 20) -> List[Dict]:
        session = get_session()
        try:
            q = session.query(ProcessedJudgment)
            if project_id:
                q = q.filter_by(project_id=project_id)
            rows = q.all()

            counter: Counter = Counter()
            detail: dict = {}
            for r in rows:
                prec = r.precedents_json or {}
                for p in prec.get("precedents", []):
                    name = p.get("case_name", "").strip()
                    if name:
                        counter[name] += 1
                        if name not in detail:
                            detail[name] = {
                                "citation": p.get("citation", ""),
                                "court":    p.get("court", ""),
                                "year":     p.get("year", ""),
                            }

            return [
                {
                    "case_name": name, "count": cnt,
                    **detail.get(name, {}),
                }
                for name, cnt in counter.most_common(top_n)
            ]
        finally:
            session.close()

    def get_statute_frequency(self, project_id: Optional[str] = None, top_n: int = 20) -> List[Dict]:
        session = get_session()
        try:
            q = session.query(ProcessedJudgment)
            if project_id:
                q = q.filter_by(project_id=project_id)
            rows = q.all()

            act_counter: Counter     = Counter()
            section_counter: Counter = Counter()
            for r in rows:
                stat = r.statutes_json or {}
                for s in stat.get("statutes", []):
                    act = s.get("act", "").strip()
                    sec = s.get("section", "").strip()
                    if act:
                        act_counter[act] += 1
                    if sec:
                        section_counter[f"{act} § {sec}" if act else sec] += 1

            return {
                "top_acts":     [{"act": a, "count": c} for a, c in act_counter.most_common(top_n)],
                "top_sections": [{"section": s, "count": c} for s, c in section_counter.most_common(top_n)],
            }
        finally:
            session.close()
