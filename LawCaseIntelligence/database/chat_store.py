"""
database/chat_store.py
Server-side chat history storage using SQLite.

clear_history() now DELETES the row entirely instead of just
emptying the fields — so the DB shows no record after clearing.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base

from database.database import get_session, Base as AppBase

logger = logging.getLogger(__name__)

# We use the shared Base from database.database to ensure the schema
# is registered correctly during init_db()
Base = AppBase

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    project_id = Column(String(64), primary_key=True)
    history    = Column(Text, default="[]")
    summary    = Column(Text, default="")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Public API ────────────────────────────────────────────────────

def get_history(project_id: str) -> List[Dict]:
    db = get_session()
    try:
        row = db.query(ChatSession).filter_by(project_id=project_id).first()
        return json.loads(row.history or "[]") if row else []
    except Exception as e:
        logger.warning(f"[ChatStore] get_history failed: {e}")
        return []
    finally:
        db.close()


def get_summary(project_id: str) -> str:
    db = get_session()
    try:
        row = db.query(ChatSession).filter_by(project_id=project_id).first()
        return row.summary if row else ""
    except Exception as e:
        logger.warning(f"[ChatStore] get_summary failed: {e}")
        return ""
    finally:
        db.close()


def save_history(project_id: str, history: List[Dict], summary: str = "") -> None:
    db = get_session()
    try:
        row     = db.query(ChatSession).filter_by(project_id=project_id).first()
        trimmed = history[-40:]
        if row:
            row.history    = json.dumps(trimmed)
            row.summary    = summary
            row.updated_at = datetime.now(timezone.utc)
        else:
            db.add(ChatSession(
                project_id=project_id,
                history=json.dumps(trimmed),
                summary=summary,
            ))
        db.commit()
    except Exception as e:
        logger.warning(f"[ChatStore] save_history failed: {e}")
        db.rollback()
    finally:
        db.close()


def clear_history(project_id: str) -> None:
    """
    Permanently DELETE the chat session row for this project.
    The row is removed entirely — not just emptied — so the DB
    shows no record after the user clears the conversation.
    """
    db = get_session()
    try:
        row = db.query(ChatSession).filter_by(project_id=project_id).first()
        if row:
            db.delete(row)
            db.commit()
            logger.info(f"[ChatStore] Chat history deleted for project {project_id}")
        else:
            logger.debug(f"[ChatStore] No chat session found for project {project_id}")
    except Exception as e:
        logger.warning(f"[ChatStore] clear_history failed: {e}")
        db.rollback()
    finally:
        db.close()
