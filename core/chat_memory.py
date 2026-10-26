"""Conversation memory: persisted per user, with a 'start fresh' marker (no schema change to chat_messages)."""
from __future__ import annotations

import json

from sqlalchemy import func, select

from core import db


def _last_reset_id(username: str) -> int:
    r = db.rows(select(db.chat_resets.c.up_to_id).where(db.chat_resets.c.username == username)
                .order_by(db.chat_resets.c.id.desc()).limit(1))
    return int(r[0]["up_to_id"]) if r else 0


def load_history(username: str, limit: int = 30) -> list[dict]:
    m = db.chat_messages
    rows = db.rows(select(m).where(m.c.username == username, m.c.id > _last_reset_id(username))
                   .order_by(m.c.id.desc()).limit(limit))
    out = []
    for r in reversed(rows):
        try:
            sources = json.loads(r["sources"] or "[]")
        except ValueError:
            sources = []
        out.append({"id": r["id"], "q": r["question"], "answer": r["answer"], "sources": sources,
                    "found": bool(r["found"]), "mode": "restored"})
    return out


def reset(username: str) -> None:
    m = db.chat_messages
    top = db.rows(select(func.max(m.c.id).label("mx")).where(m.c.username == username))[0]["mx"] or 0
    db.insert(db.chat_resets, username=username, up_to_id=top, created_at=db.now())


def turns_for_prompt(history: list[dict], n: int = 4) -> list[dict]:
    """Last n answered turns, as {'q','a'} (what the model needs to resolve follow-ups)."""
    ok = [h for h in history if h.get("found")]
    return [{"q": h["q"], "a": h["answer"]} for h in ok[-n:]]
