from __future__ import annotations

from core import db
from core.config import settings


def log(user: dict | None, module: str, action: str, model: str | None = None) -> None:
    try:
        db.insert(db.audit_log, username=(user or {}).get("username", "anonymous"),
                  role=(user or {}).get("role", ""), module=module, action=action,
                  model=model or settings()["model"], created_at=db.now())
    except Exception:
        pass  # logging must never break the app
