"""Password hashing (PBKDF2, stdlib only) and user management."""
from __future__ import annotations

import hashlib
import hmac
import secrets

from sqlalchemy import select, update

from core import db

ROLES = ("student", "faculty", "admin")
ITER = 120_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(8)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITER).hex()
    return f"{salt}${dk}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, dk = stored.split("$", 1)
    except ValueError:
        return False
    test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITER).hex()
    return hmac.compare_digest(test, dk)


def authenticate(username: str, password: str) -> dict | None:
    rows = db.rows(select(db.users).where(db.users.c.username == (username or "").strip().lower()))
    if not rows:
        return None
    u = rows[0]
    if not u["active"] or not verify_password(password or "", u["password_hash"]):
        return None
    return {"id": u["id"], "username": u["username"], "name": u["name"], "role": u["role"]}


def create_user(username: str, name: str, role: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Username and password are required."
    if role not in ROLES:
        return False, f"Role must be one of {ROLES}."
    if db.rows(select(db.users.c.id).where(db.users.c.username == username)):
        return False, f"User '{username}' already exists."
    db.insert(db.users, username=username, name=name or username, role=role,
              password_hash=hash_password(password), active=True)
    return True, "created"


def list_users() -> list[dict]:
    return db.rows(select(db.users.c.id, db.users.c.username, db.users.c.name,
                          db.users.c.role, db.users.c.active).order_by(db.users.c.id))


def set_active(username: str, active: bool) -> None:
    db.execute(update(db.users).where(db.users.c.username == username).values(active=active))


def set_password(username: str, password: str) -> None:
    db.execute(update(db.users).where(db.users.c.username == username)
               .values(password_hash=hash_password(password)))


DEMO_USERS = [
    ("student1", "Demo Student", "student", "student123"),
    ("faculty1", "Demo Faculty", "faculty", "faculty123"),
    ("admin", "Demo Admin", "admin", "admin123"),
]


def seed_default_users() -> None:
    if db.rows(select(db.users.c.id).limit(1)):
        return
    for u in DEMO_USERS:
        create_user(*u)
