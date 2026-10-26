"""Database layer (SQLAlchemy Core). SQLite by default; set DATABASE_URL for Postgres."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData, String, Table,
                        Text, create_engine)

from core.config import settings

metadata = MetaData()

users = Table("users", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(80), unique=True, nullable=False),
    Column("name", String(200)),
    Column("role", String(20), nullable=False),
    Column("password_hash", String(300), nullable=False),
    Column("active", Boolean, default=True))

documents = Table("documents", metadata,
    Column("id", Integer, primary_key=True),
    Column("title", String(300)),
    Column("category", String(50)),
    Column("audience", String(20)),
    Column("version", String(30)),
    Column("effective_date", String(20)),
    Column("status", String(20)),
    Column("filename", String(300)),
    Column("uploaded_by", String(80)),
    Column("created_at", DateTime))

chunks = Table("chunks", metadata,
    Column("id", Integer, primary_key=True),
    Column("document_id", Integer, index=True),
    Column("page", Integer),
    Column("text", Text),
    Column("embedding", Text))

chat_messages = Table("chat_messages", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(80)),
    Column("question", Text),
    Column("answer", Text),
    Column("sources", Text),
    Column("found", Boolean),
    Column("feedback", Integer),
    Column("created_at", DateTime))

evaluations = Table("evaluations", metadata,
    Column("id", Integer, primary_key=True),
    Column("faculty", String(80)),
    Column("course", String(50)),
    Column("question_id", String(20)),
    Column("student_ref", String(100)),
    Column("transcript", Text),
    Column("result_json", Text),
    Column("final_marks", Float),
    Column("max_marks", Float),
    Column("status", String(20)),
    Column("created_at", DateTime))

paper_audits = Table("paper_audits", metadata,
    Column("id", Integer, primary_key=True),
    Column("faculty", String(80)),
    Column("title", String(300)),
    Column("questions_json", Text),
    Column("coverage_json", Text),
    Column("created_at", DateTime))

past_questions = Table("past_questions", metadata,
    Column("id", Integer, primary_key=True),
    Column("paper_title", String(300)),
    Column("year", Integer),
    Column("question_no", String(20)),
    Column("question_text", Text),
    Column("marks", Float),
    Column("embedding", Text))

audit_log = Table("audit_log", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(80)),
    Column("role", String(20)),
    Column("module", String(60)),
    Column("action", String(100)),
    Column("model", String(100)),
    Column("created_at", DateTime))

chat_resets = Table("chat_resets", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(80), index=True),
    Column("up_to_id", Integer),
    Column("created_at", DateTime))

timetables = Table("timetables", metadata,
    Column("id", Integer, primary_key=True),
    Column("share_code", String(40), unique=True),
    Column("owner", String(80)),
    Column("title", String(300)),
    Column("payload", Text),
    Column("created_at", DateTime))

interview_sessions = Table("interview_sessions", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(80), index=True),
    Column("mode", String(30)),
    Column("category", String(100)),
    Column("question", Text),
    Column("transcript", Text),
    Column("correctness", Float),
    Column("clarity", Float),
    Column("overall", Float),
    Column("feedback", Text),
    Column("created_at", DateTime))

profiles = Table("profiles", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(80), unique=True),
    Column("name", String(200)),
    Column("skills", Text),
    Column("interests", Text),
    Column("open_to_team", Boolean, default=False),
    Column("updated_at", DateTime))

projects = Table("projects", metadata,
    Column("id", Integer, primary_key=True),
    Column("owner", String(80), index=True),
    Column("owner_name", String(200)),
    Column("title", String(300)),
    Column("description", Text),
    Column("link", String(500)),
    Column("tech", Text),
    Column("domains", Text),
    Column("difficulty", String(30)),
    Column("summary", Text),
    Column("embedding", Text),
    Column("created_at", DateTime))

_engine = None
_engine_url = None


def normalized_url() -> str:
    url = settings()["database_url"]
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def engine():
    global _engine, _engine_url
    url = normalized_url()
    if _engine is None or url != _engine_url:
        kw = {}
        if url.startswith("sqlite"):
            path = url.replace("sqlite:///", "", 1)
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            kw["connect_args"] = {"check_same_thread": False}
        else:
            kw["pool_pre_ping"] = True
        _engine = create_engine(url, **kw)
        _engine_url = url
    return _engine


def dialect() -> str:
    return engine().dialect.name


def init() -> None:
    metadata.create_all(engine())


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def insert(table: Table, **values) -> int:
    with engine().begin() as c:
        res = c.execute(table.insert().values(**values))
        return res.inserted_primary_key[0]


def rows(stmt) -> list[dict]:
    with engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(stmt)]


def execute(stmt) -> int:
    with engine().begin() as c:
        return c.execute(stmt).rowcount
