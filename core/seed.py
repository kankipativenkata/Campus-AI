"""First-run bootstrap: create tables, demo users, sample documents and a past question paper."""
from __future__ import annotations

from sqlalchemy import select

import json

from core import copo, db, projects, rag
from core.auth import seed_default_users
from core.config import SAMPLE_DOCS, TEST_DATA, load_yaml, university


def seed_all() -> dict:
    db.init()
    seed_default_users()
    info = {"docs": 0, "past_questions": 0}
    if university().get("seed_sample_data", True):
        if not db.rows(select(db.documents.c.id).limit(1)) and (SAMPLE_DOCS / "manifest.yaml").exists():
            for d in load_yaml(SAMPLE_DOCS / "manifest.yaml").get("documents", []):
                p = SAMPLE_DOCS / d["file"]
                if p.exists():
                    try:
                        rag.ingest_document(p.name, p.read_bytes(), d["title"], d.get("category", "general"),
                                            d.get("audience", "all"), str(d.get("version", "1.0")),
                                            str(d.get("effective_date", "")), "seed")
                        info["docs"] += 1
                    except Exception:  # noqa: BLE001
                        pass
        prev = TEST_DATA / "question_papers" / "previous_paper_2025.txt"
        if prev.exists() and not db.rows(select(db.past_questions.c.id).limit(1)):
            for q in copo.parse_questions_regex(prev.read_text(encoding="utf-8")):
                db.insert(db.past_questions, paper_title="Data Structures 2025", year=2025,
                          question_no=q["no"], question_text=q["text"], marks=q["marks"], embedding=None)
                info["past_questions"] += 1
    info["projects"] = _seed_projects() if university().get("seed_sample_data", True) else 0
    return info


def _seed_projects() -> int:
    """Fictional sample projects so similar-project and teammate suggestions work on first launch."""
    path = TEST_DATA / "projects_sample.json"
    if not path.exists() or db.rows(select(db.projects.c.id).limit(1)):
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    for p in data["profiles"]:
        projects.save_profile(p["username"], p["name"], p["skills"], p["interests"], p["open_to_team"])
    for p in data["projects"]:
        projects.add_project(p["owner"], p["owner_name"], p["title"], p["description"], p["link"],
                             projects.auto_tag(p["title"], p["description"], use_ai=False), embed=False)
    return len(data["projects"])
