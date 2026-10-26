"""Project showcase: auto-tagging (rules + optional AI), similar projects, teammate suggestions."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
from sqlalchemy import select

from core import db, llm
from core.rag import tokenize

TECH = {
    "python": ["python", "pandas", "numpy"], "javascript": ["javascript", "node.js", "nodejs", "typescript"],
    "react": ["react", "next.js", "nextjs"], "streamlit": ["streamlit"], "flask": ["flask"], "fastapi": ["fastapi"],
    "django": ["django"], "java": ["java", "spring"], "c++": ["c++", "cpp"], "sql": ["sql", "mysql", "postgres", "sqlite", "postgresql"],
    "mongodb": ["mongodb", "mongo"], "firebase": ["firebase"], "tensorflow": ["tensorflow", "keras"], "pytorch": ["pytorch", "torch"],
    "scikit-learn": ["scikit-learn", "sklearn"], "opencv": ["opencv"], "nlp": ["nlp", "natural language", "transformer", "bert"],
    "llm": ["llm", "gemma", "gpt", "llama", "language model", "rag", "langchain"], "yolo": ["yolo"],
    "docker": ["docker"], "aws": ["aws", "ec2", "s3 bucket"], "gcp": ["gcp", "google cloud"], "azure": ["azure"],
    "arduino": ["arduino"], "esp32": ["esp32", "esp8266"], "raspberry-pi": ["raspberry"], "flutter": ["flutter", "dart"],
    "android": ["android", "kotlin"], "solidity": ["solidity", "ethereum", "smart contract"], "unity": ["unity"],
    "git": ["github", "git "], "html-css": ["html", "css", "bootstrap", "tailwind"], "tableau": ["tableau", "power bi", "powerbi"],
}
DOMAINS = {
    "AI / ML": ["machine learning", "deep learning", "neural", "classification", "prediction", "model", "nlp", "llm", "rag", "yolo", "detection"],
    "Web": ["website", "web app", "frontend", "backend", "portal", "react", "flask", "django", "rest api"],
    "Mobile": ["android", "ios", "flutter", "mobile app"],
    "IoT / Embedded": ["arduino", "esp32", "sensor", "iot", "raspberry", "embedded", "microcontroller"],
    "Data Analytics": ["dashboard", "analytics", "visualization", "tableau", "power bi", "eda", "analysis"],
    "Cybersecurity": ["security", "encryption", "vulnerability", "phishing", "malware", "authentication"],
    "Blockchain": ["blockchain", "solidity", "ethereum", "smart contract"],
    "Healthcare": ["medical", "health", "patient", "disease", "mri", "diagnosis"],
    "EdTech": ["education", "student", "learning platform", "quiz", "tutor", "campus"],
    "Cloud / DevOps": ["docker", "kubernetes", "aws", "ci/cd", "deployment", "devops"],
    "Games": ["game", "unity", "unreal"],
}


def _has(t: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", t) is not None


def rule_tags(text: str) -> dict:
    t = re.sub(r"\s+", " ", text.lower())
    tech = [k for k, al in TECH.items() if any(_has(t, a.strip()) for a in al)]
    domains = [d for d, kws in DOMAINS.items() if any(_has(t, k) for k in kws)]
    return {"tech": tech, "domains": domains}


def auto_tag(title: str, description: str, use_ai: bool = True) -> dict:
    text = f"{title}. {description}"
    base = rule_tags(text)
    out = {"tech": base["tech"], "domains": base["domains"], "difficulty": "Intermediate",
           "summary": (description.strip().split(". ")[0][:200] if description.strip() else title), "source": "rules"}
    n = len(text.split())
    out["difficulty"] = "Beginner" if n < 40 and len(base["tech"]) <= 1 else "Advanced" if len(base["tech"]) >= 4 else "Intermediate"
    if use_ai and llm.available():
        try:
            r = llm.generate_json(f"""Project title: {title}
Description: {description}
Extract tags. Return JSON: {{"tech_stack": [lowercase technology names], "domains": [from: {list(DOMAINS)}],
"difficulty": "Beginner|Intermediate|Advanced", "summary": "one sentence"}}""")
            tech = {str(x).lower().strip() for x in r.get("tech_stack", []) if str(x).strip()}
            out["tech"] = sorted(set(out["tech"]) | tech)
            out["domains"] = sorted(set(out["domains"]) | {d for d in r.get("domains", []) if d in DOMAINS})
            if r.get("difficulty") in ("Beginner", "Intermediate", "Advanced"):
                out["difficulty"] = r["difficulty"]
            out["summary"] = str(r.get("summary") or out["summary"])
            out["source"] = "AI + rules"
        except (llm.LLMError, AttributeError, TypeError):
            pass
    return out


def _tfidf_cos(a_tokens, b_tokens, df: Counter, n: int) -> float:
    def vec(toks):
        tf = Counter(toks)
        return {t: (1 + math.log(c)) * math.log(1 + n / (1 + df[t])) for t, c in tf.items()}
    va, vb = vec(a_tokens), vec(b_tokens)
    num = sum(va[t] * vb.get(t, 0) for t in va)
    den = math.sqrt(sum(x * x for x in va.values())) * math.sqrt(sum(x * x for x in vb.values()))
    return num / den if den else 0.0


def _jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def similar_projects(target: dict, others: list[dict], k: int = 5) -> list[dict]:
    """target/others: dicts with title, description, tech(list), domains(list), embedding(list|None)."""
    docs = [tokenize(o["title"] + " " + o["description"]) for o in others]
    tt = tokenize(target["title"] + " " + target["description"])
    df: Counter = Counter()
    for d in docs + [tt]:
        for t in set(d):
            df[t] += 1
    out = []
    for o, d in zip(others, docs):
        if target.get("embedding") and o.get("embedding") and len(target["embedding"]) == len(o["embedding"]):
            a, b = np.array(target["embedding"]), np.array(o["embedding"])
            text_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) or 1))
        else:
            text_sim = _tfidf_cos(tt, d, df, len(docs) + 1)
        tag_sim = 0.6 * _jacc(set(target["tech"]), set(o["tech"])) + 0.4 * _jacc(set(target["domains"]), set(o["domains"]))
        shared = sorted((set(target["tech"]) & set(o["tech"])) | (set(target["domains"]) & set(o["domains"])))
        out.append({**o, "score": round(0.5 * text_sim + 0.5 * tag_sim, 3), "shared": shared})
    return sorted(out, key=lambda x: -x["score"])[:k]


def teammate_suggestions(target: dict, similar: list[dict], profiles: list[dict], me: str, k: int = 5) -> list[dict]:
    """Complementary teammates: people who know technologies that similar projects use but this project lacks."""
    mine = set(target["tech"]) | set(target.get("my_skills", []))
    gap = {t for s in similar for t in s["tech"]} - mine
    out = []
    for p in profiles:
        if p["username"] == me or not p["open_to_team"]:
            continue
        theirs = set(p["skills"])
        fills = sorted(theirs & gap)
        dom = set(target["domains"]) & set(p.get("interest_domains", []))
        score = 0.6 * (len(fills) / max(1, len(gap))) + 0.4 * (len(dom) / max(1, len(target["domains"])))
        if score > 0:
            out.append({"name": p["name"] or p["username"], "username": p["username"], "score": round(score, 3),
                        "brings": fills, "shared_interests": sorted(dom)})
    return sorted(out, key=lambda x: -x["score"])[:k]


def load_projects() -> list[dict]:
    rows = db.rows(select(db.projects).order_by(db.projects.c.id.desc()))
    for r in rows:
        r["tech"] = json.loads(r["tech"] or "[]")
        r["domains"] = json.loads(r["domains"] or "[]")
        r["embedding"] = json.loads(r["embedding"]) if r["embedding"] else None
    return rows


def load_profiles() -> list[dict]:
    rows = db.rows(select(db.profiles))
    for r in rows:
        r["skills"] = json.loads(r["skills"] or "[]")
        r["interest_domains"] = [x for x in json.loads(r["interests"] or "[]") if x in DOMAINS]
        r["open_to_team"] = bool(r["open_to_team"])
    return rows


def save_profile(username: str, name: str, skills: list, interests: list, open_to_team: bool) -> None:
    from sqlalchemy import delete
    db.execute(delete(db.profiles).where(db.profiles.c.username == username))
    db.insert(db.profiles, username=username, name=name, skills=json.dumps(sorted(set(skills))),
              interests=json.dumps(sorted(set(interests))), open_to_team=open_to_team, updated_at=db.now())


def add_project(owner: str, owner_name: str, title: str, description: str, link: str, tags: dict, embed: bool = True) -> int:
    vec = llm.embed([f"{title}. {description}"], "SEMANTIC_SIMILARITY") if embed else None
    return db.insert(db.projects, owner=owner, owner_name=owner_name, title=title, description=description, link=link,
                     tech=json.dumps(tags["tech"]), domains=json.dumps(tags["domains"]), difficulty=tags["difficulty"],
                     summary=tags["summary"], embedding=json.dumps(vec[0]) if vec else None, created_at=db.now())


def delete_project(pid: int, owner: str, is_admin: bool = False) -> None:
    from sqlalchemy import delete
    cond = [db.projects.c.id == pid] + ([] if is_admin else [db.projects.c.owner == owner])
    db.execute(delete(db.projects).where(*cond))
