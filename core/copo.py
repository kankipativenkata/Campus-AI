"""CO-PO mapping helpers: question parsing, Bloom tagging, coverage, PO weights, repeat detection."""
from __future__ import annotations

import re

import numpy as np

from core.rag import bm25_scores, tokenize

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
BLOOM_VERBS = {
    "Remember": ["define", "list", "state", "name", "identify", "recall", "mention", "what is", "write down"],
    "Understand": ["explain", "describe", "differentiate", "distinguish", "summarize", "discuss", "illustrate", "classify"],
    "Apply": ["apply", "solve", "implement", "write an algorithm", "write a program", "trace", "compute", "calculate", "demonstrate", "use"],
    "Analyze": ["analyze", "analyse", "compare", "contrast", "examine", "break down", "derive", "differentiate between"],
    "Evaluate": ["evaluate", "justify", "assess", "critique", "judge", "recommend", "defend"],
    "Create": ["design", "develop", "construct", "formulate", "propose", "create", "build"],
}

Q_RE = re.compile(r"^\s*(?:Q\.?\s*)?(\d{1,2}[a-z]?)[\.\)]\s+(.*)$", re.I)
MARK_RE = re.compile(r"[\(\[]\s*(\d{1,2})\s*(?:marks?|m)?\s*[\)\]]", re.I)
SECTION_RE = re.compile(r"^\s*(section|part|total|time|instructions?)\b", re.I)


def parse_questions_regex(text: str) -> list[dict]:
    qs: list[dict] = []
    cur = None
    for line in text.splitlines():
        if SECTION_RE.match(line):
            cur = None
            continue
        m = Q_RE.match(line)
        if m:
            cur = {"no": m.group(1).upper() if m.group(1).isdigit() else m.group(1), "text": m.group(2).strip()}
            qs.append(cur)
        elif cur is not None and line.strip():
            cur["text"] += " " + line.strip()
    for q in qs:
        marks = MARK_RE.findall(q["text"])
        q["marks"] = float(marks[-1]) if marks else 1.0
        q["text"] = MARK_RE.sub("", q["text"]).strip()
    return qs


def bloom_by_verb(text: str) -> str:
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    best, pos = "Understand", 10**9
    for level, verbs in BLOOM_VERBS.items():
        for v in verbs:
            i = t.find(" " + v + " ")
            if i != -1 and i < pos:
                best, pos = level, i
    return best


def co_by_overlap(question: str, cos: list[dict]) -> str:
    if not cos:
        return "Unmapped"
    scores = bm25_scores(tokenize(question), [tokenize(c["text"]) for c in cos])
    if max(scores) <= 0:
        return "Unmapped"
    return cos[int(np.argmax(scores))]["id"]


def coverage(questions: list[dict], cos: list[dict], bloom_targets: dict | None = None) -> dict:
    total = sum(q.get("marks", 0) for q in questions) or 1.0
    by_co = {c["id"]: 0.0 for c in cos}
    by_bloom = {b: 0.0 for b in BLOOM_ORDER}
    for q in questions:
        by_co[q.get("co", "Unmapped")] = by_co.get(q.get("co", "Unmapped"), 0.0) + q.get("marks", 0)
        by_bloom[q.get("bloom", "Understand")] = by_bloom.get(q.get("bloom", "Understand"), 0.0) + q.get("marks", 0)
    pct_co = {k: round(100 * v / total, 1) for k, v in by_co.items()}
    pct_bl = {k: round(100 * v / total, 1) for k, v in by_bloom.items()}
    warnings = [f"{c['id']} has zero coverage in this paper." for c in cos if by_co.get(c["id"], 0) == 0]
    for level, rule in (bloom_targets or {}).items():
        p = pct_bl.get(level, 0.0)
        if "max_pct" in rule and p > rule["max_pct"]:
            warnings.append(f"{level} is {p}% (target max {rule['max_pct']}%).")
        if "min_pct" in rule and p < rule["min_pct"]:
            warnings.append(f"{level} is {p}% (target min {rule['min_pct']}%).")
    return {"total_marks": total, "marks_by_co": by_co, "pct_by_co": pct_co,
            "marks_by_bloom": by_bloom, "pct_by_bloom": pct_bl, "warnings": warnings}


def po_weights(pct_by_co: dict, matrix: dict) -> dict:
    """PO weight = sum over COs of (CO share of marks x mapping level / 3), as a percentage."""
    out: dict[str, float] = {}
    for co, pct in pct_by_co.items():
        for po, level in (matrix.get(co) or {}).items():
            out[po] = round(out.get(po, 0.0) + pct * float(level) / 3.0, 1)
    return dict(sorted(out.items(), key=lambda kv: int("".join(ch for ch in kv[0] if ch.isdigit()) or 0)))


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x and y else 0.0


def similarity(a_text: str, a_vec, b_text: str, b_vec) -> float:
    if a_vec is not None and b_vec is not None and len(a_vec) == len(b_vec):
        a, b = np.array(a_vec), np.array(b_vec)
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d else 0.0
    return _jaccard(a_text, b_text)


def find_repeats(questions: list[dict], past: list[dict], q_vecs=None, past_vecs=None,
                 threshold_vec: float = 0.88, threshold_text: float = 0.6) -> list[dict]:
    reps = []
    for i, q in enumerate(questions):
        qv = q_vecs[i] if q_vecs else None
        for j, p in enumerate(past):
            pv = past_vecs[j] if past_vecs else None
            s = similarity(q["text"], qv, p["question_text"], pv)
            th = threshold_vec if (qv is not None and pv is not None) else threshold_text
            if s >= th:
                reps.append({"question": q["no"], "matches": f"{p['paper_title']} Q{p['question_no']}",
                             "similarity": round(s, 2), "past_text": p["question_text"]})
    return reps
