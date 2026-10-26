"""Rubric loading/validation and safe handling of model-suggested marks."""
from __future__ import annotations

CONF_THRESHOLD = 0.6


def validate_rubric(r: dict) -> list[str]:
    errs = []
    if not r.get("course"):
        errs.append("Missing 'course'.")
    qs = r.get("questions") or []
    if not qs:
        errs.append("No 'questions' defined.")
    for q in qs:
        qid = q.get("id", "?")
        pts = q.get("points") or []
        if not pts:
            errs.append(f"{qid}: no 'points'.")
            continue
        total = sum(float(p.get("marks", 0)) for p in pts)
        if abs(total - float(q.get("max_marks", -1))) > 1e-6:
            errs.append(f"{qid}: point marks sum to {total} but max_marks is {q.get('max_marks')}.")
        ids = [p.get("id") for p in pts]
        if len(ids) != len(set(ids)):
            errs.append(f"{qid}: duplicate point ids.")
    return errs


def build_prompt(question: dict, has_image: bool) -> str:
    pts = "\n".join(f"- {p['id']} ({p['marks']} marks): {p['text']}" for p in question["points"])
    source = "the attached image of a handwritten answer" if has_image else "the typed answer below"
    return f"""You assist a faculty member evaluating a student's answer. Be strict and fair.
QUESTION ({question['id']}, {question['max_marks']} marks): {question['text']}
RUBRIC POINTS:
{pts}

Read {source}. First transcribe it. Then, for EACH rubric point decide:
status = "covered" (full marks), "partial" (some marks) or "missing" (0), give a short evidence quote
from the answer (empty if missing), suggested_marks (0 to that point's marks), and confidence 0-1.
Also write 2-3 sentences of constructive feedback for the student.
Return JSON: {{"transcript": str, "points": [{{"id": str, "status": str, "evidence": str,
"suggested_marks": number, "confidence": number}}], "feedback": str}}"""


def normalize_result(question: dict, result: dict) -> dict:
    """Clamp marks to rubric limits, fill missing points, flag low confidence. Never trust raw model marks."""
    by_id = {p.get("id"): p for p in (result.get("points") or []) if isinstance(p, dict)}
    rows = []
    for p in question["points"]:
        r = by_id.get(p["id"], {})
        try:
            marks = float(r.get("suggested_marks", 0) or 0)
        except (TypeError, ValueError):
            marks = 0.0
        marks = max(0.0, min(marks, float(p["marks"])))
        try:
            conf = float(r.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(conf, 1.0))
        rows.append({"id": p["id"], "criterion": p["text"], "max": float(p["marks"]),
                     "suggested": marks, "final": marks,
                     "status": r.get("status", "missing") if r else "missing",
                     "confidence": conf, "needs_review": conf < CONF_THRESHOLD or p["id"] not in by_id,
                     "evidence": str(r.get("evidence", ""))})
    total = sum(r["suggested"] for r in rows)
    return {"transcript": str(result.get("transcript", "")), "points": rows,
            "feedback": str(result.get("feedback", "")), "total": total,
            "max_marks": float(question["max_marks"])}
