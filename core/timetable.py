"""Timetable generator (theory + lab blocks) with hard-constraint validation and share/export helpers.
Pure Python: randomized constructive search with restarts. Hard constraints: no faculty/section/room double-booking,
labs in lab rooms as consecutive periods that never cross a break, faculty unavailability respected."""
from __future__ import annotations

import colorsys
import hashlib
import html
import io
import json
import random
import secrets
from collections import defaultdict

import pandas as pd
from sqlalchemy import select

from core import db

DAYS_ALL = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _fmt(t: int) -> str:
    return f"{t // 60:02d}:{t % 60:02d}"


def make_periods(start: str = "09:00", n_periods: int = 8, duration: int = 50, breaks: dict | None = None) -> list[dict]:
    """breaks = {after_period_number: minutes}. Break rows are non-teaching."""
    h, m = map(int, start.split(":"))
    t, out, breaks = h * 60 + m, [], breaks or {}
    for i in range(1, n_periods + 1):
        out.append({"label": f"P{i}", "start": _fmt(t), "end": _fmt(t + duration), "teach": True})
        t += duration
        if i in breaks and breaks[i] > 0:
            out.append({"label": "Break", "start": _fmt(t), "end": _fmt(t + int(breaks[i])), "teach": False})
            t += int(breaks[i])
    return out


def build_sessions(courses: list[dict]) -> list[dict]:
    sessions = []
    for c in courses:
        if not str(c.get("course", "")).strip():
            continue
        base = {"course": str(c["course"]).strip(), "name": str(c.get("name") or "").strip(),
                "faculty": str(c.get("faculty") or "TBA").strip(), "section": str(c.get("section") or "A").strip(),
                "assistant": str(c.get("lab_assistant") or "").strip()}
        for _ in range(int(c.get("theory_hours") or 0)):
            sessions.append({**base, "kind": "theory", "len": 1})
        remaining, block = int(c.get("lab_hours") or 0), max(1, int(c.get("lab_block") or 2))
        while remaining > 0:
            L = min(block, remaining)
            sessions.append({**base, "kind": "lab", "len": L})
            remaining -= L
    return sessions


def _attempt(cfg: dict, courses: list[dict], rng: random.Random) -> dict:
    periods, days = cfg["periods"], cfg["days"]
    theory_rooms, lab_rooms = cfg["theory_rooms"] or ["Room-1"], cfg["lab_rooms"]
    unavailable = {(u["faculty"], u["day"], u["period"]) for u in cfg.get("unavailable", [])}
    labels = [p["label"] for p in periods]
    teach = [p["teach"] for p in periods]
    busy: dict[tuple, str] = {}
    section_slots = defaultdict(set)            # (section, day) -> set of period idx
    fac_load = defaultdict(int)                 # (faculty, day) -> hours
    course_day = defaultdict(int)               # (course, section, kind, day) -> sessions
    section_room = {}
    placed, unplaced = [], []
    sessions = build_sessions(courses)
    rng.shuffle(sessions)
    sessions.sort(key=lambda s: (0 if s["kind"] == "lab" else 1, -s["len"]))
    penalty = 0.0
    for s in sessions:
        rooms = lab_rooms if s["kind"] == "lab" else theory_rooms
        people = [s["faculty"]] + ([s["assistant"]] if s["assistant"] else [])
        why = defaultdict(int)
        best, best_pen = None, 1e9
        if not rooms:
            unplaced.append({**s, "reason": "No lab rooms defined." if s["kind"] == "lab" else "No rooms defined."})
            continue
        for d in days:
            if s["kind"] == "lab" and course_day[(s["course"], s["section"], "lab", d)] >= 1:
                continue
            if s["kind"] == "theory" and course_day[(s["course"], s["section"], "theory", d)] >= 2:
                continue
            for st_ in range(len(periods) - s["len"] + 1):
                idxs = range(st_, st_ + s["len"])
                if not all(teach[i] for i in idxs):
                    continue
                if any((s["section"], d, i) in busy for i in idxs):
                    why["section busy"] += 1
                    continue
                if any((f, d, i) in busy or (f, d, labels[i]) in unavailable for f in people for i in idxs):
                    why["faculty busy/unavailable"] += 1
                    continue
                for room in rooms:
                    if any((room, d, i) in busy for i in idxs):
                        why["room busy"] += 1
                        continue
                    pen = 0.0
                    pen += 10 * course_day[(s["course"], s["section"], s["kind"], d)]
                    pen += 2 * len(section_slots[(s["section"], d)])
                    pen += 8 * max(0, fac_load[(s["faculty"], d)] + s["len"] - 5)
                    occ = section_slots[(s["section"], d)] | set(idxs)
                    tp = [i for i in range(min(occ), max(occ) + 1) if teach[i]]
                    pen += 3 * (len(tp) - len(occ))                      # idle gaps
                    first_half = st_ < len(periods) / 2
                    pen += (-1.5 if (s["kind"] == "lab") != first_half else 0)   # labs afternoon, theory morning
                    if s["kind"] == "theory" and section_room.get(s["section"]) not in (None, room):
                        pen += 3
                    pen += rng.random() * 0.6
                    if pen < best_pen:
                        best, best_pen = (d, st_, room), pen
        if best is None:
            top = max(why, key=why.get) if why else "no teaching slot fits"
            unplaced.append({**s, "reason": f"No free slot: mostly blocked by {top}."})
            continue
        d, st_, room = best
        for i in range(st_, st_ + s["len"]):
            for key in [(s["section"], d, i), (room, d, i)] + [(f, d, i) for f in people]:
                busy[key] = s["course"]
            section_slots[(s["section"], d)].add(i)
        for f in people:
            fac_load[(f, d)] += s["len"]
        course_day[(s["course"], s["section"], s["kind"], d)] += 1
        if s["kind"] == "theory":
            section_room.setdefault(s["section"], room)
        penalty += best_pen
        placed.append({**{k: s[k] for k in ("course", "name", "faculty", "assistant", "section", "kind")},
                       "day": d, "start": st_, "len": s["len"], "room": room})
    return {"periods": periods, "days": days, "placements": placed, "unplaced": unplaced, "penalty": round(penalty, 1)}


def solve(cfg: dict, courses: list[dict], seed: int = 0, restarts: int = 25) -> dict:
    best = None
    for r in range(restarts):
        res = _attempt(cfg, courses, random.Random(seed * 1000 + r))
        key = (len(res["unplaced"]), res["penalty"])
        if best is None or key < (len(best["unplaced"]), best["penalty"]):
            best = res
    best["cfg"] = {k: cfg.get(k) for k in ("theory_rooms", "lab_rooms")}
    best["conflicts"] = validate(best)
    return best


def validate(tt: dict) -> list[str]:
    """Independent check of every hard constraint on a finished timetable."""
    periods, errs = tt["periods"], []
    seen: dict[tuple, str] = {}
    lab_rooms = set((tt.get("cfg") or {}).get("lab_rooms") or [])
    for p in tt["placements"]:
        idxs = range(p["start"], p["start"] + p["len"])
        if p["start"] + p["len"] > len(periods) or not all(periods[i]["teach"] for i in idxs):
            errs.append(f"{p['course']} {p['section']}: placed outside teaching periods / across a break.")
            continue
        if p["kind"] == "lab" and lab_rooms and p["room"] not in lab_rooms:
            errs.append(f"{p['course']} lab is not in a lab room ({p['room']}).")
        for i in idxs:
            for kind, who in (("section", p["section"]), ("room", p["room"]), ("faculty", p["faculty"]),
                              ("faculty", p.get("assistant") or None)):
                if not who:
                    continue
                key = (kind, who, p["day"], i)
                if key in seen:
                    errs.append(f"{kind} clash: {who} on {p['day']} {periods[i]['label']} ({seen[key]} vs {p['course']})")
                seen[key] = p["course"]
    return errs


# ---------- views / exports ----------

def _cells(tt: dict, pick, label) -> pd.DataFrame:
    periods, days = tt["periods"], tt["days"]
    grid = {d: [""] * len(periods) for d in days}
    for p in tt["placements"]:
        if not pick(p):
            continue
        for k in range(p["len"]):
            txt = label(p) + (" (cont.)" if k else "")
            grid[p["day"]][p["start"] + k] = txt
    df = pd.DataFrame(grid, index=[f"{x['label']} {x['start']}-{x['end']}" if x["teach"] else f"Break {x['start']}-{x['end']}"
                                   for x in periods])
    return df


def section_grid(tt, section):
    return _cells(tt, lambda p: p["section"] == section,
                  lambda p: f"{p['course']}{' LAB' if p['kind'] == 'lab' else ''} | {p['faculty']} | {p['room']}")


def faculty_grid(tt, faculty):
    return _cells(tt, lambda p: faculty in (p["faculty"], p.get("assistant")),
                  lambda p: f"{p['course']}{' LAB' if p['kind'] == 'lab' else ''} | {p['section']} | {p['room']}")


def room_grid(tt, room):
    return _cells(tt, lambda p: p["room"] == room,
                  lambda p: f"{p['course']}{' LAB' if p['kind'] == 'lab' else ''} | {p['section']} | {p['faculty']}")


def names(tt) -> dict:
    pl = tt["placements"]
    return {"section": sorted({p["section"] for p in pl}),
            "faculty": sorted({p["faculty"] for p in pl} | {p["assistant"] for p in pl if p.get("assistant")}),
            "room": sorted({p["room"] for p in pl})}


def _color(course: str) -> str:
    h = int(hashlib.md5(course.encode()).hexdigest()[:6], 16) % 360 / 360
    r, g, b = colorsys.hls_to_rgb(h, 0.88, 0.55)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def grid_html(df: pd.DataFrame, caption: str = "") -> str:
    """Self-contained HTML table. Every user-supplied string is escaped."""
    rows = ["<table class='tt'>"]
    if caption:
        rows.append(f"<caption>{html.escape(caption)}</caption>")
    rows.append("<tr><th></th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>")
    for idx, r in df.iterrows():
        is_break = str(idx).startswith("Break")
        cells = []
        for v in r:
            v = str(v)
            if is_break:
                cells.append("<td class='brk'>break</td>")
            elif v:
                course = v.split(" ")[0]
                cells.append(f"<td style='background:{_color(course)}'>{html.escape(v).replace(' | ', '<br>')}</td>")
            else:
                cells.append("<td></td>")
        rows.append(f"<tr><th class='ph'>{html.escape(str(idx))}</th>{''.join(cells)}</tr>")
    rows.append("</table>")
    return "".join(rows)


CSS = ("body{font-family:Arial,sans-serif;margin:20px}table.tt{border-collapse:collapse;margin:14px 0;width:100%}"
       ".tt th,.tt td{border:1px solid #bbb;padding:5px 7px;font-size:12px;text-align:center;vertical-align:middle}"
       ".tt th{background:#f1f3f5}.tt .ph{white-space:nowrap}.tt .brk{background:#eee;color:#888}"
       "caption{font-weight:bold;text-align:left;padding:4px 0}h1{font-size:20px}")


def to_html(tt: dict, title: str) -> str:
    n = names(tt)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>"]
    for s in n["section"]:
        parts.append(grid_html(section_grid(tt, s), f"Section {s}"))
    for f in n["faculty"]:
        parts.append(grid_html(faculty_grid(tt, f), f"Faculty: {f}"))
    for r in n["room"]:
        parts.append(grid_html(room_grid(tt, r), f"Room: {r}"))
    parts.append("</body></html>")
    return "".join(parts)


def to_long_df(tt: dict) -> pd.DataFrame:
    rows = []
    for p in tt["placements"]:
        a, b = tt["periods"][p["start"]], tt["periods"][p["start"] + p["len"] - 1]
        rows.append({"day": p["day"], "from": a["start"], "to": b["end"], "section": p["section"], "course": p["course"],
                     "name": p["name"], "type": p["kind"], "faculty": p["faculty"], "lab_assistant": p.get("assistant", ""),
                     "room": p["room"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {d: i for i, d in enumerate(DAYS_ALL)}
    return df.sort_values(by=["section", "day", "from"], key=lambda c: c.map(order) if c.name == "day" else c).reset_index(drop=True)


def to_excel(tt: dict) -> bytes:
    buf = io.BytesIO()
    n = names(tt)
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        to_long_df(tt).to_excel(xw, sheet_name="All sessions", index=False)
        for s in n["section"]:
            section_grid(tt, s).to_excel(xw, sheet_name=f"Sec {s}"[:31])
        for f in n["faculty"]:
            faculty_grid(tt, f).to_excel(xw, sheet_name=f"Fac {f}"[:31])
        for r in n["room"]:
            room_grid(tt, r).to_excel(xw, sheet_name=f"Room {r}"[:31])
    return buf.getvalue()


# ---------- sharing ----------

def save(owner: str, title: str, tt: dict) -> str:
    code = secrets.token_urlsafe(6)
    db.insert(db.timetables, share_code=code, owner=owner, title=title, payload=json.dumps(tt), created_at=db.now())
    return code


def load(code: str) -> dict | None:
    r = db.rows(select(db.timetables).where(db.timetables.c.share_code == code))
    return {"title": r[0]["title"], "owner": r[0]["owner"], "tt": json.loads(r[0]["payload"]),
            "created_at": r[0]["created_at"]} if r else None


def list_saved(owner: str) -> list[dict]:
    return db.rows(select(db.timetables.c.share_code, db.timetables.c.title, db.timetables.c.created_at)
                   .where(db.timetables.c.owner == owner).order_by(db.timetables.c.id.desc()))


def unshare(code: str, owner: str) -> None:
    from sqlalchemy import delete
    db.execute(delete(db.timetables).where(db.timetables.c.share_code == code, db.timetables.c.owner == owner))


SAMPLE_COURSES = [
    {"course": "CSE1001", "name": "Data Structures", "faculty": "Dr. Rao", "section": "A", "theory_hours": 3, "lab_hours": 2, "lab_block": 2, "lab_assistant": ""},
    {"course": "CSE1002", "name": "Database Systems", "faculty": "Dr. Devi", "section": "A", "theory_hours": 3, "lab_hours": 2, "lab_block": 2, "lab_assistant": "Mr. Kiran"},
    {"course": "MAT1001", "name": "Linear Algebra", "faculty": "Dr. Sharma", "section": "A", "theory_hours": 4, "lab_hours": 0, "lab_block": 2, "lab_assistant": ""},
    {"course": "CSE1001", "name": "Data Structures", "faculty": "Dr. Rao", "section": "B", "theory_hours": 3, "lab_hours": 2, "lab_block": 2, "lab_assistant": ""},
    {"course": "CSE1002", "name": "Database Systems", "faculty": "Dr. Devi", "section": "B", "theory_hours": 3, "lab_hours": 2, "lab_block": 2, "lab_assistant": "Mr. Kiran"},
    {"course": "MAT1001", "name": "Linear Algebra", "faculty": "Dr. Sharma", "section": "B", "theory_hours": 4, "lab_hours": 0, "lab_block": 2, "lab_assistant": ""},
]
SAMPLE_UNAVAILABLE = [{"faculty": "Dr. Rao", "day": "Mon", "period": "P1"}, {"faculty": "Dr. Rao", "day": "Mon", "period": "P2"}]
