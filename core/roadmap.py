"""Goal-based course roadmap: weighted skill coverage with prerequisite closure, scheduled by semester."""
from __future__ import annotations

from core.config import CONFIG_DIR, load_yaml


def load_catalog() -> dict:
    return {c["id"]: c for c in load_yaml(CONFIG_DIR / "roadmaps" / "courses.yaml")["courses"]}


def load_goals() -> dict:
    return load_yaml(CONFIG_DIR / "roadmaps" / "goals.yaml")["goals"]


def prereq_closure(cid: str, catalog: dict, done: set | None = None) -> set:
    done = done or set()
    out, stack = set(), [cid]
    while stack:
        c = stack.pop()
        if c in out or c in done:
            continue
        out.add(c)
        stack.extend(catalog[c].get("prereqs", []))
    return out


def plan(goal: dict, catalog: dict, completed: set | None = None, credit_cap: int = 20, max_semesters: int = 8) -> dict:
    completed = set(completed or [])
    need = dict(goal["skills"])
    have = {s for c in completed for s in catalog[c].get("skills", [])}
    missing = {s: w for s, w in need.items() if s not in have}
    chosen: set[str] = set()
    while True:
        covered = have | {s for c in chosen for s in catalog[c].get("skills", [])}
        gaps = {s: w for s, w in need.items() if s not in covered}
        best, best_ratio = None, 0.0
        for cid in catalog:
            if cid in completed or cid in chosen:
                continue
            gain = sum(gaps.get(s, 0) for s in set(catalog[cid].get("skills", [])))
            if gain <= 0:
                continue
            extra = prereq_closure(cid, catalog, completed | chosen)
            cost = sum(catalog[x]["credits"] for x in extra)
            ratio = gain / cost
            if ratio > best_ratio:
                best, best_ratio = (cid, extra), ratio
        if not best:
            break
        chosen |= best[1]
    # schedule: only courses whose prereqs are finished in an EARLIER semester
    remaining, done, semesters = set(chosen), set(completed), []
    dependents = {c: sum(1 for d in remaining if c in catalog[d].get("prereqs", [])) for c in remaining}
    while remaining and len(semesters) < max_semesters:
        avail = sorted((c for c in remaining if all(p in done for p in catalog[c].get("prereqs", []))),
                       key=lambda c: (-dependents.get(c, 0), catalog[c]["credits"], c))
        sem, credits = [], 0
        for c in avail:
            if credits + catalog[c]["credits"] <= credit_cap:
                sem.append(c)
                credits += catalog[c]["credits"]
        if not sem:
            break
        semesters.append({"courses": sem, "credits": credits})
        done |= set(sem)
        remaining -= set(sem)
    covered = have | {s for c in chosen for s in catalog[c].get("skills", [])}
    total_w = sum(need.values()) or 1
    return {"semesters": semesters, "unscheduled": sorted(remaining),
            "covered_pct": round(100 * sum(w for s, w in need.items() if s in covered) / total_w, 1),
            "uncovered": {s: w for s, w in need.items() if s not in covered},
            "total_credits": sum(x["credits"] for x in semesters), "chosen": sorted(chosen),
            "already_have": sorted(s for s in need if s in have), "missing_at_start": missing}


def to_dot(plan_: dict, catalog: dict, completed: set) -> str:
    sem_of = {c: i + 1 for i, s in enumerate(plan_["semesters"]) for c in s["courses"]}
    nodes = set(sem_of) | {p for c in sem_of for p in catalog[c].get("prereqs", []) if p in completed}
    lines = ["digraph G { rankdir=LR; node [shape=box, style=filled, fontsize=11];"]
    for c in sorted(nodes):
        color = "#d3d3d3" if c in completed else "#b7e4c7"
        tag = "done" if c in completed else f"Sem {sem_of.get(c, '?')}"
        lines.append(f'"{c}" [label="{c}\\n{catalog[c]["name"]}\\n{tag}", fillcolor="{color}"];')
    for c in sem_of:
        for p in catalog[c].get("prereqs", []):
            if p in nodes:
                lines.append(f'"{p}" -> "{c}";')
    lines.append("}")
    return "\n".join(lines)


def to_markdown(goal_name: str, plan_: dict, catalog: dict) -> str:
    out = [f"# Roadmap: {goal_name}", f"Skill coverage after this plan: {plan_['covered_pct']}%", ""]
    for i, s in enumerate(plan_["semesters"], 1):
        out.append(f"## Semester {i} ({s['credits']} credits)")
        out += [f"- {c}: {catalog[c]['name']} ({catalog[c]['credits']} cr)" for c in s["courses"]]
        out.append("")
    if plan_["uncovered"]:
        out.append("## Learn outside the curriculum")
        out += [f"- {s}" for s in plan_["uncovered"]]
    return "\n".join(out)
