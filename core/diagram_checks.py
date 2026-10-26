"""Deterministic rule checks on a diagram structure extracted by the vision model.
Structure: {"nodes":[{"id","label","type"}], "edges":[{"from","to","label"}]}"""
from __future__ import annotations

from collections import defaultdict, deque

DIAGRAM_TYPES = ["Flowchart", "ER diagram", "UML class diagram", "Other"]


def _f(sev: str, msg: str) -> dict:
    return {"severity": sev, "message": msg}


def check_structure(dtype: str, s: dict) -> list[dict]:
    nodes = s.get("nodes") or []
    edges = s.get("edges") or []
    ids = {n["id"] for n in nodes}
    label = {n["id"]: n.get("label", n["id"]) for n in nodes}
    typ = {n["id"]: str(n.get("type", "")).lower() for n in nodes}
    out_e, in_e = defaultdict(list), defaultdict(list)
    f: list[dict] = []
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            f.append(_f("error", f"Arrow {e['from']} -> {e['to']} points to a missing shape."))
            continue
        out_e[e["from"]].append(e)
        in_e[e["to"]].append(e)
    for n in nodes:
        if not out_e[n["id"]] and not in_e[n["id"]]:
            f.append(_f("error", f"Shape '{label[n['id']]}' is not connected to anything."))
    labels = [str(n.get("label", "")).strip().lower() for n in nodes]
    for l in set(labels):
        if l and labels.count(l) > 1 and dtype != "Flowchart":
            f.append(_f("warning", f"Duplicate label '{l}'."))

    if dtype == "Flowchart":
        starts = [i for i in ids if typ[i] in ("start", "terminator", "terminal")
                  and any(w in label[i].lower() for w in ("start", "begin"))]
        ends = [i for i in ids if typ[i] in ("end", "terminator", "terminal")
                and any(w in label[i].lower() for w in ("end", "stop"))]
        if not starts:
            f.append(_f("error", "No Start terminator found."))
        if not ends:
            f.append(_f("error", "No End terminator found."))
        for i in ids:
            if typ[i] == "decision":
                outs = out_e[i]
                if len(outs) < 2:
                    f.append(_f("error", f"Decision '{label[i]}' has {len(outs)} outgoing arrow(s); needs at least 2."))
                elif any(not str(e.get("label", "")).strip() for e in outs):
                    f.append(_f("warning", f"Branches of decision '{label[i]}' should be labelled (Yes/No)."))
        for i in ids:
            if i not in ends and in_e[i] is not None and out_e[i] == [] and in_e[i]:
                f.append(_f("error", f"'{label[i]}' is a dead end (no outgoing arrow and not an End)."))
        if starts:
            seen, q = set(starts), deque(starts)
            while q:
                cur = q.popleft()
                for e in out_e[cur]:
                    if e["to"] not in seen:
                        seen.add(e["to"])
                        q.append(e["to"])
            for i in ids - seen:
                if in_e[i] or out_e[i]:
                    f.append(_f("error", f"'{label[i]}' cannot be reached from Start."))
    elif dtype == "ER diagram":
        ents = [i for i in ids if typ[i] == "entity"]
        for i in ents:
            nb = {e["to"] for e in out_e[i]} | {e["from"] for e in in_e[i]}
            attrs = [x for x in nb if typ.get(x) in ("attribute", "key")]
            if not attrs:
                f.append(_f("warning", f"Entity '{label[i]}' has no attributes."))
            elif not any(typ[x] == "key" or "pk" in label[x].lower() or "(key)" in label[x].lower() or label[x].lower().endswith("id")
                         for x in attrs):
                f.append(_f("warning", f"Entity '{label[i]}' has no identifiable key attribute."))
        for i in ids:
            if typ[i] == "relationship":
                nb = {e["to"] for e in out_e[i]} | {e["from"] for e in in_e[i]}
                if len([x for x in nb if typ.get(x) == "entity"]) < 2:
                    f.append(_f("error", f"Relationship '{label[i]}' must connect at least 2 entities."))
                touching = out_e[i] + in_e[i]
                if not any(str(e.get("label", "")).strip() for e in touching):
                    f.append(_f("warning", f"Relationship '{label[i]}' has no cardinality labels (1, N, M)."))
    elif dtype == "UML class diagram":
        for i in ids:
            if not str(label[i]).strip():
                f.append(_f("error", "A class has no name."))
        graph = defaultdict(list)
        for e in edges:
            if str(e.get("label", "")).lower() in ("extends", "inherits", "generalization", "is-a"):
                graph[e["from"]].append(e["to"])
        state: dict = {}

        def dfs(u):
            state[u] = 1
            for v in graph[u]:
                if state.get(v) == 1 or (state.get(v) is None and dfs(v)):
                    return True
            state[u] = 2
            return False
        if any(state.get(u) is None and dfs(u) for u in list(graph)):
            f.append(_f("error", "Inheritance cycle detected."))
    return f


def score(findings: list[dict]) -> int:
    errs = sum(1 for x in findings if x["severity"] == "error")
    warns = sum(1 for x in findings if x["severity"] == "warning")
    return max(0, 100 - 20 * errs - 5 * warns)


def to_dot(s: dict) -> str:
    shape = {"start": "ellipse", "end": "ellipse", "terminator": "ellipse", "decision": "diamond",
             "entity": "box", "relationship": "diamond", "attribute": "ellipse", "key": "ellipse",
             "class": "box", "process": "box", "io": "parallelogram"}
    esc = lambda x: str(x).replace('"', "'")  # noqa: E731
    lines = ["digraph G { rankdir=TB; node [fontsize=11];"]
    for n in s.get("nodes", []):
        lines.append(f'"{esc(n["id"])}" [label="{esc(n.get("label", n["id"]))}" '
                     f'shape={shape.get(str(n.get("type", "")).lower(), "box")}];')
    for e in s.get("edges", []):
        lines.append(f'"{esc(e["from"])}" -> "{esc(e["to"])}" [label="{esc(e.get("label", ""))}"];')
    lines.append("}")
    return "\n".join(lines)
