"""Transparent Bayesian shortlisting-odds model.

1. Company base rate: Beta posterior. theta ~ Beta(a0 + shortlisted, b0 + applicants - shortlisted),
   where (a0, b0) is a weak prior from config. Few past drives => wide uncertainty.
2. Student adjustment on the log-odds scale: logit(p) = logit(theta) + sum_i w_i * (x_i - ref_i) - penalty * backlogs,
   where x are the student's features and ref is the company's typical applicant.
3. Monte Carlo over theta gives the mean and a 90% credible interval.

The weights W are expert-set hyperparameters, NOT fitted to real data. This is a decision aid, not a prediction guarantee.
"""
from __future__ import annotations

import math

import numpy as np

from core.config import CONFIG_DIR, load_yaml

W = {"skill_match": 3.0, "cgpa": 0.9, "coding": 1.5, "projects": 1.0, "internship": 0.8, "comm": 0.6}
BACKLOG_PENALTY = 0.8
N_DRAWS = 4000


def load() -> dict:
    return load_yaml(CONFIG_DIR / "placement" / "companies.yaml")


def features(student: dict, company: dict) -> dict:
    req = company["required_skills"]
    owned = set(student.get("skills", []))
    total = sum(req.values()) or 1
    return {"skill_match": sum(w for s, w in req.items() if s in owned) / total,
            "cgpa": float(student.get("cgpa", 0)),
            "coding": min(1.0, math.log1p(max(0, student.get("problems", 0))) / math.log1p(500)),
            "projects": min(int(student.get("projects", 0)), 4) / 4,
            "internship": min(int(student.get("internships", 0)), 1),
            "comm": (float(student.get("comm", 3)) - 3) / 2}


def eligibility(student: dict, company: dict) -> list[str]:
    why = []
    if student.get("cgpa", 0) < company["min_cgpa"]:
        why.append(f"CGPA {student.get('cgpa', 0):.2f} is below the minimum {company['min_cgpa']}.")
    if student.get("backlogs", 0) > company["max_backlogs"]:
        why.append(f"{student.get('backlogs', 0)} active backlog(s); the company allows {company['max_backlogs']}.")
    return why


def posterior_params(company: dict, prior: dict) -> tuple[float, float]:
    k, r = prior["strength"], prior["rate"]
    h = company["history"]
    return k * r + h["shortlisted"], k * (1 - r) + (h["applicants"] - h["shortlisted"])


def _shift(student: dict, company: dict) -> float:
    f, ref = features(student, company), company["typical_applicant"]
    d = sum(W[k] * (f[k] - ref[k]) for k in W)
    return d - BACKLOG_PENALTY * student.get("backlogs", 0)


def _draws(a: float, b: float, seed: int = 0) -> np.ndarray:
    return np.clip(np.random.default_rng(seed).beta(a, b, N_DRAWS), 1e-6, 1 - 1e-6)


def _p(theta: np.ndarray, shift: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.log(theta / (1 - theta)) + shift)))


def estimate(student: dict, company: dict, prior: dict, seed: int = 0) -> dict:
    a, b = posterior_params(company, prior)
    why = eligibility(student, company)
    base_mean = a / (a + b)
    if why:
        return {"eligible": False, "reasons": why, "mean": 0.0, "low": 0.0, "high": 0.0, "base_rate": base_mean, "a": a, "b": b}
    p = _p(_draws(a, b, seed), _shift(student, company))
    return {"eligible": True, "reasons": [], "mean": float(p.mean()), "low": float(np.percentile(p, 5)),
            "high": float(np.percentile(p, 95)), "base_rate": base_mean, "a": a, "b": b,
            "evidence": company["history"]["applicants"]}


def improvements(student: dict, company: dict, prior: dict, seed: int = 0) -> list[dict]:
    """Rank 'what if' changes by the gain in shortlisting probability (same random draws => stable differences)."""
    base = estimate(student, company, prior, seed)
    if not base["eligible"]:
        return []
    out = []

    def trial(label, **changes):
        s2 = {**student, **changes}
        e = estimate(s2, company, prior, seed)
        gain = (e["mean"] - base["mean"]) * 100
        if gain > 0.05:
            out.append({"action": label, "gain_pp": round(gain, 1), "new_p": round(e["mean"] * 100, 1)})

    owned = set(student.get("skills", []))
    for s, w in sorted(company["required_skills"].items(), key=lambda kv: -kv[1]):
        if s not in owned:
            trial(f"Learn {s} (weight {w})", skills=sorted(owned | {s}))
    trial("Solve 100 more practice problems", problems=student.get("problems", 0) + 100)
    trial("Add one more project", projects=min(4, student.get("projects", 0) + 1))
    trial("Do an internship", internships=1)
    trial("Raise CGPA by 0.3", cgpa=min(10.0, student.get("cgpa", 0) + 0.3))
    trial("Improve communication by one level", comm=min(5, student.get("comm", 3) + 1))
    if student.get("backlogs", 0) > 0:
        trial("Clear one backlog", backlogs=student["backlogs"] - 1)
    return sorted(out, key=lambda x: -x["gain_pp"])
