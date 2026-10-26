"""Exercises the LLM-dependent code paths with a fake model (no network, no key)."""
import hashlib
import json

import numpy as np
import pytest

from core import config, llm, rag, rubric as rb


def _fake_embed(texts, task="RETRIEVAL_DOCUMENT"):
    out = []
    for t in texts:
        v = np.zeros(128)
        for w in rag.tokenize(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
        out.append(v.tolist())
    return out


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "embed", _fake_embed)
    calls = {}

    def gen(prompt, images=None, system=None, temperature=0.2):
        calls["prompt"], calls["system"] = prompt, system
        return "Attendance must be at least 75% [1]."
    monkeypatch.setattr(llm, "generate", gen)
    return calls


def test_vector_hybrid_search_and_llm_answer(fake_llm):
    rag.ingest_document("a.txt", config.SAMPLE_DOCS.joinpath("academic_regulations.txt").read_bytes(),
                        "Academic Regulations", "academic", "all", "2.0", "2026-07-01")
    res = rag.answer("What is the minimum attendance for exam eligibility?", "student")
    assert res["found"] and res["mode"] == "llm" and "75%" in res["answer"]
    assert "CONTEXT" in fake_llm["prompt"] and "NOT_FOUND" in fake_llm["system"]


def test_llm_not_found_is_honoured(fake_llm, monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "NOT_FOUND")
    res = rag.answer("When is breakfast served in the mess?", "student")
    assert res["found"] is False


def test_llm_error_falls_back_to_extractive(fake_llm, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMError("quota exceeded")
    monkeypatch.setattr(llm, "generate", boom)
    res = rag.answer("When is breakfast served in the mess?", "student")
    assert res["found"] and "quota exceeded" in res["mode"] and "7:30" in res["answer"]


def test_generate_json_retries_once(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    replies = iter(["not json at all", '{"ok": true}'])
    monkeypatch.setattr(llm, "generate", lambda *a, **k: next(replies))
    assert llm.generate_json("x") == {"ok": True}


def test_feedback_prompt_includes_rubric_and_normalizes_model_output():
    q = config.load_yaml(config.CONFIG_DIR / "rubrics" / "CSE1001_data_structures.yaml")["questions"][0]
    prompt = rb.build_prompt(q, has_image=True)
    assert "LIFO" in prompt and "p5" in prompt and "attached image" in prompt
    raw = json.loads('{"transcript":"t","points":[{"id":"p1","status":"covered","suggested_marks":2,"confidence":0.95}],"feedback":"f"}')
    res = rb.normalize_result(q, raw)
    assert res["total"] == 2 and sum(p["needs_review"] for p in res["points"]) == 4
