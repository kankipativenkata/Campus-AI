import json

import pytest

from core import config, diagram_checks as dc, llm, rubric as rb

R = config.load_yaml(config.CONFIG_DIR / "rubrics" / "CSE1001_data_structures.yaml")


def test_bundled_rubric_is_valid():
    assert rb.validate_rubric(R) == []


def test_rubric_validation_catches_bad_sum():
    bad = {"course": "X", "questions": [{"id": "Q1", "max_marks": 10, "points": [{"id": "a", "text": "t", "marks": 3}]}]}
    assert any("sum" in e for e in rb.validate_rubric(bad))


def test_model_marks_are_clamped_and_missing_points_flagged():
    q = R["questions"][0]
    raw = {"points": [{"id": "p1", "status": "covered", "suggested_marks": 99, "confidence": 0.9},
                      {"id": "p2", "suggested_marks": -4, "confidence": 5}], "feedback": "ok"}
    res = rb.normalize_result(q, raw)
    by = {p["id"]: p for p in res["points"]}
    assert by["p1"]["suggested"] == 2 and by["p2"]["suggested"] == 0 and by["p2"]["confidence"] == 1.0
    assert by["p3"]["needs_review"] is True and res["total"] == 2


def _load(name):
    return json.loads((config.TEST_DATA / "diagrams" / name).read_text())


def test_correct_flowchart_has_no_findings():
    assert dc.check_structure("Flowchart", _load("flowchart_correct.json")) == []


def test_broken_flowchart_is_caught():
    msgs = " | ".join(f["message"] for f in dc.check_structure("Flowchart", _load("flowchart_broken.json")))
    assert "No End" in msgs and "outgoing" in msgs and "not connected" in msgs
    assert dc.score(dc.check_structure("Flowchart", _load("flowchart_broken.json"))) < 50


def test_er_diagram_ok_and_relationship_error():
    assert dc.check_structure("ER diagram", _load("er_diagram.json")) == []
    s = _load("er_diagram.json")
    s["edges"] = [e for e in s["edges"] if e["to"] != "e2"]
    assert any("at least 2 entities" in f["message"] for f in dc.check_structure("ER diagram", s))


def test_dot_output():
    assert dc.to_dot(_load("flowchart_correct.json")).startswith("digraph")


def test_extract_json_variants():
    assert llm.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('Sure! Here it is: {"a": [1,2]} hope it helps') == {"a": [1, 2]}
    with pytest.raises(llm.LLMError):
        llm.extract_json("no json here")


def test_no_llm_configured_fails_gracefully():
    assert llm.available() is False
    with pytest.raises(llm.LLMError):
        llm.generate("hi")
    assert llm.embed(["x"]) is None
