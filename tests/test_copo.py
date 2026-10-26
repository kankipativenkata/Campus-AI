from core import config, copo

PAPER = (config.TEST_DATA / "question_papers" / "question_paper_sample.txt").read_text(encoding="utf-8")
OC = config.load_yaml(config.CONFIG_DIR / "outcomes" / "CSE1001_outcomes.yaml")


def _tagged():
    qs = copo.parse_questions_regex(PAPER)
    for q in qs:
        q["co"], q["bloom"] = copo.co_by_overlap(q["text"], OC["cos"]), copo.bloom_by_verb(q["text"])
    return qs


def test_parse_questions_and_marks():
    qs = copo.parse_questions_regex(PAPER)
    assert len(qs) == 9
    assert sum(q["marks"] for q in qs) == 54
    assert qs[0]["text"] == "Define a stack."


def test_bloom_verbs():
    assert copo.bloom_by_verb("Define a stack.") == "Remember"
    assert copo.bloom_by_verb("Explain the push operation") == "Understand"
    assert copo.bloom_by_verb("Design a hash table") == "Create"
    assert copo.bloom_by_verb("Justify your choice") == "Evaluate"


def test_co_mapping_by_overlap():
    qs = {q["no"]: q for q in _tagged()}
    assert qs["1"]["co"] == "CO1"
    assert qs["7"]["co"] == "CO2"
    assert qs["3"]["co"] == "CO3"
    assert qs["9"]["co"] == "CO4"


def test_coverage_sums_to_100_and_warns():
    cov = copo.coverage(_tagged(), OC["cos"], OC["bloom_targets"])
    assert abs(sum(cov["pct_by_co"].values()) - 100) < 0.5
    assert abs(sum(cov["pct_by_bloom"].values()) - 100) < 0.5
    cov2 = copo.coverage([{"no": "1", "text": "x", "marks": 5, "co": "CO1", "bloom": "Remember"}], OC["cos"], OC["bloom_targets"])
    assert any("CO2 has zero coverage" in w for w in cov2["warnings"])
    assert any("Remember is 100.0%" in w for w in cov2["warnings"])


def test_po_weights():
    w = copo.po_weights({"CO1": 100.0}, {"CO1": {"PO1": 3, "PO2": 2}})
    assert w == {"PO1": 100.0, "PO2": 66.7}


def test_repeat_detection_finds_known_repeats():
    past = [{"paper_title": "DS 2025", "question_no": "1", "question_text": "Define a stack."},
            {"paper_title": "DS 2025", "question_no": "3", "question_text": "Differentiate between a singly linked list and a doubly linked list."}]
    reps = copo.find_repeats(copo.parse_questions_regex(PAPER), past)
    assert {r["question"] for r in reps} == {"1", "5"}
