"""Scores the RAG assistant on data/test_data/rag_test_questions.csv.
Usage:  python scripts/eval_rag.py            (uses your configured LLM/embeddings if a key is set)
        LLM_BACKEND=none python scripts/eval_rag.py   (keyword-only, offline)"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, llm, rag, seed  # noqa: E402

seed.seed_all()
print(f"LLM: {llm.status()}\n")
rows = list(csv.DictReader(open(config.TEST_DATA / "rag_test_questions.csv", encoding="utf-8")))
ok = 0
for r in rows:
    res = rag.answer(r["question"], r["role"])
    blob = (res["answer"] + " " + " ".join(s["snippet"] for s in res["sources"])).lower()
    if r["should_answer"] == "yes":
        passed = res["found"] and r["expected_keyword"].lower() in blob
    else:
        passed = not res["found"]
    ok += passed
    note = " (needs embeddings)" if r["requires_embeddings"] == "yes" and not passed else ""
    print(f"{'PASS' if passed else 'FAIL'}  [{r['role']:7}] {r['question'][:70]}{note}")
print(f"\nScore: {ok}/{len(rows)} = {100 * ok / len(rows):.0f}%")
