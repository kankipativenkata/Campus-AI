import csv

from core import config, rag


def test_seed_loaded_documents():
    from sqlalchemy import select
    from core import db
    titles = {r["title"] for r in db.rows(select(db.documents))}
    assert {"Academic Regulations", "Hostel Rules", "Fee Structure 2026", "Academic Calendar Fall 2026", "Placement FAQ",
            "Club Directory", "Merit Scholarship Policy", "Faculty Leave Policy"} <= titles


def test_chunking_respects_size_and_overlap():
    para = "Sentence one is here. " * 40
    chunks = rag.chunk_text([(1, "\n\n".join([para] * 6))], max_chars=500, overlap=50)
    assert len(chunks) > 3
    assert all(len(t) <= 600 for _, t in chunks)


def test_stemming_and_stopwords():
    assert rag.tokenize("What are the hostel rules?") == ["hostel", "rule"]
    assert rag.stem("searching") == "search"


def test_attendance_retrieval_top_hit():
    hits = rag.search("What is the minimum attendance for exam eligibility?", "student")
    assert hits and "75%" in hits[0]["text"]


def test_docx_and_pdf_content_is_searchable():
    assert "Wednesday" in rag.search("Which day does the Math Club meet?", "student")[0]["text"]
    assert "50%" in rag.search("scholarship waiver for toppers", "student")[0]["text"]


def test_role_filter_blocks_faculty_only_docs():
    q = "How many casual leaves do faculty get?"
    assert rag.search(q, "student") == []
    assert "12 days" in rag.search(q, "faculty")[0]["text"]


def test_out_of_scope_questions_are_refused():
    assert rag.search("Who won the football world cup in 2022?", "student") == []
    res = rag.answer("What is the capital of France?", "student")
    assert res["found"] is False and res["sources"] == []


def test_offline_answer_is_extractive_with_sources():
    res = rag.answer("When is breakfast served in the mess?", "student")
    assert res["found"] and "7:30" in res["answer"] and res["sources"]
    assert "extractive" in res["mode"]


def test_archived_documents_are_not_searched():
    from sqlalchemy import select
    from core import db
    d = [x for x in db.rows(select(db.documents)) if x["title"] == "Hostel Rules"][0]
    rag.set_status(d["id"], "archived")
    assert rag.search("What time do hostel gates close?", "student") == []
    rag.set_status(d["id"], "active")
    assert rag.search("What time do hostel gates close?", "student")


def test_new_version_archives_old():
    r = rag.ingest_document("x.txt", b"Library fine is Rs. 5 per day for late return.", "Library Rules", version="1")
    r2 = rag.ingest_document("x.txt", b"Library fine is Rs. 10 per day for late return.", "Library Rules", version="2")
    hits = rag.search("library fine per day", "student")
    assert "Rs. 10" in hits[0]["text"] and all("Rs. 5 " not in h["text"] for h in hits)
    assert r["doc_id"] != r2["doc_id"]


def test_bundled_question_set_passes_without_embeddings():
    path = config.TEST_DATA / "rag_test_questions.csv"
    failures = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        if row["requires_embeddings"] == "yes":
            continue
        res = rag.answer(row["question"], row["role"])
        if row["should_answer"] == "yes":
            blob = res["answer"] + " ".join(s["snippet"] for s in res["sources"])
            if not (res["found"] and row["expected_keyword"].lower() in blob.lower()):
                failures.append(row["question"])
        elif res["found"]:
            failures.append(row["question"])
    assert not failures, failures
