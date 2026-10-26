import io
import json

import pandas as pd
import streamlit as st
from sqlalchemy import select

from core import audit, config, copo, db, llm, parsing, rag
import numpy as np

MANIFEST = {"id": "copo", "label": "CO-PO Mapper & Paper Auditor", "icon": "🎯",
            "roles": ["faculty"], "order": 50}


def _extract_questions(text, prompts):
    if llm.available():
        try:
            out = llm.generate_json(prompts["extract"] + "\n\nPAPER:\n<<<\n" + text + "\n>>>")
            qs = out.get("questions", out) if isinstance(out, dict) else out
            return [{"no": str(q.get("no")), "text": str(q.get("text")), "marks": float(q.get("marks", 1) or 1)} for q in qs], "AI"
        except Exception:  # noqa: BLE001
            pass
    return copo.parse_questions_regex(text), "rule-based"


def _tag(questions, cos, prompts):
    if llm.available():
        try:
            out = llm.generate_json(prompts["tag"].format(
                cos="\n".join(f"{c['id']}: {c['text']}" for c in cos),
                questions="\n".join(f"{q['no']}. {q['text']}" for q in questions)))
            tags = {str(t["no"]): t for t in out.get("tags", [])}
            valid = {c["id"] for c in cos}
            for q in questions:
                t = tags.get(q["no"], {})
                q["co"] = t.get("co") if t.get("co") in valid else copo.co_by_overlap(q["text"], cos)
                q["bloom"] = t.get("bloom") if t.get("bloom") in copo.BLOOM_ORDER else copo.bloom_by_verb(q["text"])
                q["reason"] = t.get("reason", "")
            return "AI"
        except Exception:  # noqa: BLE001
            pass
    for q in questions:
        q["co"], q["bloom"], q["reason"] = copo.co_by_overlap(q["text"], cos), copo.bloom_by_verb(q["text"]), "rule-based"
    return "rule-based"


def render(user):
    st.header("🎯 CO-PO Mapper & Question Paper Auditor")
    st.caption("Tags each question with a Course Outcome and Bloom's level, checks coverage against targets, "
               "flags repeats from past papers, and exports the CO-PO table. Works without an LLM using rule-based tagging.")
    prompts = config.load_yaml(config.CONFIG_DIR / "prompts" / "question_tagging.yaml")
    files = config.list_yaml("outcomes")
    path = st.selectbox("Course outcomes file", files, format_func=lambda p: p.name)
    oc = config.load_yaml(path)
    with st.expander("Course outcomes in use"):
        st.table(pd.DataFrame(oc["cos"]))
    src = st.radio("Question paper source", ["Upload file (txt/pdf/docx/image)", "Paste text", "Use bundled sample"], horizontal=True)
    text = ""
    if src.startswith("Upload"):
        f = st.file_uploader("Question paper", type=[e.strip(".") for e in parsing.SUPPORTED], key="cp_up")
        if f:
            try:
                text = "\n".join(t for _, t in parsing.extract_pages(f.name, f.getvalue()))
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    elif src.startswith("Paste"):
        text = st.text_area("Paste paper text", height=200)
    else:
        text = (config.TEST_DATA / "question_papers" / "question_paper_sample.txt").read_text(encoding="utf-8")
        st.code(text, language="text")
    title = st.text_input("Paper title", value="Data Structures 2026")
    if st.button("Analyze paper", type="primary", disabled=not text.strip()):
        with st.spinner("Analyzing…"):
            qs, how1 = _extract_questions(text, prompts)
            if not qs:
                st.error("No questions detected. Use numbering like 'Q1. text (5 marks)'.")
                return
            how2 = _tag(qs, oc["cos"], prompts)
            cov = copo.coverage(qs, oc["cos"], oc.get("bloom_targets"))
            past = db.rows(select(db.past_questions))
            q_vecs = llm.embed([q["text"] for q in qs], "SEMANTIC_SIMILARITY") if past and llm.available() else None
            past_vecs = [json.loads(p["embedding"]) if p["embedding"] else None for p in past] if q_vecs else None
            reps = copo.find_repeats(qs, past, q_vecs, past_vecs)
            st.session_state["cp"] = {"qs": qs, "cov": cov, "reps": reps, "title": title, "how": f"extraction: {how1}, tagging: {how2}",
                                      "po": copo.po_weights(cov["pct_by_co"], oc.get("matrix", {}))}
            db.insert(db.paper_audits, faculty=user["username"], title=title, questions_json=json.dumps(qs),
                      coverage_json=json.dumps(cov), created_at=db.now())
            audit.log(user, "copo", f"analyze {title}")
    r = st.session_state.get("cp")
    if not r:
        return
    st.caption(f"Method: {r['how']}. Review the tags; AI and rule-based tagging can both be wrong.")
    qdf = pd.DataFrame(r["qs"])[["no", "text", "marks", "co", "bloom", "reason"]]
    st.subheader("Question tags")
    st.dataframe(qdf, use_container_width=True, hide_index=True)
    cov = r["cov"]
    for w in cov["warnings"]:
        st.warning(w)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Marks by CO (%)")
        st.bar_chart(pd.Series(cov["pct_by_co"]))
    with c2:
        st.subheader("Marks by Bloom's level (%)")
        st.bar_chart(pd.Series(cov["pct_by_bloom"]))
    st.subheader("PO weights (%)")
    st.caption("PO weight = sum over COs of (CO share of marks × mapping level ÷ 3).")
    st.dataframe(pd.DataFrame([r["po"]]), hide_index=True)
    st.subheader("Repeated questions")
    if r["reps"]:
        st.dataframe(pd.DataFrame(r["reps"]), use_container_width=True, hide_index=True)
    else:
        st.success("No repeats found against stored past papers.")
    if st.button("Add this paper to history (for future repeat checks)"):
        vecs = llm.embed([q["text"] for q in r["qs"]], "SEMANTIC_SIMILARITY") if llm.available() else None
        for i, q in enumerate(r["qs"]):
            db.insert(db.past_questions, paper_title=r["title"], year=2026, question_no=q["no"], question_text=q["text"],
                      marks=q["marks"], embedding=json.dumps(vecs[i]) if vecs else None)
        st.success("Added.")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        qdf.to_excel(xw, sheet_name="Questions", index=False)
        pd.DataFrame({"CO": list(cov["pct_by_co"]), "Percent": list(cov["pct_by_co"].values())}).to_excel(xw, sheet_name="CO coverage", index=False)
        pd.DataFrame({"Bloom": list(cov["pct_by_bloom"]), "Percent": list(cov["pct_by_bloom"].values())}).to_excel(xw, sheet_name="Bloom", index=False)
        pd.DataFrame([r["po"]]).to_excel(xw, sheet_name="PO weights", index=False)
        pd.DataFrame(r["reps"]).to_excel(xw, sheet_name="Repeats", index=False)
    st.download_button("Download Excel report", buf.getvalue(), "co_po_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download questions CSV", qdf.to_csv(index=False).encode(), "questions_tagged.csv", "text/csv")
