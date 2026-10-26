import json

import pandas as pd
import streamlit as st

from core import audit, config, db, llm, rubric as rb

MANIFEST = {"id": "feedback", "label": "Answer-Script Feedback", "icon": "📝",
            "roles": ["faculty"], "order": 40}


def render(user):
    st.header("📝 Answer-Script Feedback Assistant")
    st.caption("The AI only SUGGESTS marks per rubric point. You review, edit and approve. "
               "Only approved marks are saved and exported.")
    files = config.list_yaml("rubrics")
    if not files:
        st.error("No rubric files in config/rubrics.")
        return
    path = st.selectbox("Rubric file", files, format_func=lambda p: p.name)
    rubric = config.load_yaml(path)
    errs = rb.validate_rubric(rubric)
    if errs:
        st.error("Rubric problems: " + "; ".join(errs))
        return
    q = st.selectbox("Question", rubric["questions"], format_func=lambda x: f"{x['id']} ({x['max_marks']} marks): {x['text'][:70]}")
    with st.expander("Rubric points"):
        st.table(pd.DataFrame(q["points"]))
    student_ref = st.text_input("Student reference (use roll no./code, not the name)", value="STU-001")
    mode = st.radio("Answer input", ["Scanned image(s)", "Typed text (also works without an LLM key for the UI)"], horizontal=True)
    imgs, typed = [], ""
    if mode.startswith("Scanned"):
        ups = st.file_uploader("Answer script image(s)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="fb_up")
        imgs = [(u.getvalue(), u.type or "image/png") for u in ups or []]
    else:
        typed = st.text_area("Paste the student's answer", height=160)
    ready = bool(imgs or typed.strip())
    if not llm.available():
        st.warning("Evaluation needs the Gemma 4 model. Set GEMINI_API_KEY (or LLM_BACKEND=ollama).")
    if st.button("Suggest marks", type="primary", disabled=not (ready and llm.available())):
        prompt = rb.build_prompt(q, bool(imgs))
        if typed:
            prompt += f"\n\nSTUDENT ANSWER:\n<<<\n{typed}\n>>>"
        with st.spinner("Evaluating against the rubric…"):
            try:
                raw = llm.generate_json(prompt, images=imgs or None)
                st.session_state["fb_result"] = {"q": q["id"], "ref": student_ref, "course": rubric["course"],
                                                 "res": rb.normalize_result(q, raw)}
                audit.log(user, "feedback", f"suggest {q['id']} for {student_ref}")
            except llm.LLMError as e:
                st.error(f"Evaluation failed: {e}")
    cur = st.session_state.get("fb_result")
    if cur and cur["q"] == q["id"]:
        res = cur["res"]
        st.subheader("Review and approve")
        with st.expander("Transcript", expanded=False):
            st.write(res["transcript"] or "(none)")
        df = pd.DataFrame(res["points"])[["id", "criterion", "max", "suggested", "final", "status", "confidence", "evidence"]]
        edited = st.data_editor(df, use_container_width=True, hide_index=True, key=f"ed_{cur['ref']}_{cur['q']}",
                                disabled=["id", "criterion", "max", "suggested", "status", "confidence", "evidence"],
                                column_config={"final": st.column_config.NumberColumn("final (edit)", min_value=0.0, step=0.5)})
        edited["final"] = [min(max(float(f), 0.0), float(m)) for f, m in zip(edited["final"], edited["max"])]
        total = float(edited["final"].sum())
        low = [r["id"] for r in res["points"] if r["needs_review"]]
        st.metric("Total (after your edits)", f"{total:g} / {res['max_marks']:g}")
        if low:
            st.warning(f"Low confidence on: {', '.join(low)}. Please check these manually.")
        fb = st.text_area("Feedback for the student (editable)", value=res["feedback"])
        if st.button("✅ Approve and save"):
            db.insert(db.evaluations, faculty=user["username"], course=cur["course"], question_id=cur["q"],
                      student_ref=cur["ref"], transcript=res["transcript"],
                      result_json=json.dumps({"points": edited.to_dict("records"), "feedback": fb}),
                      final_marks=total, max_marks=res["max_marks"], status="approved", created_at=db.now())
            audit.log(user, "feedback", f"approve {cur['q']} {cur['ref']} = {total:g}")
            st.success("Saved.")
    from sqlalchemy import select
    saved = db.rows(select(db.evaluations).where(db.evaluations.c.faculty == user["username"]).order_by(db.evaluations.c.id.desc()))
    if saved:
        st.subheader("Approved marks")
        out = pd.DataFrame(saved)[["created_at", "course", "question_id", "student_ref", "final_marks", "max_marks", "status"]]
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", out.to_csv(index=False).encode(), "approved_marks.csv", "text/csv")
