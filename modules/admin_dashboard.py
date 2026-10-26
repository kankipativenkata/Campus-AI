import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from core import config, db, llm

MANIFEST = {"id": "dashboard", "label": "Usage, Audit & Health", "icon": "📊",
            "roles": ["admin"], "order": 90}


def render(user):
    st.header("📊 Usage, Audit & Health")
    s = config.settings()
    st_ = llm.status()
    c = st.columns(4)
    count = lambda t: db.rows(select(func.count()).select_from(t))[0]["count_1"]  # noqa: E731
    c[0].metric("Users", count(db.users))
    c[1].metric("Documents", count(db.documents))
    c[2].metric("Chunks", count(db.chunks))
    c[3].metric("Questions asked", count(db.chat_messages))
    st.subheader("System health")
    st.table(pd.DataFrame([
        {"item": "Database", "value": db.dialect()},
        {"item": "LLM backend", "value": st_["backend"]},
        {"item": "Model", "value": st_["model"]},
        {"item": "Embedding model", "value": s["embed_model"] if st_["backend"] == "gemini" else s["ollama_embed_model"]},
        {"item": "LLM ready", "value": "yes" if st_["ready"] else f"no ({st_['reason']})"},
    ]))
    b1, b2 = st.columns(2)
    if b1.button("Test LLM call"):
        try:
            st.success(f"OK: {llm.generate('Reply with the single word: pong')[:80]}")
        except llm.LLMError as e:
            st.error(f"Failed: {e}")
    if b2.button("Test embeddings"):
        v = llm.embed(["hello"], "RETRIEVAL_QUERY")
        st.success(f"OK: vector of length {len(v[0])}") if v else st.error(f"Failed: {llm.LAST_ERROR['embed'] or 'no LLM configured'}")
    st.subheader("Usage by module")
    a = db.audit_log
    use = db.rows(select(a.c.module, func.count().label("events")).group_by(a.c.module).order_by(func.count().desc()))
    if use:
        st.bar_chart(pd.DataFrame(use).set_index("module"))
    st.subheader("Answer feedback")
    m = db.chat_messages
    fb = db.rows(select(m.c.feedback, func.count().label("n")).where(m.c.feedback.is_not(None)).group_by(m.c.feedback))
    st.write({("helpful" if r["feedback"] == 1 else "not helpful"): r["n"] for r in fb} or "No feedback yet.")
    st.subheader("Unanswered questions (content gaps)")
    un = db.rows(select(m.c.created_at, m.c.username, m.c.question).where(m.c.found.is_(False)).order_by(m.c.id.desc()).limit(100))
    st.dataframe(pd.DataFrame(un), use_container_width=True, hide_index=True) if un else st.info("None yet.")
    st.subheader("Audit log (latest 200)")
    log = pd.DataFrame(db.rows(select(a).order_by(a.c.id.desc()).limit(200)))
    if not log.empty:
        st.dataframe(log, use_container_width=True, hide_index=True)
        st.download_button("Download audit log", log.to_csv(index=False).encode(), "audit_log.csv", "text/csv")
