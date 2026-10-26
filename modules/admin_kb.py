import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from core import audit, db, llm, parsing, rag

MANIFEST = {"id": "kb", "label": "Knowledge Base Manager", "icon": "🗂️",
            "roles": ["admin"], "order": 60}


def render(user):
    st.header("🗂️ Knowledge Base Manager")
    st.caption("Upload university documents. They are chunked, embedded and become searchable for students and faculty "
               "according to the audience you choose.")
    if not llm.available():
        st.warning("No embeddings available (no API key): documents are indexed for keyword search only. "
                   "Add GEMINI_API_KEY and press 'Re-index' to enable semantic and Telugu search.")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Documents", type=[e.strip(".") for e in parsing.SUPPORTED], accept_multiple_files=True)
        c1, c2, c3 = st.columns(3)
        title = c1.text_input("Title (blank = file name)")
        category = c2.selectbox("Category", ["academic", "hostel", "fees", "placement", "hr", "general"])
        audience = c3.selectbox("Audience", ["all", "students", "faculty", "admin"])
        c4, c5 = st.columns(2)
        version = c4.text_input("Version", value="1.0")
        eff = c5.date_input("Effective date", value=datetime.date.today())
        go = st.form_submit_button("Upload and index", type="primary")
    if go and files:
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            t = title if (title and len(files) == 1) else f.name.rsplit(".", 1)[0]
            try:
                res = rag.ingest_document(f.name, f.getvalue(), t, category, audience, version, str(eff), user["username"])
                st.success(f"{f.name}: {res['chunks']} chunks, semantic index: {'yes' if res['embedded'] else 'no (keyword only)'}")
                audit.log(user, "kb", f"upload {f.name}")
            except Exception as e:  # noqa: BLE001
                st.error(f"{f.name}: {e}")
            bar.progress((i + 1) / len(files))
    d, c = db.documents, db.chunks
    docs = db.rows(select(d, func.count(c.c.id).label("chunks"), func.count(c.c.embedding).label("embedded"))
                   .outerjoin(c, c.c.document_id == d.c.id).group_by(d.c.id).order_by(d.c.id.desc()))
    st.subheader(f"Documents ({len(docs)})")
    if not docs:
        st.info("Nothing uploaded yet.")
        return
    df = pd.DataFrame(docs)[["id", "title", "category", "audience", "version", "effective_date", "status", "chunks", "embedded", "filename"]]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("**Manage a document**")
    sel = st.selectbox("Document", docs, format_func=lambda x: f"#{x['id']} {x['title']} (v{x['version']}, {x['status']})")
    a, b, c3, d4 = st.columns(4)
    if a.button("Archive" if sel["status"] == "active" else "Activate"):
        rag.set_status(sel["id"], "archived" if sel["status"] == "active" else "active")
        audit.log(user, "kb", f"toggle {sel['id']}")
        st.rerun()
    if b.button("Re-index (embeddings)"):
        st.success("Re-indexed.") if rag.reindex(sel["id"]) else st.error(f"Embedding failed: {llm.LAST_ERROR['embed'] or 'no LLM configured'}")
    if c3.button("Delete", type="secondary"):
        rag.delete_document(sel["id"])
        audit.log(user, "kb", f"delete {sel['id']}")
        st.rerun()
    st.markdown("**Test a search** (as a chosen role)")
    role = st.selectbox("Search as", ["student", "faculty", "admin"])
    q = st.text_input("Query")
    if q:
        hits = rag.search(q, role)
        if not hits:
            st.info("No relevant passage found (the assistant would answer 'not found').")
        for h in hits:
            st.markdown(f"**{h['title']}** p{h['page']}: score {h['score']}, cosine {h['cosine']}, term coverage {h['coverage']}")
            st.caption(h["text"][:300])
