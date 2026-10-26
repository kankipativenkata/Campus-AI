import json

import streamlit as st
from sqlalchemy import update

from core import audit, chat_memory, config, db, llm, rag

MANIFEST = {"id": "knowledge", "label": "Campus Knowledge Assistant", "icon": "📚",
            "roles": ["student", "faculty", "admin"], "order": 10}


def _feedback(msg_id: int, value: int):
    db.execute(update(db.chat_messages).where(db.chat_messages.c.id == msg_id).values(feedback=value))


def render(user):
    cfg = config.university()
    st.header("📚 Campus Knowledge Assistant")
    st.caption("Answers come only from documents uploaded by the university admin, with sources cited. "
               "It remembers your conversation, so you can ask follow-ups like “and can it be condoned?”.")
    if not llm.available():
        st.info("No LLM key configured: the assistant shows the closest matching passages instead of a written answer. "
                "Add GEMINI_API_KEY for full answers.")
    # conversation memory: restore the saved conversation once per login session
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = chat_memory.load_history(user["username"])
    hist = st.session_state["rag_history"]
    top = st.columns([6, 2])
    if hist:
        top[0].caption(f"Conversation memory: {len(hist)} message(s) saved. Your follow-ups use the last few turns.")
        if top[1].button("🧹 Start new conversation"):
            chat_memory.reset(user["username"])
            st.session_state["rag_history"] = []
            st.rerun()
    q = None
    sugg = cfg["rag"].get("suggested_questions", [])
    if sugg and not hist:
        cols = st.columns(len(sugg))
        for i, s in enumerate(sugg):
            if cols[i].button(s, key=f"sq{i}", use_container_width=True):
                q = s
    typed = st.chat_input("Ask about rules, fees, calendar, hostel, placements… (English or Telugu)")
    q = typed or q
    if q:
        with st.spinner("Searching university documents…"):
            res = rag.answer(q, user["role"], history=chat_memory.turns_for_prompt(hist))
        mid = db.insert(db.chat_messages, username=user["username"], question=q, answer=res["answer"],
                        sources=json.dumps(res["sources"]), found=res["found"], feedback=None, created_at=db.now())
        audit.log(user, "knowledge", f"ask (found={res['found']}, mode={res['mode'][:30]})")
        hist.append({"id": mid, "q": q, **res})
    for h in hist:
        with st.chat_message("user"):
            st.write(h["q"])
        with st.chat_message("assistant"):
            if h.get("standalone") and h["standalone"] != h["q"]:
                st.caption(f"Understood as: {h['standalone']}")
            st.markdown(h["answer"])
            if h["sources"]:
                with st.expander(f"Sources ({len(h['sources'])})"):
                    for s in h["sources"]:
                        st.markdown(f"**[{s['n']}] {s['title']}**, page {s['page']}, v{s['version']}, "
                                    f"effective {s['effective_date'] or 'n/a'}")
                        st.caption(s["snippet"])
            c1, c2, c3, _ = st.columns([1, 1, 2, 6])
            c1.button("👍", key=f"up{h['id']}", on_click=_feedback, args=(h["id"], 1))
            c2.button("👎", key=f"dn{h['id']}", on_click=_feedback, args=(h["id"], -1))
            if c3.button("Report outdated", key=f"out{h['id']}"):
                audit.log(user, "knowledge", f"report_outdated: {h['q'][:80]}")
                st.toast("Thanks, the admin can see this report in the audit log.")
