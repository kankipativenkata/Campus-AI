import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import select

from core import audit, db, interview as I, llm

MANIFEST = {"id": "interview", "label": "Interview Mock Bot", "icon": "🎤",
            "roles": ["student"], "order": 22}


def _speak(text: str):
    components.html(
        "<button style='padding:6px 12px;font-size:14px;cursor:pointer' onclick=\"var u=new SpeechSynthesisUtterance("
        f"{json.dumps(text)});u.lang='en-IN';speechSynthesis.cancel();speechSynthesis.speak(u)\">🔊 Hear the question</button>",
        height=48)


def _do_transcribe(idx: int):
    audio = st.session_state.get(f"audio_{idx}")
    if not audio:
        st.session_state["iv_msg"] = "Record your answer first."
        return
    data = audio.getvalue()
    st.session_state["iv_dur"] = I.wav_duration(data)
    try:
        st.session_state[f"ans_{idx}"] = llm.transcribe(data, "audio/wav")
        st.session_state["iv_msg"] = "Transcribed. Fix any mistakes below, then submit."
    except llm.LLMError as e:
        st.session_state["iv_msg"] = f"Transcription failed: {e}. You can type your answer instead."


def _start(mode, category, n):
    bank = I.load_bank()
    st.session_state["iv"] = {"mode": mode, "category": category, "qs": I.pick_questions(bank, category, n),
                              "i": 0, "results": [], "scored": None}
    for k in [k for k in st.session_state if k.startswith(("ans_", "audio_"))]:
        del st.session_state[k]


def _history(user):
    s = db.rows(select(db.interview_sessions).where(db.interview_sessions.c.username == user["username"])
                .order_by(db.interview_sessions.c.id))
    if not s:
        return
    df = pd.DataFrame(s)
    st.subheader("Your progress")
    c = st.columns(3)
    c[0].metric("Answers practised", len(df))
    c[1].metric("Average overall", f"{df['overall'].mean():.1f} / 10")
    c[2].metric("Last 5 average", f"{df['overall'].tail(5).mean():.1f} / 10")
    st.line_chart(df.reset_index()[["correctness", "clarity", "overall"]])
    st.dataframe(df.groupby("category")[["correctness", "clarity", "overall"]].mean().round(1), use_container_width=True)


def render(user):
    st.header("🎤 Interview Mock Bot")
    st.caption("Practise viva and placement interviews. The question is read aloud by your browser, you answer by voice "
               "or text, and you get scores for correctness and clarity with specific tips.")
    bank = I.load_bank()
    iv = st.session_state.get("iv")
    if not iv:
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Type", ["viva", "placement"], format_func=lambda m: "Viva (subject)" if m == "viva" else "Placement (HR + technical)")
        cats = [k for k, v in bank.items() if v["mode"] == mode]
        category = c2.selectbox("Topic", cats)
        n = c3.slider("Questions", 1, len(bank[category]["questions"]), min(3, len(bank[category]["questions"])))
        if not llm.can_transcribe():
            st.info("Voice transcription needs GEMINI_API_KEY. Without it you can still record, then type what you said.")
        if st.button("▶️ Start interview", type="primary"):
            _start(mode, category, n)
            st.rerun()
        _history(user)
        return
    qs, i = iv["qs"], iv["i"]
    if i >= len(qs):
        res = iv["results"]
        st.success("Interview finished.")
        m = st.columns(3)
        m[0].metric("Overall", f"{sum(r['overall'] for r in res) / len(res):.1f} / 10")
        m[1].metric("Correctness", f"{sum(r['correctness'] for r in res) / len(res):.1f} / 10")
        m[2].metric("Clarity", f"{sum(r['clarity'] for r in res) / len(res):.1f} / 10")
        st.dataframe(pd.DataFrame([{"Question": q["text"], "Correctness": r["correctness"], "Clarity": r["clarity"], "Overall": r["overall"]}
                                   for q, r in zip(qs, res)]), use_container_width=True, hide_index=True)
        weakest = min(zip(qs, res), key=lambda x: x[1]["overall"])
        st.info(f"Revise next: **{weakest[0]['text']}**")
        if st.button("Start another interview"):
            del st.session_state["iv"]
            st.rerun()
        _history(user)
        return
    q = qs[i]
    st.progress(i / len(qs), text=f"Question {i + 1} of {len(qs)} · {iv['category']}")
    st.subheader(q["text"])
    _speak(q["text"])
    scored = iv["scored"]
    if scored is None:
        st.audio_input("🎙️ Record your answer (click, speak, click again)", key=f"audio_{i}")
        st.button("📝 Transcribe my recording", on_click=_do_transcribe, args=(i,), disabled=not llm.can_transcribe())
        if st.session_state.get("iv_msg"):
            st.caption(st.session_state.pop("iv_msg"))
        answer = st.text_area("Your answer (transcript or typed)", key=f"ans_{i}", height=160)
        if st.button("✅ Submit answer", type="primary", disabled=not answer.strip()):
            with st.spinner("Scoring…"):
                r = I.score_answer(q, answer, st.session_state.pop("iv_dur", None))
            iv["results"].append(r)
            iv["scored"] = r
            db.insert(db.interview_sessions, username=user["username"], mode=iv["mode"], category=iv["category"],
                      question=q["text"], transcript=answer, correctness=r["correctness"], clarity=r["clarity"],
                      overall=r["overall"], feedback=r["feedback"], created_at=db.now())
            audit.log(user, "interview", f"answer {q['id']} overall={r['overall']}")
            st.rerun()
        if st.button("Quit interview"):
            del st.session_state["iv"]
            st.rerun()
        return
    r = scored
    m = st.columns(4)
    m[0].metric("Correctness", f"{r['correctness']} / 10")
    m[1].metric("Clarity", f"{r['clarity']} / 10")
    m[2].metric("Overall", f"{r['overall']} / 10")
    m[3].caption(f"Scored by: {r['source']}")
    st.markdown("**Key points**")
    for p in r["points"]:
        st.markdown(f"{'✅' if p['covered'] else '🟡' if p['partial'] else '❌'} {p['point']}")
    cd = r["clarity_detail"]
    st.caption(f"Words: {cd['words']} · Fillers: {cd['fillers']} · Avg sentence: {cd['avg_sentence']} words"
               + (f" · Pace: {cd['wpm']} words/min" if cd["wpm"] else ""))
    for t in cd["tips"]:
        st.markdown(f"- 💡 {t}")
    if r["feedback"]:
        st.info(r["feedback"])
    if r["ai"] and r["ai"].get("model_answer"):
        with st.expander("Model answer"):
            st.write(r["ai"]["model_answer"])
        for x in r["ai"].get("mistakes", []):
            st.warning(f"Check: {x}")
    if st.button("Next question ➡️" if i + 1 < len(qs) else "Finish ➡️", type="primary"):
        iv["i"] += 1
        iv["scored"] = None
        st.rerun()
