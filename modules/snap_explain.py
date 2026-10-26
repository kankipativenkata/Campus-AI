import streamlit as st

from core import audit, config, llm

MANIFEST = {"id": "snap", "label": "Snap-and-Explain", "icon": "📸",
            "roles": ["student", "faculty"], "order": 20}


def render(user):
    st.header("📸 Snap-and-Explain")
    st.caption("Photograph a textbook page, circuit, graph or handwritten problem and get a simple explanation "
               "plus 3 practice questions.")
    if not llm.available():
        st.warning("This module needs the Gemma 4 vision model. Set GEMINI_API_KEY (or LLM_BACKEND=ollama).")
    prompts = config.load_yaml(config.CONFIG_DIR / "prompts" / "snap_explain.yaml")
    c1, c2 = st.columns([1, 1])
    with c1:
        use_cam = st.toggle("Use camera instead of file upload")
        f = st.camera_input("Take a photo") if use_cam else st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "webp"], key="snap_file")
        language = st.selectbox("Explanation language", ["English", "Telugu", "English + Telugu"])
        level = st.selectbox("Level", ["Beginner", "Exam preparation"])
        extra = st.text_input("Anything specific? (optional)", placeholder="e.g. explain why the answer is 10 ms")
    with c2:
        if f:
            st.image(f, caption="Your image", use_container_width=True)
    if f and st.button("Explain this", type="primary", disabled=not llm.available()):
        data = f.getvalue()
        mime = f.type or "image/png"
        from PIL import Image
        import io
        w, h = Image.open(io.BytesIO(data)).size
        if min(w, h) < 300:
            st.warning("The image is quite small, so the explanation may be wrong. Consider retaking the photo.")
        with st.spinner("Gemma 4 is reading the image…"):
            try:
                res = llm.generate_json(prompts["user"].format(
                    language=language, level=level, extra=f"Student's question: {extra}" if extra else ""),
                    images=[(data, mime)], system=prompts["system"])
                st.session_state["snap_result"] = res
                audit.log(user, "snap", "explain")
            except llm.LLMError as e:
                st.error(f"Could not get an explanation: {e}")
    res = st.session_state.get("snap_result")
    if res and f:
        st.subheader(res.get("topic", "Explanation"))
        st.markdown(res.get("explanation", ""))
        if res.get("key_formulas"):
            st.markdown("**Key formulas**")
            for x in res["key_formulas"]:
                st.code(x, language="text")
        if res.get("common_mistakes"):
            st.markdown("**Common mistakes**")
            for x in res["common_mistakes"]:
                st.markdown(f"- {x}")
        st.markdown("**Practice questions**")
        for i, qa in enumerate(res.get("practice_questions", []), 1):
            with st.expander(f"Q{i}. {qa.get('q', '')}"):
                st.markdown(f"**Answer:** {qa.get('a', '')}")
        st.caption("AI explanations can contain mistakes. Verify with your faculty or textbook.")
