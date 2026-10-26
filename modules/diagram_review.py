import json

import streamlit as st

from core import audit, config, diagram_checks as dc, llm

MANIFEST = {"id": "diagram", "label": "Diagram Reviewer", "icon": "🧩",
            "roles": ["student", "faculty"], "order": 30}


def _show_findings(dtype, structure):
    findings = dc.check_structure(dtype, structure)
    st.metric("Rule-check score", f"{dc.score(findings)} / 100")
    if not findings:
        st.success("No rule violations found.")
    for f in findings:
        {"error": st.error, "warning": st.warning}.get(f["severity"], st.info)(f["message"])
    st.graphviz_chart(dc.to_dot(structure))
    return findings


def render(user):
    st.header("🧩 Diagram Reviewer")
    st.caption("Step 1: Gemma 4 reads your diagram into a structure. Step 2: fixed rules check it (so scoring is "
               "consistent). Step 3: the model suggests fixes.")
    prompts = config.load_yaml(config.CONFIG_DIR / "prompts" / "diagram_extract.yaml")
    dtype = st.selectbox("Diagram type", dc.DIAGRAM_TYPES)
    tab_img, tab_json = st.tabs(["From image (needs LLM)", "From structure JSON (works offline)"])
    with tab_img:
        if not llm.available():
            st.warning("Image extraction needs GEMINI_API_KEY. Use the JSON tab to try the rule checks offline.")
        f = st.file_uploader("Upload a flowchart / ER / UML image", type=["png", "jpg", "jpeg", "webp"], key="dg_file")
        if f:
            st.image(f, use_container_width=True)
        if f and st.button("Review diagram", type="primary", disabled=not llm.available()):
            with st.spinner("Extracting structure…"):
                try:
                    st.session_state["dg_struct"] = llm.generate_json(
                        prompts["user"].format(dtype=dtype), images=[(f.getvalue(), f.type or "image/png")])
                    st.session_state["dg_type"] = dtype
                    audit.log(user, "diagram", f"extract {dtype}")
                except llm.LLMError as e:
                    st.error(f"Extraction failed: {e}")
    with tab_json:
        sample = (config.TEST_DATA / "diagrams" / "flowchart_broken.json").read_text(encoding="utf-8")
        txt = st.text_area("Structure JSON", value=sample, height=220)
        if st.button("Check structure"):
            try:
                st.session_state["dg_struct"] = json.loads(txt)
                st.session_state["dg_type"] = dtype
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    s = st.session_state.get("dg_struct")
    if s:
        st.subheader("Result")
        findings = _show_findings(st.session_state.get("dg_type", dtype), s)
        if llm.available() and st.button("Get improvement suggestions"):
            with st.spinner("Thinking…"):
                try:
                    out = llm.generate_json(prompts["suggest"].format(
                        dtype=dtype, structure=json.dumps(s), findings=json.dumps(findings)))
                    st.markdown(out.get("summary", ""))
                    for x in out.get("suggestions", []):
                        st.markdown(f"- {x}")
                except llm.LLMError as e:
                    st.error(str(e))
