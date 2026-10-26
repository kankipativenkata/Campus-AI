import pandas as pd
import streamlit as st
import yaml

from core import audit, config, rubric as rb

MANIFEST = {"id": "config", "label": "Config Manager", "icon": "⚙️",
            "roles": ["admin", "faculty"], "order": 70}


def _validate(path, data):
    if "rubrics" in path.parts:
        return rb.validate_rubric(data)
    if "outcomes" in path.parts:
        errs = []
        ids = {c.get("id") for c in data.get("cos", [])}
        if not ids:
            errs.append("No 'cos' defined.")
        for co in (data.get("matrix") or {}):
            if co not in ids:
                errs.append(f"matrix refers to unknown CO '{co}'.")
        return errs
    return []


def render(user):
    can_edit = user["role"] == "admin"
    st.header("⚙️ Config Manager" if can_edit else "⚙️ Config Viewer")
    st.caption("Rubrics, outcomes, prompts and branding live in YAML files, so another university only edits files. "
               + ("" if can_edit else "Faculty have read-only access."))
    files = config.config_files()
    path = st.selectbox("File", files, format_func=lambda p: str(p.relative_to(config.CONFIG_DIR)))
    raw = path.read_text(encoding="utf-8")
    text = st.text_area("YAML", value=raw, height=380, disabled=not can_edit, key=f"cfg_{path.name}")
    try:
        data = yaml.safe_load(text) or {}
        errs = _validate(path, data)
        st.success("YAML is valid.") if not errs else [st.error(e) for e in errs]
    except yaml.YAMLError as e:
        data, errs = None, [str(e)]
        st.error(f"Invalid YAML: {e}")
    c1, c2 = st.columns(2)
    if can_edit and c1.button("Save", type="primary", disabled=bool(errs)):
        path.write_text(text, encoding="utf-8")
        audit.log(user, "config", f"edit {path.name}")
        st.success("Saved. (On Streamlit Cloud, edits are lost on reboot: download the file and commit it to GitHub.)")
    c2.download_button("Download file", text.encode(), path.name, "text/yaml")
