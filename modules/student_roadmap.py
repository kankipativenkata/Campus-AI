import pandas as pd
import streamlit as st

from core import audit, config, llm, roadmap as R

MANIFEST = {"id": "roadmap", "label": "My Career Roadmap", "icon": "🧭",
            "roles": ["student"], "order": 15}


def render(user):
    st.header("🧭 My Career Roadmap")
    st.caption("Pick a goal. The planner chooses the fewest credits that cover the skills the goal needs, adds the "
               "prerequisites, and spreads courses over semesters. (Demo catalogue: replace config/roadmaps/*.yaml with "
               "your real curriculum.)")
    catalog, goals = R.load_catalog(), R.load_goals()
    c1, c2 = st.columns([2, 3])
    with c1:
        goal_name = st.selectbox("Career goal", list(goals))
        goal = goals[goal_name]
        st.write(goal.get("description", ""))
        done = st.multiselect("Courses already completed", list(catalog), format_func=lambda c: f"{c} {catalog[c]['name']}")
        cap = st.slider("Max credits per semester", 8, 28, 20)
        max_sem = st.slider("Semesters available", 1, 8, 6)
    p = R.plan(goal, catalog, set(done), cap, max_sem)
    with c2:
        m = st.columns(3)
        m[0].metric("Skill coverage", f"{p['covered_pct']}%")
        m[1].metric("Credits to take", p["total_credits"])
        m[2].metric("Semesters used", len(p["semesters"]))
        st.progress(min(p["covered_pct"] / 100, 1.0))
    if p["unscheduled"]:
        st.warning("These courses did not fit in the semesters available (raise credits or semesters): " + ", ".join(p["unscheduled"]))
    st.subheader("Semester plan")
    for i, s in enumerate(p["semesters"], 1):
        with st.expander(f"Semester {i} · {s['credits']} credits", expanded=i <= 2):
            st.table(pd.DataFrame([{"Course": c, "Name": catalog[c]["name"], "Credits": catalog[c]["credits"],
                                    "Prerequisites": ", ".join(catalog[c].get("prereqs", [])) or "none",
                                    "Skills": ", ".join(catalog[c].get("skills", []))} for c in s["courses"]]))
    if not p["semesters"]:
        st.success("Nothing left to schedule for this goal with your completed courses.")
    else:
        st.subheader("Prerequisite map")
        st.graphviz_chart(R.to_dot(p, catalog, set(done)))
        st.caption("Green = planned, grey = completed.")
    if p["uncovered"]:
        st.subheader("Skills the curriculum does not cover (self-study)")
        for s, w in sorted(p["uncovered"].items(), key=lambda kv: -kv[1]):
            st.markdown(f"- **{s}** (importance {w}): {goal.get('self_study', {}).get(s, 'Learn through projects and free online material.')}")
    st.subheader("Projects and certificates to add")
    for x in goal.get("projects", []):
        st.markdown(f"- Project: {x}")
    for x in goal.get("certifications", []):
        st.markdown(f"- Certificate: {x}")
    st.download_button("⬇️ Download roadmap (Markdown)", R.to_markdown(goal_name, p, catalog).encode(), "roadmap.md", "text/markdown")
    if llm.available() and st.button("✨ Explain my roadmap in plain words"):
        try:
            with st.spinner("Writing…"):
                st.markdown(llm.generate("Explain this study roadmap to a first-year student in 150 words, motivating but "
                                         "honest, mentioning why the first semester matters most.\n\n" +
                                         R.to_markdown(goal_name, p, catalog)))
            audit.log(user, "roadmap", f"explain {goal_name}")
        except llm.LLMError as e:
            st.error(str(e))
