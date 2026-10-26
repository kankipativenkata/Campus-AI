import pandas as pd
import streamlit as st

from core import audit, parsing, projects as PJ

MANIFEST = {"id": "projects", "label": "Project Showcase", "icon": "🚀",
            "roles": ["student", "faculty", "admin"], "order": 28}


def _tags_ui(prefix: str, tags: dict) -> dict:
    tech_opts = sorted(set(PJ.TECH) | set(tags["tech"]))
    tech = st.multiselect("Tech stack", tech_opts, default=tags["tech"], key=f"{prefix}_tech")
    extra = st.text_input("Other technologies (comma separated)", key=f"{prefix}_extra")
    tech = sorted(set(tech) | {x.strip().lower() for x in extra.split(",") if x.strip()})
    domains = st.multiselect("Domains", list(PJ.DOMAINS), default=tags["domains"], key=f"{prefix}_dom")
    diff = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"],
                        index=["Beginner", "Intermediate", "Advanced"].index(tags["difficulty"]), key=f"{prefix}_diff")
    return {"tech": tech, "domains": domains, "difficulty": diff, "summary": tags["summary"]}


def _card(p: dict):
    with st.container(border=True):
        st.markdown(f"**{p['title']}** · _{p['difficulty']}_ · by {p['owner_name'] or p['owner']}")
        st.write(p["summary"] or p["description"][:200])
        st.caption("Tech: " + (", ".join(p["tech"]) or "n/a") + "  |  Domains: " + (", ".join(p["domains"]) or "n/a"))
        if p["link"]:
            st.markdown(f"[Open project link]({p['link']})")


def render(user):
    st.header("🚀 Peer Project Showcase")
    st.caption("Share a project, get AI-assisted tags, find similar work, and meet people with complementary skills.")
    allp = PJ.load_projects()
    t_browse, t_up, t_sim, t_prof = st.tabs(["Browse", "Upload a project", "Similar projects & teammates", "My profile"])
    with t_browse:
        c1, c2, c3 = st.columns(3)
        q = c1.text_input("Search title or description")
        tech_f = c2.multiselect("Filter by tech", sorted({t for p in allp for t in p["tech"]}))
        dom_f = c3.multiselect("Filter by domain", sorted({d for p in allp for d in p["domains"]}))
        shown = [p for p in allp if (not q or q.lower() in (p["title"] + p["description"]).lower())
                 and set(tech_f) <= set(p["tech"]) and set(dom_f) <= set(p["domains"])]
        st.caption(f"{len(shown)} project(s)")
        for p in shown:
            _card(p)
            if user["role"] == "admin" or p["owner"] == user["username"]:
                if st.button("Delete", key=f"del_{p['id']}"):
                    PJ.delete_project(p["id"], user["username"], user["role"] == "admin")
                    audit.log(user, "projects", f"delete {p['id']}")
                    st.rerun()
    with t_up:
        title = st.text_input("Project title", key="pj_title")
        desc = st.text_area("Description (what it does, how, which tools)", height=150, key="pj_desc")
        link = st.text_input("Link (GitHub or demo, optional)", key="pj_link")
        readme = st.file_uploader("Optional README / report (txt, md, pdf, docx)", type=["txt", "md", "pdf", "docx"], key="pj_readme")
        text_for_tags = desc
        if readme:
            try:
                text_for_tags += "\n" + "\n".join(t for _, t in parsing.extract_pages(readme.name, readme.getvalue()))[:6000]
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not read the file: {e}")
        if st.button("🏷️ Analyze and suggest tags", disabled=not (title.strip() and desc.strip())):
            st.session_state["pj_tags"] = PJ.auto_tag(title, text_for_tags)
            audit.log(user, "projects", "auto-tag")
        tags = st.session_state.get("pj_tags")
        if tags and title.strip():
            st.success(f"Tags suggested by: {tags['source']}. Edit them if needed.")
            final = _tags_ui("up", tags)
            sims = PJ.similar_projects({"title": title, "description": desc, **final, "embedding": None}, allp, 3)
            sims = [s for s in sims if s["score"] > 0.25]
            if sims:
                st.info("Similar projects already exist: " + "; ".join(f"{s['title']} ({s['score']})" for s in sims))
            if st.button("📤 Publish project", type="primary"):
                PJ.add_project(user["username"], user["name"], title.strip(), desc.strip(), link.strip(), final)
                audit.log(user, "projects", f"publish {title[:60]}")
                st.session_state.pop("pj_tags", None)
                st.success("Published. See it in Browse.")
                st.rerun()
    with t_sim:
        if not allp:
            st.info("No projects yet.")
        else:
            sel = st.selectbox("Pick a project", allp, format_func=lambda p: f"{p['title']} (by {p['owner_name'] or p['owner']})")
            others = [p for p in allp if p["id"] != sel["id"]]
            sims = PJ.similar_projects(sel, others, 5)
            st.subheader("Similar projects")
            for s in sims:
                _card(s)
                st.caption(f"Similarity {s['score']}" + (f" · shared: {', '.join(s['shared'])}" if s["shared"] else ""))
            if sel["owner"] == user["username"]:
                st.subheader("Potential teammates")
                mine = next((p for p in PJ.load_profiles() if p["username"] == user["username"]), None)
                target = {**sel, "my_skills": mine["skills"] if mine else []}
                tm = PJ.teammate_suggestions(target, sims, PJ.load_profiles(), user["username"])
                if not tm:
                    st.info("No suggestions yet. Others appear here when they open their profile to teaming.")
                if tm:
                    st.dataframe(pd.DataFrame([{"Name": t["name"], "Brings (missing from your stack)": ", ".join(t["brings"]) or "-",
                                                "Shared interests": ", ".join(t["shared_interests"]) or "-", "Match": t["score"]}
                                               for t in tm]), use_container_width=True, hide_index=True)
            else:
                st.caption("Open one of your own projects to see teammate suggestions.")
    with t_prof:
        st.caption("Only people who tick “open to teaming” appear in teammate suggestions.")
        mine = next((p for p in PJ.load_profiles() if p["username"] == user["username"]), None)
        skills = st.multiselect("My skills", sorted(PJ.TECH), default=[s for s in (mine["skills"] if mine else []) if s in PJ.TECH])
        interests = st.multiselect("Interest domains", list(PJ.DOMAINS), default=mine["interest_domains"] if mine else [])
        open_t = st.checkbox("I am open to teaming up (show my name to others)", value=mine["open_to_team"] if mine else False)
        if st.button("Save profile"):
            PJ.save_profile(user["username"], user["name"], skills, interests, open_t)
            audit.log(user, "projects", "profile")
            st.success("Saved.")
