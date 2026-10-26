"""Campus AI Platform - main entry point. Run: streamlit run app.py"""
import streamlit as st

from core import auth, config, registry, seed

cfg = config.university()
st.set_page_config(page_title=cfg["name"], page_icon=cfg.get("icon", "🎓"), layout="wide")


@st.cache_resource(show_spinner="Setting up database and demo data (first run only)…")
def bootstrap():
    return seed.seed_all()


bootstrap()

# Public read-only timetable link: https://<your-app>/?tt=CODE (no login needed)
_share = st.query_params.get("tt")
if _share:
    from modules import faculty_timetable
    faculty_timetable.render_public(_share)
    st.stop()


def login_view():
    st.title(f"{cfg.get('icon', '🎓')} {cfg['name']}")
    st.caption(cfg["tagline"])
    left, right = st.columns(2)
    with left:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary")
        if ok:
            user = auth.authenticate(u, p)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Wrong username or password, or the account is inactive.")
    with right:
        if cfg.get("show_demo_credentials", True):
            st.info("**Demo accounts**\n\n"
                    "| Role | Username | Password |\n|---|---|---|\n"
                    + "\n".join(f"| {r} | `{n}` | `{pw}` |" for n, _, r, pw in auth.DEMO_USERS))
            st.caption("Sample documents are fictional. Change or hide these credentials before a real deployment.")


user = st.session_state.get("user")
if not user:
    login_view()
    st.stop()

mods = registry.for_role(user["role"])
with st.sidebar:
    st.markdown(f"### {cfg.get('icon', '🎓')} {cfg['name']}")
    st.write(f"**{user['name']}** · {user['role']}")
    labels = [f"{m['icon']} {m['label']}" for m, _ in mods]
    choice = st.radio("Modules", labels, label_visibility="collapsed")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()
mods[labels.index(choice)][1].render(user)
