import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core import audit, timetable as T

MANIFEST = {"id": "timetable", "label": "Timetable Generator", "icon": "🗓️",
            "roles": ["faculty", "admin"], "order": 35}

COURSE_COLS = ["course", "name", "faculty", "section", "theory_hours", "lab_hours", "lab_block", "lab_assistant"]


def _share_url(code: str) -> str:
    try:
        host = st.context.headers.get("Host", "")
        scheme = "http" if host.startswith(("localhost", "127.")) else "https"
        if host:
            return f"{scheme}://{host}/?tt={code}"
    except Exception:  # noqa: BLE001
        pass
    return f"?tt={code}"


def _views(tt: dict, key: str):
    n = T.names(tt)
    tabs = st.tabs(["By section", "By faculty", "By room / lab", "All sessions"])
    for tab, (label, items, fn) in zip(tabs[:3], [("Section", n["section"], T.section_grid),
                                                  ("Faculty", n["faculty"], T.faculty_grid),
                                                  ("Room", n["room"], T.room_grid)]):
        with tab:
            if not items:
                st.info("Nothing placed.")
                continue
            sel = st.selectbox(label, items, key=f"{key}_{label}")
            st.markdown(T.grid_html(fn(tt, sel), f"{label}: {sel}"), unsafe_allow_html=True)
    with tabs[3]:
        st.dataframe(T.to_long_df(tt), use_container_width=True, hide_index=True)


def render_public(code: str):
    """Read-only page for a shared link (no login). Called from app.py when ?tt=CODE is present."""
    rec = T.load(code)
    if not rec:
        st.error("This timetable link is invalid or was removed by its owner.")
        return
    st.title(f"🗓️ {rec['title']}")
    st.caption("Read-only shared timetable.")
    st.markdown(f"<style>{T.CSS}</style>", unsafe_allow_html=True)
    _views(rec["tt"], "pub")
    st.download_button("⬇️ Download printable HTML", T.to_html(rec["tt"], rec["title"]).encode(), "timetable.html", "text/html")


def render(user):
    st.header("🗓️ Timetable Generator (theory + labs)")
    st.caption("Enter your set-up, generate a clash-free timetable, then share a link or download it. "
               "All rules are checked again after generation.")
    ss = st.session_state
    ss.setdefault("tt_courses", pd.DataFrame(T.SAMPLE_COURSES, columns=COURSE_COLS))
    ss.setdefault("tt_unavail", pd.DataFrame(T.SAMPLE_UNAVAILABLE, columns=["faculty", "day", "period"]))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Days and periods")
        days = st.multiselect("Working days", T.DAYS_ALL, default=T.DAYS_ALL[:5])
        a, b, c = st.columns(3)
        n_periods = a.number_input("Periods per day", 2, 12, 8)
        start = b.text_input("First period starts (HH:MM)", "09:00")
        duration = c.number_input("Period length (min)", 30, 120, 50, step=5)
        d, e, f, g = st.columns(4)
        lunch_after = d.number_input("Lunch after period (0 = none)", 0, 12, 4)
        lunch_min = e.number_input("Lunch minutes", 0, 120, 60, step=5)
        tea_after = f.number_input("Short break after (0 = none)", 0, 12, 0)
        tea_min = g.number_input("Break minutes", 0, 60, 10, step=5)
    with c2:
        st.subheader("2. Rooms and labs")
        theory_rooms = [x.strip() for x in st.text_area("Classrooms (one per line)", "R101\nR102\nR103").splitlines() if x.strip()]
        lab_rooms = [x.strip() for x in st.text_area("Lab rooms (one per line)", "LAB-1\nLAB-2").splitlines() if x.strip()]
    st.subheader("3. Courses, faculty and sections")
    st.caption("`theory_hours` = 1-period sessions per week. `lab_hours` = lab periods per week, scheduled in blocks of "
               "`lab_block` consecutive periods. `lab_assistant` (optional) must also be free for labs.")
    if st.button("Load sample data"):
        ss["tt_courses"] = pd.DataFrame(T.SAMPLE_COURSES, columns=COURSE_COLS)
        ss["tt_unavail"] = pd.DataFrame(T.SAMPLE_UNAVAILABLE, columns=["faculty", "day", "period"])
        st.rerun()
    courses = st.data_editor(ss["tt_courses"], num_rows="dynamic", use_container_width=True, key="tt_courses_ed",
                             column_config={"theory_hours": st.column_config.NumberColumn(min_value=0, max_value=10, step=1),
                                            "lab_hours": st.column_config.NumberColumn(min_value=0, max_value=10, step=1),
                                            "lab_block": st.column_config.NumberColumn(min_value=1, max_value=4, step=1)})
    st.subheader("4. Faculty unavailable slots (optional)")
    unavail = st.data_editor(ss["tt_unavail"], num_rows="dynamic", use_container_width=True, key="tt_unavail_ed",
                             column_config={"day": st.column_config.SelectboxColumn(options=T.DAYS_ALL),
                                            "period": st.column_config.SelectboxColumn(options=[f"P{i}" for i in range(1, 13)])})
    with st.expander("Advanced"):
        seed = st.number_input("Random seed (change to get a different valid timetable)", 0, 9999, 1)
        restarts = st.slider("Search effort (restarts)", 5, 60, 25)
    if st.button("⚙️ Generate timetable", type="primary"):
        if not days:
            st.error("Pick at least one working day.")
            return
        try:
            br = {int(lunch_after): int(lunch_min)} if lunch_after else {}
            if tea_after:
                br[int(tea_after)] = int(tea_min)
            periods = T.make_periods(start, int(n_periods), int(duration), br)
        except ValueError:
            st.error("Start time must look like 09:00.")
            return
        rows = [r for r in courses.fillna("").to_dict("records") if str(r.get("course", "")).strip()]
        if not rows:
            st.error("Add at least one course.")
            return
        cfg = {"days": days, "periods": periods, "theory_rooms": theory_rooms, "lab_rooms": lab_rooms,
               "unavailable": [r for r in unavail.fillna("").to_dict("records") if r.get("faculty")]}
        with st.spinner("Searching for a clash-free timetable…"):
            ss["tt_result"] = T.solve(cfg, rows, seed=int(seed), restarts=int(restarts))
        audit.log(user, "timetable", f"generate {len(rows)} courses")
        ss.pop("tt_share", None)
    tt = ss.get("tt_result")
    if not tt:
        return
    st.markdown(f"<style>{T.CSS}</style>", unsafe_allow_html=True)
    st.divider()
    m = st.columns(3)
    m[0].metric("Sessions placed", len(tt["placements"]))
    m[1].metric("Not placed", len(tt["unplaced"]))
    m[2].metric("Rule violations", len(tt["conflicts"]))
    if tt["conflicts"]:
        st.error("Violations found: " + "; ".join(tt["conflicts"][:5]))
    elif not tt["unplaced"]:
        st.success("Every session placed and every clash rule verified (faculty, section, room, labs, breaks).")
    for u in tt["unplaced"]:
        st.warning(f"Could not place {u['course']} ({u['kind']}, section {u['section']}, {u['faculty']}): {u['reason']} "
                   "Try more periods/rooms, fewer hours, or fewer unavailable slots.")
    _views(tt, "own")
    st.subheader("Share or download")
    title = st.text_input("Title", "Timetable, Semester 1")
    x, y, z = st.columns(3)
    x.download_button("⬇️ Excel (all views)", T.to_excel(tt), "timetable.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    y.download_button("⬇️ Printable HTML", T.to_html(tt, title).encode(), "timetable.html", "text/html")
    z.download_button("⬇️ CSV", T.to_long_df(tt).to_csv(index=False).encode(), "timetable.csv", "text/csv")
    if st.button("🔗 Create share link (public, read-only)"):
        ss["tt_share"] = T.save(user["username"], title, tt)
        audit.log(user, "timetable", f"share {ss['tt_share']}")
    if ss.get("tt_share"):
        st.success("Anyone with this link can view (not edit) this timetable:")
        st.code(_share_url(ss["tt_share"]), language="text")
        components.html(f"""<button onclick="navigator.clipboard.writeText({_share_url(ss['tt_share'])!r})">Copy link</button>""", height=40)
    saved = T.list_saved(user["username"])
    if saved:
        with st.expander(f"My shared timetables ({len(saved)})"):
            for s in saved:
                a, b = st.columns([6, 1])
                a.write(f"**{s['title']}** · `{_share_url(s['share_code'])}`")
                if b.button("Remove", key=f"rm_{s['share_code']}"):
                    T.unshare(s["share_code"], user["username"])
                    st.rerun()
