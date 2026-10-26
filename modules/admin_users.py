import io

import pandas as pd
import streamlit as st

from core import audit, auth

MANIFEST = {"id": "users", "label": "User Management", "icon": "👥",
            "roles": ["admin"], "order": 80}


def render(user):
    st.header("👥 User Management")
    st.dataframe(pd.DataFrame(auth.list_users()), use_container_width=True, hide_index=True)
    t1, t2, t3 = st.tabs(["Add user", "Bulk import (CSV)", "Activate / reset password"])
    with t1:
        with st.form("add_user", clear_on_submit=True):
            c = st.columns(4)
            u, n = c[0].text_input("Username"), c[1].text_input("Full name")
            r, p = c[2].selectbox("Role", auth.ROLES), c[3].text_input("Password", type="password")
            if st.form_submit_button("Create"):
                ok, msg = auth.create_user(u, n, r, p)
                (st.success if ok else st.error)(msg)
                audit.log(user, "users", f"create {u}") if ok else None
    with t2:
        st.caption("Columns: username, name, role, password. A sample is in data/test_data/users/users_bulk_import.csv")
        f = st.file_uploader("CSV", type=["csv"], key="bulk_csv")
        if f and st.button("Import"):
            df = pd.read_csv(io.BytesIO(f.getvalue()), dtype=str).fillna("")
            made, failed = 0, []
            for _, row in df.iterrows():
                ok, msg = auth.create_user(row.get("username"), row.get("name"), row.get("role"), row.get("password"))
                made += ok
                if not ok:
                    failed.append(f"{row.get('username')}: {msg}")
            st.success(f"Created {made} user(s).")
            for x in failed:
                st.warning(x)
            audit.log(user, "users", f"bulk import {made}")
    with t3:
        names = [x["username"] for x in auth.list_users()]
        sel = st.selectbox("User", names)
        c1, c2 = st.columns(2)
        if c1.button("Toggle active"):
            cur = next(x for x in auth.list_users() if x["username"] == sel)
            if sel == user["username"]:
                st.error("You cannot deactivate yourself.")
            else:
                auth.set_active(sel, not cur["active"])
                st.rerun()
        pw = c2.text_input("New password", type="password")
        if c2.button("Reset password") and pw:
            auth.set_password(sel, pw)
            st.success("Password updated.")
