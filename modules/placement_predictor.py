import pandas as pd
import streamlit as st

from core import audit, placement as P

MANIFEST = {"id": "placement", "label": "Placement Probability", "icon": "📈",
            "roles": ["student"], "order": 25}


def render(user):
    st.header("📈 Campus Placement Probability")
    st.caption("A Bayesian estimate of your odds of being SHORTLISTED if you apply, from each company's past results and "
               "your profile. It is a planning aid, not a promise. Bundled company data is fictional demo data.")
    data = P.load()
    prior, companies = data["prior"], data["companies"]
    c1, c2 = st.columns(2)
    with c1:
        cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.05)
        backlogs = st.number_input("Active backlogs", 0, 10, 0)
        projects = st.slider("Projects completed", 0, 6, 1)
        internships = st.slider("Internships", 0, 3, 0)
    with c2:
        problems = st.number_input("Coding problems solved (any platform)", 0, 5000, 60, step=10)
        comm = st.slider("Communication self-rating (1 low to 5 high)", 1, 5, 3)
        skills = st.multiselect("Your skills", data["skills_catalog"], default=["python"])
    student = {"cgpa": cgpa, "backlogs": int(backlogs), "projects": projects, "internships": internships,
               "problems": int(problems), "comm": comm, "skills": skills}
    st.subheader("All companies")
    rows = []
    for c in companies:
        e = P.estimate(student, c, prior)
        rows.append({"Company": c["name"], "Role": c["role"],
                     "Eligible": "yes" if e["eligible"] else "no",
                     "Your odds %": round(e["mean"] * 100, 1), "Low (5%)": round(e["low"] * 100, 1),
                     "High (95%)": round(e["high"] * 100, 1), "Past applicants": c["history"]["applicants"],
                     "Typical rate %": round(e["base_rate"] * 100, 1)})
    df = pd.DataFrame(rows).sort_values("Your odds %", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Company")["Your odds %"])
    st.subheader("Look closer at one company")
    comp = st.selectbox("Company", companies, format_func=lambda c: f"{c['name']} ({c['role']})")
    e = P.estimate(student, comp, prior)
    if not e["eligible"]:
        for r in e["reasons"]:
            st.error(r)
        st.info("Fix the eligibility rule first; the model shows 0% until you are eligible.")
    else:
        m = st.columns(3)
        m[0].metric("Estimated odds", f"{e['mean'] * 100:.1f}%")
        m[1].metric("90% credible range", f"{e['low'] * 100:.1f}% to {e['high'] * 100:.1f}%")
        m[2].metric("Company base rate", f"{e['base_rate'] * 100:.1f}%")
        if comp["history"]["applicants"] < 60:
            st.warning("Very little history for this company, so the range is wide. Treat the number as rough.")
        st.subheader("What to improve")
        imp = P.improvements(student, comp, prior)
        if imp:
            st.dataframe(pd.DataFrame(imp).rename(columns={"action": "Action", "gain_pp": "Gain (percentage points)",
                                                         "new_p": "Odds after %"}), use_container_width=True, hide_index=True)
        else:
            st.success("No single change adds meaningfully more; your profile already matches this company well.")
        req = comp["required_skills"]
        miss = [s for s in req if s not in skills]
        st.markdown("**Required skills you are missing:** " + (", ".join(miss) if miss else "none"))
    audit.log(user, "placement", "estimate")
    with st.expander("How this model works"):
        st.markdown(
            "1. **Company base rate** = Beta posterior from past shortlisted/applicants plus a weak prior. Little history means a wide range.\n"
            "2. **Your adjustment** shifts the log-odds by how far your skill match, CGPA, coding practice, projects, internship and "
            "communication are above or below the company's typical applicant, minus a penalty per backlog.\n"
            "3. **Credible range** comes from 4,000 Monte Carlo draws of the base rate.\n\n"
            "The adjustment weights are expert-set, **not trained** on real outcomes. Ask your placement cell to load real history "
            "into `config/placement/companies.yaml` (Admin, Config Manager).")
