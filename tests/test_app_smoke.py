"""Boots the real Streamlit app headlessly, logs in as each role, and opens every module."""
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

ROLE_USERS = [("student1", "student123"), ("faculty1", "faculty123"), ("admin", "admin123")]


def _login(u, p):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.text_input[0].set_value(u)
    at.text_input[1].set_value(p)
    at.button[0].click().run()
    return at


def test_login_page_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and len(at.text_input) == 2 and at.button[0].label == "Sign in"


def test_wrong_password_shows_error():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].set_value("admin")
    at.text_input[1].set_value("bad")
    at.button[0].click().run()
    assert at.error and "Wrong username" in at.error[0].value


@pytest.mark.parametrize("u,p", ROLE_USERS)
def test_every_module_opens_for_each_role(u, p):
    at = _login(u, p)
    assert not at.exception
    options = list(at.sidebar.radio[0].options)
    assert len(options) >= 3
    for opt in options:
        at.sidebar.radio[0].set_value(opt).run()
        assert not at.exception, f"{u} / {opt}: {[e.value for e in at.exception]}"


def test_role_menus_are_different():
    menus = {}
    for u, p in ROLE_USERS:
        menus[u] = set(_login(u, p).sidebar.radio[0].options)
    assert any("Knowledge Base Manager" in o for o in menus["admin"])
    assert not any("Knowledge Base Manager" in o for o in menus["student1"] | menus["faculty1"])
    assert any("CO-PO" in o for o in menus["faculty1"]) and not any("CO-PO" in o for o in menus["student1"])
    assert any("Snap" in o for o in menus["student1"])


def test_new_modules_are_role_scoped():
    stu, fac = set(_login("student1", "student123").sidebar.radio[0].options), set(_login("faculty1", "faculty123").sidebar.radio[0].options)
    assert any("Roadmap" in o for o in stu) and any("Interview" in o for o in stu) and any("Placement" in o for o in stu)
    assert not any("Roadmap" in o or "Interview" in o or "Placement" in o for o in fac)
    assert any("Timetable" in o for o in fac) and not any("Timetable" in o for o in stu)
    assert all(any("Project Showcase" in o for o in m) for m in (stu, fac))


def test_public_timetable_link_needs_no_login():
    from core import timetable as T
    tt = T.solve({"days": ["Mon", "Tue"], "periods": T.make_periods("09:00", 4, 50, {}), "theory_rooms": ["R1"], "lab_rooms": ["L1"], "unavailable": []},
                 T.SAMPLE_COURSES[:1], seed=0, restarts=2)
    code = T.save("fac", "Shared TT", tt)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["tt"] = code
    at.run()
    assert not at.exception and "Shared TT" in at.title[0].value and len(at.text_input) == 0
    bad = AppTest.from_file(APP, default_timeout=60)
    bad.query_params["tt"] = "nonexistent"
    bad.run()
    assert not bad.exception and "invalid" in bad.error[0].value
