from core import auth, db


def test_demo_users_can_login():
    for u, _, role, pw in auth.DEMO_USERS:
        got = auth.authenticate(u, pw)
        assert got and got["role"] == role


def test_wrong_password_and_unknown_user():
    assert auth.authenticate("admin", "nope") is None
    assert auth.authenticate("ghost", "x") is None


def test_username_is_case_insensitive_and_trimmed():
    assert auth.authenticate("  ADMIN ", "admin123")["role"] == "admin"


def test_duplicate_and_bad_role():
    assert auth.create_user("admin", "x", "admin", "p")[0] is False
    assert auth.create_user("zed", "x", "wizard", "p")[0] is False


def test_inactive_user_cannot_login():
    auth.create_user("temp1", "Temp", "student", "pw12345")
    auth.set_active("temp1", False)
    assert auth.authenticate("temp1", "pw12345") is None
    auth.set_active("temp1", True)
    assert auth.authenticate("temp1", "pw12345") is not None


def test_password_hashes_are_salted():
    assert auth.hash_password("a") != auth.hash_password("a")
    assert auth.verify_password("a", auth.hash_password("a"))


def test_sqlite_dialect():
    assert db.dialect() == "sqlite"
