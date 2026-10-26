import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_env():
    """Fresh SQLite DB and NO LLM, so tests are deterministic, free and offline."""
    tmp = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.db"
    os.environ["LLM_BACKEND"] = "none"
    os.environ.pop("GEMINI_API_KEY", None)
    from core import seed
    seed.seed_all()
    yield
