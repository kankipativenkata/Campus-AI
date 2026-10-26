"""Settings and config-file helpers. Reads env vars, .env, then Streamlit secrets."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
TEST_DATA = ROOT / "data" / "test_data"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def secret(key: str, default=None):
    """Env var first, then st.secrets (Streamlit Cloud), then default."""
    val = os.getenv(key)
    if val not in (None, ""):
        return val
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default


def settings() -> dict:
    key = secret("GEMINI_API_KEY")
    model = secret("GEMMA_MODEL", "gemma-4-26b-a4b-it")
    fallback = secret("FALLBACK_MODEL") or ("gemma-4-31b-it" if model != "gemma-4-31b-it" else "gemma-4-26b-a4b-it")
    return {
        "gemini_key": key,
        "backend": str(secret("LLM_BACKEND") or ("gemini" if key else "none")).lower(),
        "model": model,
        "fallback_model": fallback,
        "embed_model": secret("EMBED_MODEL", "gemini-embedding-001"),
        "transcribe_model": secret("TRANSCRIBE_MODEL", "gemini-2.5-flash"),
        "ollama_url": secret("OLLAMA_URL", "http://localhost:11434"),
        "ollama_model": secret("OLLAMA_MODEL", "gemma4"),
        "ollama_embed_model": secret("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "database_url": secret("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'campus.db'}"),
    }


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


DEFAULT_UNI = {
    "name": "Campus AI",
    "tagline": "One platform, three roles.",
    "icon": "🎓",
    "show_demo_credentials": True,
    "seed_sample_data": True,
    "not_found_message": "I could not find this in the university documents. Please contact the relevant office.",
    "rag": {
        "top_k": 5,
        "chunk_chars": 900,
        "chunk_overlap": 150,
        "min_similarity": 0.55,
        "min_term_coverage": 0.34,
        "suggested_questions": [],
    },
}


def university() -> dict:
    cfg = dict(DEFAULT_UNI)
    path = CONFIG_DIR / "university.yaml"
    if path.exists():
        loaded = load_yaml(path)
        rag = {**DEFAULT_UNI["rag"], **(loaded.get("rag") or {})}
        cfg.update(loaded)
        cfg["rag"] = rag
    return cfg


def config_files() -> list[Path]:
    return sorted(p for p in CONFIG_DIR.rglob("*.yaml"))


def list_yaml(folder: str) -> list[Path]:
    return sorted((CONFIG_DIR / folder).glob("*.yaml"))
