"""Deletes the local SQLite database so the next run re-creates tables, demo users and sample documents."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "campus.db"
print("Deleted" if p.exists() and not p.unlink() else "No local database found", p)
