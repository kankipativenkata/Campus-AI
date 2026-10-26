"""Plugin loader: every file in modules/ with a MANIFEST dict and render(user) becomes a menu entry."""
from __future__ import annotations

import importlib
import pkgutil

import modules as pkg


def discover() -> list[tuple[dict, object]]:
    found = []
    for m in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"modules.{m.name}")
        man = getattr(mod, "MANIFEST", None)
        if man and hasattr(mod, "render"):
            found.append((man, mod))
    found.sort(key=lambda x: x[0].get("order", 100))
    return found


def for_role(role: str) -> list[tuple[dict, object]]:
    return [(m, mod) for m, mod in discover() if role in m["roles"]]
