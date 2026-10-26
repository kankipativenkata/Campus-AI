"""LLM + embedding client. Backends: gemini (Gemma 4 via Gemini API), ollama (local), none."""
from __future__ import annotations

import base64
import json
import re
import time
from functools import lru_cache

import requests

from core.config import settings

Image = tuple[bytes, str]  # (bytes, mime type)
LAST_ERROR = {"generate": "", "embed": ""}


class LLMError(Exception):
    pass


def status() -> dict:
    s = settings()
    b = s["backend"]
    if b == "gemini":
        ready = bool(s["gemini_key"])
        return {"backend": b, "model": s["model"], "ready": ready,
                "reason": "" if ready else "GEMINI_API_KEY is not set"}
    if b == "ollama":
        return {"backend": b, "model": s["ollama_model"], "ready": True, "reason": ""}
    return {"backend": "none", "model": "-", "ready": False,
            "reason": "No LLM configured (set GEMINI_API_KEY or LLM_BACKEND=ollama)"}


def available() -> bool:
    return status()["ready"]


@lru_cache(maxsize=2)
def _gemini(key: str):
    from google import genai
    return genai.Client(api_key=key)


def _retry(fn, attempts: int = 3):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            if any(t in msg for t in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timed out")):
                time.sleep(2 ** i)
                continue
            break
    raise LLMError(str(last))


def generate(prompt: str, images: list[Image] | None = None, system: str | None = None,
             temperature: float = 0.2) -> str:
    st = status()
    if not st["ready"]:
        raise LLMError(st["reason"])
    s = settings()
    full = f"{system}\n\n{prompt}" if system else prompt  # Gemma: instructions go in the prompt
    images = images or []
    try:
        if s["backend"] == "gemini":
            from google.genai import types
            parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in images] + [full]
            cfg = types.GenerateContentConfig(temperature=temperature)
            resp = _retry(lambda: _gemini(s["gemini_key"]).models.generate_content(
                model=s["model"], contents=parts, config=cfg))
            text = resp.text or ""
        else:
            msg = {"role": "user", "content": full}
            if images:
                msg["images"] = [base64.b64encode(b).decode() for b, _ in images]
            r = requests.post(f"{s['ollama_url']}/api/chat", timeout=300, json={
                "model": s["ollama_model"], "messages": [msg], "stream": False,
                "options": {"temperature": temperature}})
            r.raise_for_status()
            text = r.json()["message"]["content"]
        LAST_ERROR["generate"] = ""
        return text.strip()
    except LLMError as e:
        LAST_ERROR["generate"] = str(e)
        raise
    except Exception as e:  # noqa: BLE001
        LAST_ERROR["generate"] = str(e)
        raise LLMError(str(e)) from e


def can_transcribe() -> bool:
    s = settings()
    return s["backend"] == "gemini" and bool(s["gemini_key"])


def transcribe(audio: bytes, mime: str = "audio/wav") -> str:
    """Speech to text via the Gemini API (set TRANSCRIBE_MODEL to any audio-capable model)."""
    if not can_transcribe():
        raise LLMError("Voice transcription needs GEMINI_API_KEY with the gemini backend. Type your answer instead.")
    from google.genai import types
    s = settings()
    parts = [types.Part.from_bytes(data=audio, mime_type=mime),
             "Transcribe this speech exactly as spoken. The speaker may use English, Telugu or a mix. "
             "Output only the transcript, no commentary. If there is no speech, output nothing."]
    try:
        resp = _retry(lambda: _gemini(s["gemini_key"]).models.generate_content(model=s["transcribe_model"], contents=parts))
        LAST_ERROR["generate"] = ""
        return (resp.text or "").strip()
    except LLMError as e:
        LAST_ERROR["generate"] = str(e)
        raise


def extract_json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if starts:
        i = min(starts)
        closer = "}" if t[i] == "{" else "]"
        j = t.rfind(closer)
        if j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:  # noqa: BLE001
                pass
    raise LLMError("Model did not return valid JSON")


def generate_json(prompt: str, images: list[Image] | None = None, system: str | None = None):
    """Ask for JSON; retry once with a stricter reminder if parsing fails."""
    text = generate(prompt + "\n\nReturn ONLY valid JSON. No markdown fences, no commentary.",
                    images, system, temperature=0.1)
    try:
        return extract_json(text)
    except LLMError:
        text = generate(prompt + "\n\nYour previous reply was not valid JSON. "
                        "Return ONLY the JSON object, nothing else.", images, system, temperature=0.0)
        return extract_json(text)


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]] | None:
    """Return one vector per text, or None if embeddings are unavailable (callers fall back)."""
    st = status()
    if not st["ready"] or not texts:
        return None
    s = settings()
    try:
        out: list[list[float]] = []
        if s["backend"] == "gemini":
            from google.genai import types
            for i in range(0, len(texts), 50):
                batch = texts[i:i + 50]
                resp = _retry(lambda b=batch: _gemini(s["gemini_key"]).models.embed_content(
                    model=s["embed_model"], contents=b,
                    config=types.EmbedContentConfig(task_type=task)))
                out.extend([list(e.values) for e in resp.embeddings])
        else:
            r = requests.post(f"{s['ollama_url']}/api/embed", timeout=300,
                              json={"model": s["ollama_embed_model"], "input": texts})
            r.raise_for_status()
            out = r.json()["embeddings"]
        LAST_ERROR["embed"] = ""
        return out
    except Exception as e:  # noqa: BLE001
        LAST_ERROR["embed"] = str(e)
        return None
