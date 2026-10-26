"""Interview practice: question bank, deterministic clarity/correctness heuristics, optional LLM scoring."""
from __future__ import annotations

import io
import random
import re
import wave

from core import llm
from core.config import CONFIG_DIR, load_yaml
from core.rag import tokenize

FILLERS = ["um", "uh", "umm", "er", "like", "you know", "basically", "actually", "kind of", "sort of", "i mean", "literally"]
W_CORRECT, W_CLARITY = 0.6, 0.4


def load_bank() -> dict:
    return load_yaml(CONFIG_DIR / "interview" / "question_bank.yaml")["categories"]


def pick_questions(bank: dict, category: str, n: int, seed: int | None = None) -> list[dict]:
    qs = [{**q, "category": category} for q in bank[category]["questions"]]
    random.Random(seed).shuffle(qs)
    return qs[:max(1, n)]


def wav_duration(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return None


def filler_count(text: str) -> int:
    t = " " + re.sub(r"[^a-z' ]", " ", text.lower()) + " "
    return sum(len(re.findall(rf"(?<=\s){re.escape(f)}(?=\s)", t)) for f in FILLERS)


def clarity_heuristic(text: str, duration_s: float | None = None) -> dict:
    """0-10 from measurable delivery signals. Transparent so students know how to improve."""
    words = re.findall(r"[\w']+", text)
    n = len(words)
    if n == 0:
        return {"score": 0.0, "words": 0, "fillers": 0, "filler_rate": 0.0, "avg_sentence": 0.0, "wpm": None, "tips": ["No answer detected."]}
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sent = n / max(1, len(sentences))
    fillers = filler_count(text)
    rate = 100 * fillers / n
    repeats = len(re.findall(r"\b(\w+)\s+\1\b", text.lower()))
    wpm = round(n / (duration_s / 60)) if duration_s and duration_s > 1 else None
    score, tips = 10.0, []
    if n < 25:
        score -= 3.5
        tips.append("Answer is very short: add a definition, an example and a conclusion.")
    elif n < 45:
        score -= 1.0
        tips.append("Add one concrete example.")
    if n > 220:
        score -= 1.5
        tips.append("Answer is long: keep it under about 90 seconds.")
    if rate > 6:
        score -= 3.0
        tips.append(f"Many filler words ({fillers}): pause silently instead.")
    elif rate > 3:
        score -= 1.5
        tips.append(f"Some filler words ({fillers}): try to pause instead.")
    if avg_sent > 30:
        score -= 1.5
        tips.append("Sentences are very long: split them.")
    elif avg_sent < 5 and n > 25:
        score -= 1.0
        tips.append("Sentences are choppy: connect your ideas.")
    if repeats >= 2:
        score -= 1.0
        tips.append("You repeat words: slow down and structure the answer.")
    if wpm is not None:
        if wpm > 190:
            score -= 1.5
            tips.append(f"Speaking fast ({wpm} words/min): aim for 110-170.")
        elif wpm < 80:
            score -= 1.0
            tips.append(f"Speaking slowly ({wpm} words/min): aim for 110-170.")
    return {"score": round(max(0.0, min(10.0, score)), 1), "words": n, "fillers": fillers, "filler_rate": round(rate, 1),
            "avg_sentence": round(avg_sent, 1), "wpm": wpm, "tips": tips}


def correctness_heuristic(text: str, key_points: list[dict]) -> dict:
    toks = set(tokenize(text))
    stems = {t for t in toks}
    hits, score_sum = [], 0.0
    for kp in key_points:
        kws = [tokenize(k)[0] if tokenize(k) else k.lower() for k in kp["keywords"]]
        kws = [k for k in kws if k]
        found = [k for k in kws if k in stems]
        need = 1 if len(kws) <= 2 else 2
        frac = min(1.0, len(found) / need) if kws else 0.0
        score_sum += frac
        hits.append({"point": kp["text"], "covered": frac >= 1.0, "partial": 0 < frac < 1.0, "fraction": round(frac, 2)})
    score = 10 * score_sum / max(1, len(key_points))
    return {"score": round(score, 1), "points": hits}


def llm_score(question: dict, transcript: str) -> dict | None:
    if not llm.available() or not transcript.strip():
        return None
    kp = "\n".join(f"- {k['text']}" for k in question["key_points"])
    prompt = f"""You are a strict but fair interviewer. Question: {question['text']}
Expected key points:
{kp}

Candidate's spoken answer (transcript, may contain speech-to-text errors):
<<<
{transcript}
>>>
Score CORRECTNESS 0-10 (coverage and accuracy of the key points; penalise wrong statements) and CLARITY 0-10
(structure, concision, plain language; ignore transcription mistakes). Return JSON:
{{"correctness": number, "clarity": number, "covered": [str], "missing": [str], "mistakes": [str],
 "feedback": "3 sentences, specific and kind", "model_answer": "a concise ideal answer in under 80 words"}}"""
    try:
        r = llm.generate_json(prompt)
        r["correctness"] = max(0.0, min(10.0, float(r.get("correctness", 0))))
        r["clarity"] = max(0.0, min(10.0, float(r.get("clarity", 0))))
        return r
    except (llm.LLMError, ValueError, TypeError):
        return None


def score_answer(question: dict, transcript: str, duration_s: float | None = None) -> dict:
    cl = clarity_heuristic(transcript, duration_s)
    co = correctness_heuristic(transcript, question["key_points"])
    ai = llm_score(question, transcript)
    if ai:   # AI judges content; delivery metrics stay measurable. Blend clarity 50/50 so fillers still count.
        correctness = round(0.7 * ai["correctness"] + 0.3 * co["score"], 1)
        clarity = round(0.5 * ai["clarity"] + 0.5 * cl["score"], 1)
        feedback, source = ai.get("feedback", ""), "AI + rules"
    else:
        correctness, clarity = co["score"], cl["score"]
        feedback, source = "", "rules only (no AI available)"
    overall = round(10 * (W_CORRECT * correctness + W_CLARITY * clarity) / 10, 1)
    return {"correctness": correctness, "clarity": clarity, "overall": overall, "source": source, "feedback": feedback,
            "clarity_detail": cl, "points": co["points"], "ai": ai}
