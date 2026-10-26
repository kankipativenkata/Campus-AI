"""Simulates Gemini API failures (500 INTERNAL, auth errors) without any network."""
import io

import pytest
from PIL import Image

from core import llm

ERR500 = "500 INTERNAL. {'error': {'code': 500, 'message': 'Internal error encountered.', 'status': 'INTERNAL'}}"


class _Resp:
    def __init__(self, text):
        self.text = text


class FakeClient:
    def __init__(self, plan):
        self.plan, self.calls = list(plan), []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        step = self.plan.pop(0) if self.plan else "ok"
        if step == "ok":
            return _Resp("fine")
        raise Exception(step)


@pytest.fixture
def gem(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMMA_MODEL", "main-model")
    monkeypatch.setenv("FALLBACK_MODEL", "backup-model")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def install(plan):
        fake = FakeClient(plan)
        monkeypatch.setattr(llm, "_gemini", lambda key: fake)
        return fake
    return install


def test_transient_500_is_retried_then_succeeds(gem):
    fake = gem([ERR500, ERR500, "ok"])
    assert llm.generate("hi") == "fine"
    assert fake.calls == ["main-model"] * 3


def test_persistent_500_falls_back_to_second_model(gem):
    fake = gem([ERR500] * 4 + ["ok"])
    assert llm.generate("hi") == "fine"
    assert fake.calls[:4] == ["main-model"] * 4 and fake.calls[4] == "backup-model"


def test_all_models_failing_gives_friendly_message(gem):
    gem([ERR500] * 8)
    with pytest.raises(llm.LLMError) as e:
        llm.generate("hi")
    assert "temporary internal error" in str(e.value) and "smaller, clearer image" in str(e.value)


def test_bad_api_key_is_not_retried_or_fallen_back(gem):
    fake = gem(["400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key."])
    with pytest.raises(llm.LLMError) as e:
        llm.generate("hi")
    assert fake.calls == ["main-model"] and "API key was rejected" in str(e.value)


def test_missing_model_message(gem):
    gem(["404 NOT_FOUND. models/main-model is not found"] * 2)
    with pytest.raises(llm.LLMError) as e:
        llm.generate("hi")
    assert "GEMMA_MODEL" in str(e.value)


def test_json_helper_survives_a_transient_500(gem):
    gem([ERR500, "ok"])
    with pytest.raises(llm.LLMError):          # "fine" is not JSON, but the 500 was retried first
        llm.generate_json("x")


def test_default_fallback_model_is_the_other_gemma(monkeypatch):
    from core import config
    monkeypatch.delenv("FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
    assert config.settings()["fallback_model"] == "gemma-4-31b-it"
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-31b-it")
    assert config.settings()["fallback_model"] == "gemma-4-26b-a4b-it"


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def test_large_images_are_downscaled_and_small_ones_untouched():
    big, mime = llm._prep_image(_png(4000, 3000), "image/png")
    assert max(Image.open(io.BytesIO(big)).size) <= 1600 and mime in ("image/jpeg", "image/png")
    small = _png(800, 600)
    assert llm._prep_image(small, "image/png") == (small, "image/png")
    assert llm._prep_image(b"not an image", "image/png") == (b"not an image", "image/png")
