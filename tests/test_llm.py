"""LLM cache: keys are deterministic and a cached response is served with no network.

The autouse fixture in conftest.py points verity.llm.CACHE at a temp dir, so the real
data/cache/llm_*.jsonl files are never read or written.
"""
import json

import pytest

from verity import llm


@pytest.fixture
def no_backend(monkeypatch):
    def boom(self, prompt, max_tokens):
        raise AssertionError("backend called although the response should be cached")
    monkeypatch.setattr(llm.LLM, "_groq", boom)
    monkeypatch.setattr(llm.LLM, "_local", boom)


def test_cache_dir_is_isolated(llm_cache_dir):
    assert llm.CACHE == llm_cache_dir
    assert llm._cache_file("openai/gpt-oss-20b").parent == llm_cache_dir


def test_key_deterministic_and_sensitive():
    k = llm._key("m", "prompt", 100)
    assert k == llm._key("m", "prompt", 100)
    assert len(k) == 64 and int(k, 16) >= 0
    assert k != llm._key("m2", "prompt", 100)
    assert k != llm._key("m", "prompt ", 100)
    assert k != llm._key("m", "prompt", 101)


def test_cache_file_name_per_model():
    f = llm._cache_file("openai/gpt-oss-20b")
    assert f.name == "llm_openai_gpt-oss-20b.jsonl"
    assert llm._cache_file("local:Qwen/Qwen2.5").name == "llm_local_Qwen_Qwen2.5.jsonl"


def test_complete_returns_cached_from_disk(llm_cache_dir, no_backend):
    model, prompt = "openai/gpt-oss-120b", "Is the sky blue?"
    key = llm._key(model, prompt, 600)
    (llm_cache_dir / "llm_openai_gpt-oss-120b.jsonl").write_text(
        "not json\n" + json.dumps({"no_key": 1}) + "\n" +
        json.dumps({"key": key, "model": model, "response": "yes"}) + "\n", encoding="utf-8")
    llm._cache.cache_clear()
    assert llm.LLM(model).complete(prompt, max_tokens=600) == "yes"


def test_complete_cached_from_memory(monkeypatch, no_backend):
    model, prompt = "openai/gpt-oss-20b", "hello"
    d = {llm._key(model, prompt, 800): "cached!"}
    monkeypatch.setattr(llm, "_cache", lambda: d)
    assert llm.LLM(model).complete(prompt) == "cached!"


def test_cache_miss_calls_backend_once_and_stores(llm_cache_dir, monkeypatch):
    calls = []

    def fake_groq(self, prompt, max_tokens):
        calls.append(prompt)
        return "fresh"
    monkeypatch.setattr(llm.LLM, "_groq", fake_groq)
    m = llm.LLM("openai/gpt-oss-20b")
    assert m.complete("q1") == "fresh"
    assert m.complete("q1") == "fresh"
    assert calls == ["q1"]
    lines = (llm_cache_dir / "llm_openai_gpt-oss-20b.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec == {"key": llm._key("openai/gpt-oss-20b", "q1", 800), "model": "openai/gpt-oss-20b",
                   "response": "fresh"}
    # a fresh process (cleared memory cache) reads it back from disk
    llm._cache.cache_clear()
    monkeypatch.setattr(llm.LLM, "_groq", lambda *a: pytest.fail("should be cached"))
    assert llm.LLM("openai/gpt-oss-20b").complete("q1") == "fresh"


def test_local_flag():
    from verity.config import LOCAL_LLM, LOCAL_LLM_LARGE
    assert llm.LLM(LOCAL_LLM).local and llm.LLM(LOCAL_LLM_LARGE).local
    assert llm.LLM("local:foo").local
    assert not llm.LLM("openai/gpt-oss-20b").local
