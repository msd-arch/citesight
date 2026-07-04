"""LLM client tests: provider wiring, JSON repair, 429 backoff, disk cache, fakes."""
import json

import httpx
import pytest
import respx

from citesight.config.settings import Settings
from citesight.models.llm import (
    PROVIDERS,
    FakeLLM,
    LlmError,
    OpenAICompatLLM,
    make_llm,
)

BASE = "https://fake-llm.test/v1"


def _chat_response(content: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
    }


def _client(**kwargs) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        base_url=BASE, api_key="k", model="m", requests_per_min=0, **kwargs
    )


@respx.mock
def test_complete_and_json():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response('{"a": 1}'))
    )
    assert _client().complete_json("s", "u") == {"a": 1}


@respx.mock
def test_json_repair_round_trip():
    route = respx.post(f"{BASE}/chat/completions")
    route.side_effect = [
        httpx.Response(200, json=_chat_response("not json at all")),
        httpx.Response(200, json=_chat_response('{"fixed": true}')),
    ]
    assert _client().complete_json("s", "u") == {"fixed": True}
    assert route.call_count == 2


@respx.mock
def test_429_honors_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    route = respx.post(f"{BASE}/chat/completions")
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": "rate"}),
        httpx.Response(200, json=_chat_response("ok")),
    ]
    assert _client(max_retries=2).complete("s", "u") == "ok"
    assert 7.0 in sleeps


@respx.mock
def test_429_exhaustion_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "rate"})
    )
    with pytest.raises(LlmError, match="rate limited"):
        _client(max_retries=1).complete("s", "u")


@respx.mock
def test_disk_cache_hits_once(tmp_path):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("cached-answer"))
    )
    c = _client(cache_dir=str(tmp_path))
    assert c.complete("s", "u") == "cached-answer"
    assert c.complete("s", "u") == "cached-answer"
    assert route.call_count == 1  # second call served from disk
    assert list(tmp_path.glob("*.json"))


def test_fake_llm_script_and_responses():
    f = FakeLLM(script=['{"n": 1}', '{"n": 2}'])
    assert f.complete_json("s", "u")["n"] == 1
    assert f.complete_json("s", "u")["n"] == 2
    f2 = FakeLLM(responses={"route": '{"query_type": "factoid"}'})
    assert f2.complete_json("s", "please route this")["query_type"] == "factoid"


def test_make_llm_roles_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings(
        sec_user_agent="t", data_dir=tmp_path,
        agent_llm_provider="groq", judge_llm_provider="fake", _env_file=None,
    )
    assert isinstance(make_llm("judge", settings), FakeLLM)
    with pytest.raises(LlmError, match="GROQ_API_KEY"):
        make_llm("agent", settings)


def test_make_llm_ollama_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    settings = Settings(
        sec_user_agent="t", data_dir=tmp_path,
        agent_llm_provider="ollama", _env_file=None,
    )
    llm = make_llm("agent", settings)
    assert llm.model_id == PROVIDERS["ollama"].default_model
