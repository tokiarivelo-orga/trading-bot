"""GeminiAdapter — Google's Generative Language REST API has its own wire
shape (query-string key, `contents`/`parts`), distinct from the
OpenAI-compatible providers, so it gets its own adapter and test file."""

from __future__ import annotations

import httpx
import pytest

from src.ai.adapters import gemini as gemini_module
from src.ai.adapters.gemini import GeminiAdapter
from src.ai.ports.llm import LLMMessage


class _FakeAsyncClient:
    def __init__(self, handler, **kwargs) -> None:
        self._handler = handler
        self.base_url = kwargs.get("base_url")

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, path: str, *, params=None, json=None) -> httpx.Response:
        return self._handler(self.base_url, path, params, json)


def _patch_client(monkeypatch, handler):
    monkeypatch.setattr(
        gemini_module.httpx, "AsyncClient", lambda **kwargs: _FakeAsyncClient(handler, **kwargs)
    )


async def test_complete_sends_gemini_request_and_parses_response(monkeypatch):
    captured = {}

    def handler(base_url, path, params, json):
        captured["base_url"] = base_url
        captured["path"] = path
        captured["params"] = params
        captured["json"] = json
        request = httpx.Request("POST", f"{base_url}{path}")
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "pong"}]}}]},
            request=request,
        )

    _patch_client(monkeypatch, handler)

    adapter = GeminiAdapter("gm-secret", "gemini-3.5-flash")
    result = await adapter.complete(LLMMessage(system="sys", user="ping"), max_tokens=256)

    assert result == "pong"
    assert captured["path"] == "models/gemini-3.5-flash:generateContent"
    assert captured["params"] == {"key": "gm-secret"}
    assert captured["json"]["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert captured["json"]["contents"] == [{"role": "user", "parts": [{"text": "ping"}]}]
    assert captured["json"]["generationConfig"] == {"maxOutputTokens": 256}


async def test_complete_joins_multiple_response_parts(monkeypatch):
    def handler(base_url, path, params, json):
        request = httpx.Request("POST", f"{base_url}{path}")
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "hello "}, {"text": "world"}]}}]},
            request=request,
        )

    _patch_client(monkeypatch, handler)

    adapter = GeminiAdapter("gm-secret", "gemini-3.5-flash")
    result = await adapter.complete(LLMMessage(system="", user="ping"))
    assert result == "hello world"


async def test_http_error_status_raises(monkeypatch):
    def handler(base_url, path, params, json):
        request = httpx.Request("POST", f"{base_url}{path}")
        return httpx.Response(403, text="invalid api key", request=request)

    _patch_client(monkeypatch, handler)

    adapter = GeminiAdapter("bad-key", "gemini-3.5-flash")
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.complete(LLMMessage(system="s", user="u"))
