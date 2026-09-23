"""Per-task AI provider settings API endpoints (AI_PROVIDER_SETTINGS_PLAN.md
§7, Phase 10.3) — `ProviderSettingsService` wired with a real in-memory
`LLMRouter` (fake factories, no network) and a real sqlite-backed
`ProviderConfigRepository`, mirroring `test_refinement_routes.py`'s pattern."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.ai.adapters.provider_config_repository import ProviderConfigRepository
from src.ai.api.routes_settings import router
from src.ai.application.llm_router import LLMProviderNotConfiguredError, LLMRouter
from src.ai.application.provider_settings import ProviderSettingsService
from src.ai.ports.llm import LLMMessage, ProviderSpec
from src.shared.db.base import Base

_DEFAULTS = {
    "pdf_extraction": ProviderSpec(provider="claude", model="claude-sonnet-5"),
    "code_generation": ProviderSpec(provider="claude", model="claude-sonnet-5"),
    "ten_trade_review": ProviderSpec(provider="claude", model="claude-haiku-4-5"),
    "code_refinement": ProviderSpec(provider="claude", model="claude-sonnet-5"),
}


class _FakeAdapter:
    def __init__(self, behavior: str) -> None:
        self._behavior = behavior

    async def complete(self, message: LLMMessage, *, max_tokens: int = 4096) -> str:
        if self._behavior == "error":
            raise RuntimeError("connection refused")
        return "pong"


class _FakeSecretStore:
    def __init__(self) -> None:
        self._keys: dict[str, str] = {}

    def get(self, provider: str) -> str | None:
        return self._keys.get(provider)

    def has(self, provider: str) -> bool:
        return provider in self._keys

    def set(self, provider: str, api_key: str) -> None:
        self._keys[provider] = api_key

    def clear(self, provider: str) -> None:
        self._keys.pop(provider, None)


def _factories(secret_store: _FakeSecretStore) -> dict:
    def _claude(spec: ProviderSpec):
        return _FakeAdapter("ok")

    def _ollama(spec: ProviderSpec):
        return _FakeAdapter("ok")

    def _claude_code(spec: ProviderSpec):
        raise LLMProviderNotConfiguredError("provider 'claude_code' binary not found on PATH")

    def _openclaw(spec: ProviderSpec):
        return _FakeAdapter("error")

    def _openai(spec: ProviderSpec):
        if not secret_store.get("openai"):
            raise LLMProviderNotConfiguredError("provider 'openai' selected but no API key is set")
        return _FakeAdapter("ok")

    return {
        "claude": _claude,
        "ollama": _ollama,
        "claude_code": _claude_code,
        "openclaw": _openclaw,
        "openai": _openai,
    }


@pytest.fixture
def env(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    repository = ProviderConfigRepository(session_factory)
    secret_store = _FakeSecretStore()
    llm_router = LLMRouter(dict(_DEFAULTS), _factories(secret_store))
    service = ProviderSettingsService(
        repository=repository, llm_router=llm_router, provider_secrets=secret_store
    )
    return SimpleNamespace(provider_settings=service)


@pytest.fixture
async def api(env):
    app = FastAPI()
    app.include_router(router)
    app.state.container = env
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://backend") as client:
        yield client


async def test_list_tasks_reports_yaml_defaults(api):
    response = await api.get("/ai/settings/tasks")
    assert response.status_code == 200
    body = {row["task"]: row for row in response.json()}
    assert set(body) == set(_DEFAULTS)
    assert body["ten_trade_review"]["provider"] == "claude"
    assert body["ten_trade_review"]["source"] == "default"
    assert body["ten_trade_review"]["configured"] is True


async def test_set_task_provider_overrides_and_persists(api):
    response = await api.put(
        "/ai/settings/tasks/ten_trade_review", json={"provider": "ollama", "model": "hermes3:8b"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "ollama"
    assert body["model"] == "hermes3:8b"
    assert body["source"] == "override"

    listed = await api.get("/ai/settings/tasks")
    row = next(r for r in listed.json() if r["task"] == "ten_trade_review")
    assert row["source"] == "override"
    assert row["provider"] == "ollama"


async def test_set_task_provider_unknown_task_404(api):
    response = await api.put(
        "/ai/settings/tasks/not_a_real_task", json={"provider": "claude", "model": "x"}
    )
    assert response.status_code == 404


async def test_set_task_provider_unknown_provider_422(api):
    response = await api.put(
        "/ai/settings/tasks/ten_trade_review",
        json={"provider": "not_a_real_provider", "model": "x"},
    )
    assert response.status_code == 422


async def test_clear_task_provider_reverts_to_default(api):
    await api.put(
        "/ai/settings/tasks/ten_trade_review", json={"provider": "ollama", "model": "hermes3:8b"}
    )
    response = await api.delete("/ai/settings/tasks/ten_trade_review")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "default"
    assert body["provider"] == "claude"


async def test_clear_task_provider_unknown_task_404(api):
    response = await api.delete("/ai/settings/tasks/not_a_real_task")
    assert response.status_code == 404


async def test_test_provider_ok(api):
    response = await api.post("/ai/settings/providers/claude/test")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "claude"
    assert body["ok"] is True
    assert body["message"] is None
    assert body["reply"] is None


async def test_test_provider_with_message_returns_reply(api):
    response = await api.post("/ai/settings/providers/claude/test", json={"message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["reply"] == "pong"


async def test_test_provider_reports_failure_without_raising(api):
    response = await api.post("/ai/settings/providers/claude_code/test")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "not found" in body["message"]


async def test_test_provider_unknown_provider_422(api):
    response = await api.post("/ai/settings/providers/not_a_real_provider/test")
    assert response.status_code == 422


async def test_list_providers_catalog(api):
    response = await api.get("/ai/settings/providers")
    assert response.status_code == 200
    body = {row["id"]: row for row in response.json()}
    assert set(body) == {
        "claude",
        "openai",
        "gemini",
        "mistral",
        "groq",
        "deepseek",
        "xai",
        "claude_code",
        "ollama",
        "openclaw",
    }
    assert body["claude"]["needs_secret"] is True
    presets = body["ollama"]["preset_models"]
    assert {p["model"] for p in presets} == {"hermes3:8b", "hermes3:70b"}
    assert body["openclaw"]["preset_models"] is None
    # claude/ollama don't need a saved key or fake factory beyond the
    # fixture's defaults to build successfully -> configured true; gemini
    # has no fake factory wired up in this test's env -> configured false.
    assert body["claude"]["configured"] is True
    assert body["gemini"]["configured"] is False


async def test_set_provider_key_marks_provider_configured(api):
    response = await api.put("/ai/settings/providers/openai/key", json={"api_key": "sk-test"})
    assert response.status_code == 200
    assert response.json()["configured"] is True

    listed = await api.get("/ai/settings/providers")
    row = next(r for r in listed.json() if r["id"] == "openai")
    assert row["configured"] is True


async def test_set_provider_key_never_echoes_the_key(api):
    response = await api.put("/ai/settings/providers/openai/key", json={"api_key": "sk-test"})
    assert "sk-test" not in response.text


async def test_clear_provider_key_reverts_to_not_configured(api):
    await api.put("/ai/settings/providers/openai/key", json={"api_key": "sk-test"})
    response = await api.delete("/ai/settings/providers/openai/key")
    assert response.status_code == 200
    assert response.json()["configured"] is False


async def test_set_provider_key_unknown_provider_422(api):
    response = await api.put(
        "/ai/settings/providers/not_a_real_provider/key", json={"api_key": "x"}
    )
    assert response.status_code == 422


async def test_set_provider_key_empty_body_422(api):
    response = await api.put("/ai/settings/providers/openai/key", json={"api_key": ""})
    assert response.status_code == 422
