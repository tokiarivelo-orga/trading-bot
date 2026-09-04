"""ClaudeCodeAdapter — headless `claude -p` subprocess calls
(AI_PROVIDER_SETTINGS_PLAN.md §9.1). Flags asserted here match what was
verified against a real Claude Code install: `--tools ""` (not
`--allowed-tools`), `--strict-mcp-config` with no `--mcp-config`, and no
`--system-prompt` (it defeats the CLI's own prompt cache)."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.ai.adapters.claude_code import ClaudeCodeAdapter
from src.ai.ports.llm import LLMCallError, LLMMessage


class _FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.stdin_input: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:  # noqa: A002
        self.stdin_input = input
        return self._stdout, self._stderr


class _HangingProcess:
    """Never resolves `communicate()`, like a CLI call stuck past the
    timeout — used to verify the adapter kills it instead of leaking it."""

    def __init__(self) -> None:
        self.killed = False
        self.waited = False
        self.returncode: int | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:  # noqa: A002
        await asyncio.sleep(3600)
        raise AssertionError("should have been cancelled by the timeout")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or -9


def _patch_subprocess(monkeypatch, process: _FakeProcess, captured: list):
    async def _fake_exec(*args, **kwargs):
        captured.append(args)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


async def test_complete_returns_result_field(monkeypatch):
    payload = {"type": "result", "is_error": False, "result": "pong"}
    captured: list = []
    _patch_subprocess(monkeypatch, _FakeProcess(json.dumps(payload).encode()), captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet")
    result = await adapter.complete(LLMMessage(system="sys", user="say pong"))

    assert result == "pong"


async def test_complete_disables_tools_and_mcp_and_session_persistence(monkeypatch):
    payload = {"type": "result", "is_error": False, "result": "ok"}
    captured: list = []
    _patch_subprocess(monkeypatch, _FakeProcess(json.dumps(payload).encode()), captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet")
    await adapter.complete(LLMMessage(system="sys", user="user"))

    (args,) = captured
    assert args[0] == "claude"
    assert "-p" in args
    assert "--tools" in args
    assert args[args.index("--tools") + 1] == ""
    assert "--allowed-tools" not in args
    assert "--strict-mcp-config" in args
    assert "--mcp-config" not in args
    assert "--no-session-persistence" in args
    assert "--system-prompt" not in args
    assert "--model" in args
    assert args[args.index("--model") + 1] == "sonnet"


async def test_complete_folds_system_into_prompt_text(monkeypatch):
    payload = {"type": "result", "is_error": False, "result": "ok"}
    captured: list = []
    process = _FakeProcess(json.dumps(payload).encode())
    _patch_subprocess(monkeypatch, process, captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet")
    await adapter.complete(LLMMessage(system="SYS_TEXT", user="USER_TEXT"))

    assert process.stdin_input is not None
    prompt = process.stdin_input.decode()
    assert "SYS_TEXT" in prompt
    assert "USER_TEXT" in prompt


async def test_complete_sends_prompt_via_stdin_not_argv(monkeypatch):
    # A review prompt embedding a full strategy file plus 10 trades' JSON
    # snapshots is easily large enough to exceed the OS argv size limit
    # (`OSError: [Errno 7] Argument list too long`) if passed positionally —
    # it must go over stdin instead, uncapped by ARG_MAX.
    payload = {"type": "result", "is_error": False, "result": "ok"}
    captured: list = []
    process = _FakeProcess(json.dumps(payload).encode())
    _patch_subprocess(monkeypatch, process, captured)

    huge = "x" * 5_000_000
    adapter = ClaudeCodeAdapter("claude", "sonnet")
    await adapter.complete(LLMMessage(system=huge, user="u"))

    (args,) = captured
    assert not any(huge in str(a) for a in args)
    assert process.stdin_input is not None
    assert huge in process.stdin_input.decode()
    assert args[args.index("-p") + 1] == "--model"


async def test_nonzero_exit_raises(monkeypatch):
    captured: list = []
    _patch_subprocess(monkeypatch, _FakeProcess(b"", b"boom", returncode=1), captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet")
    with pytest.raises(LLMCallError, match="boom"):
        await adapter.complete(LLMMessage(system="s", user="u"))


async def test_is_error_result_raises(monkeypatch):
    payload = {"type": "result", "is_error": True, "result": ""}
    captured: list = []
    _patch_subprocess(monkeypatch, _FakeProcess(json.dumps(payload).encode()), captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet")
    with pytest.raises(LLMCallError, match="error result"):
        await adapter.complete(LLMMessage(system="s", user="u"))


async def test_timeout_kills_process_and_raises_llm_call_error(monkeypatch):
    process = _HangingProcess()
    captured: list = []
    _patch_subprocess(monkeypatch, process, captured)

    # A tiny timeout keeps the test fast; behavior under a real 480s default
    # is identical since asyncio.wait_for's cancellation path doesn't care
    # about the magnitude of the timeout.
    adapter = ClaudeCodeAdapter("claude", "sonnet", timeout_s=0.05)
    with pytest.raises(LLMCallError, match="timeout"):
        await adapter.complete(LLMMessage(system="s", user="u"))

    assert process.killed
    assert process.waited


async def test_timeout_s_defaults_to_480(monkeypatch):
    payload = {"type": "result", "is_error": False, "result": "ok"}
    captured: list = []
    _patch_subprocess(monkeypatch, _FakeProcess(json.dumps(payload).encode()), captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet")
    assert adapter._timeout_s == 480.0


async def test_extra_args_are_appended(monkeypatch):
    payload = {"type": "result", "is_error": False, "result": "ok"}
    captured: list = []
    _patch_subprocess(monkeypatch, _FakeProcess(json.dumps(payload).encode()), captured)

    adapter = ClaudeCodeAdapter("claude", "sonnet", extra_args="--agent reviewer")
    await adapter.complete(LLMMessage(system="s", user="u"))

    (args,) = captured
    assert "--agent" in args
    assert args[args.index("--agent") + 1] == "reviewer"
