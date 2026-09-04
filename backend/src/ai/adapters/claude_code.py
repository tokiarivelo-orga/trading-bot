"""Claude Code adapter for `LLMPort` — headless `claude -p` subprocess calls
using the operator's local Claude Code login/subscription instead of
`TB_ANTHROPIC_API_KEY` (AI_PROVIDER_SETTINGS_PLAN.md §2.1, §9.1).

Flags are verified against a real Claude Code install, not guessed:
`--tools ""` (not `--allowed-tools`) fully disables the built-in tool set —
confirmed by a live call returning `"permission_denials":[]` with no
permission prompts — and `--strict-mcp-config` with no `--mcp-config`
guarantees zero MCP tools load. This keeps the call a pure text-in/text-out
boundary, never touching the filesystem/network/broker.

Do not add `--system-prompt`: a live comparison showed it invalidates the
CLI's own prompt cache (11,830 fresh cache-creation tokens, $0.071) instead
of avoiding the ~8.5k-11.8k token fixed overhead every headless call pays,
so `message.system` is folded into the prompt text like every other
LLMMessage instead.
"""

from __future__ import annotations

import asyncio
import json
import shlex

from src.ai.ports.llm import LLMCallError, LLMMessage

# Fixed per-call overhead alone (see module docstring) can approach 180s
# under load, and a full `code_generation` task also has to emit up to
# ~8192 tokens of Python before the CLI exits — 180s was tuned against
# quick calls (e.g. `pdf_extraction`) and reliably timed out full code
# generation. 480s gives that headroom; `next.config.ts`'s `proxyTimeout`
# must stay above this value too, or the frontend dev proxy cuts the
# connection first.
_DEFAULT_TIMEOUT_S = 480.0


class ClaudeCodeAdapter:
    def __init__(
        self, binary: str, model: str, extra_args: str = "", timeout_s: float = _DEFAULT_TIMEOUT_S
    ) -> None:
        self._binary = binary
        self._model = model
        self._extra_args = shlex.split(extra_args) if extra_args else []
        self._timeout_s = timeout_s

    async def complete(self, message: LLMMessage, *, max_tokens: int = 4096) -> str:
        # Claude Code manages output length itself (no CLI flag maps to
        # `max_tokens`); callers that need a hard cap should keep prompting
        # for concise output, same as they already do for the other adapters.
        del max_tokens
        prompt = f"{message.system}\n\n{message.user}"
        # Fed via stdin, not as a positional argv element: a review prompt
        # embedding a full strategy file plus 10 trades' JSON snapshots
        # routinely exceeds the OS argv size limit, which
        # `asyncio.create_subprocess_exec` surfaces as `OSError: [Errno 7]
        # Argument list too long` — confirmed live via `claude -p` reading
        # a piped prompt with no positional prompt argument, per `-p`'s own
        # "useful for pipes" description in `claude --help`.
        proc = await asyncio.create_subprocess_exec(
            self._binary,
            "-p",
            "--model",
            self._model,
            "--output-format",
            "json",
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            *self._extra_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()), timeout=self._timeout_s
            )
        except TimeoutError:
            # asyncio.wait_for only cancels the *wait* — the subprocess keeps
            # running unless we kill it ourselves, otherwise every timeout
            # leaves an orphaned `claude` process behind.
            proc.kill()
            await proc.wait()
            raise LLMCallError(
                f"claude code call exceeded {self._timeout_s:.0f}s timeout and was killed"
            ) from None
        if proc.returncode != 0:
            raise LLMCallError(f"claude code exited {proc.returncode}: {stderr.decode()[:500]}")
        payload = json.loads(stdout)
        if payload.get("is_error"):
            raise LLMCallError(f"claude code returned an error result: {payload!r}")
        return payload["result"]
