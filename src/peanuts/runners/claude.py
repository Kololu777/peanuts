"""Subscription-authenticated Claude Code CLI runner (`claude -p --output-format stream-json --verbose`)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from peanuts.models import ProviderName
from peanuts.runners.base import ProgressCallback, RunnerError, TurnInput, TurnOutput

_STREAM_BUFFER_LIMIT = 10 * 1024 * 1024  # 10 MB per JSONL line (prevents LimitOverrunError on large tool outputs)


def _truncate(text: str, max_len: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"


def build_claude_command(
    binary: str,
    prompt: str,
    session_id: str | None,
    *,
    permission_mode: str | None = None,
) -> list[str]:
    """Build non-interactive streaming Claude Code CLI arguments."""
    cmd = [binary, "-p", prompt, "--output-format", "stream-json", "--verbose"]
    if permission_mode:
        cmd.extend(["--permission-mode", permission_mode])
    if session_id:
        cmd.extend(["--resume", session_id])
    return cmd


def extract_claude_progress(event: dict[str, Any]) -> list[str]:
    """Extract human-readable progress summaries from a single Claude stream-json event."""
    ev_type = event.get("type")
    lines: list[str] = []

    if ev_type == "assistant":
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    name = str(block.get("name") or "tool")
                    tool_input = block.get("input")
                    detail = ""
                    if isinstance(tool_input, dict):
                        for key in ("command", "file_path", "path", "pattern", "query", "description"):
                            val = tool_input.get(key)
                            if isinstance(val, str) and val.strip():
                                detail = f": {_truncate(val, 140)}"
                                break
                    lines.append(f"Tool {name}{detail}")
                elif btype == "thinking":
                    thinking = block.get("thinking")
                    if isinstance(thinking, str) and thinking.strip():
                        lines.append(f"Thinking: {_truncate(thinking, 160)}")
                elif btype == "text":
                    txt = block.get("text")
                    if isinstance(txt, str) and txt.strip() and "```peanuts-action" not in txt:
                        lines.append(_truncate(txt, 180))
    elif ev_type == "user":
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    is_err = bool(block.get("is_error"))
                    raw_res = block.get("content")
                    preview = _truncate(str(raw_res), 140) if raw_res else ""
                    prefix = "Tool error" if is_err else "Tool done"
                    lines.append(f"{prefix}: {preview}" if preview else prefix)

    return lines


def parse_claude_json(stdout: str, fallback_session_id: str | None = None) -> TurnOutput:
    """Parse either single-JSON (`--output-format json`) or JSONL (`--output-format stream-json`) stdout."""
    raw = stdout.strip()
    if not raw:
        raise RunnerError("Claude CLI returned empty stdout")

    session_id: str | None = fallback_session_id
    result_payload: dict[str, Any] | None = None
    assistant_texts: list[str] = []

    # Try full JSON object first (backwards compatible with --output-format json)
    try:
        parsed_full = json.loads(raw)
        if isinstance(parsed_full, dict):
            result_payload = parsed_full
    except json.JSONDecodeError:
        pass

    if result_payload is None:
        for line in raw.splitlines():
            line_s = line.strip()
            if not line_s.startswith("{"):
                continue
            try:
                event = json.loads(line_s)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            sid = event.get("session_id")
            if isinstance(sid, str) and sid:
                session_id = sid
            ev_type = event.get("type")
            if ev_type == "result":
                result_payload = event
            elif ev_type == "assistant":
                msg = event.get("message")
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            txt = block.get("text")
                            if isinstance(txt, str) and txt.strip():
                                assistant_texts.append(txt.strip())

    if result_payload is not None:
        if result_payload.get("is_error") is True:
            err_msg = str(result_payload.get("result") or result_payload.get("error") or "Claude CLI reported an error")
            raise RunnerError(err_msg)
        sid = result_payload.get("session_id")
        if isinstance(sid, str) and sid:
            session_id = sid
        res = result_payload.get("result")
        if isinstance(res, str) and res.strip():
            return TurnOutput(text=res, provider_session_id=session_id, raw_stdout=stdout)
        if assistant_texts:
            return TurnOutput(text="\n\n".join(assistant_texts), provider_session_id=session_id, raw_stdout=stdout)
        text = str(res) if res is not None else raw
        return TurnOutput(text=text, provider_session_id=session_id, raw_stdout=stdout)

    if assistant_texts:
        return TurnOutput(text="\n\n".join(assistant_texts), provider_session_id=session_id, raw_stdout=stdout)

    return TurnOutput(text=raw, provider_session_id=session_id, raw_stdout=stdout)


class ClaudeRunner:
    """Runs turns via the local subscription-authenticated `claude` CLI with live progress streaming."""

    provider: ProviderName = "claude"

    def __init__(self, binary: str = "claude") -> None:
        self.binary = binary

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        cmd = build_claude_command(
            self.binary,
            turn.prompt,
            turn.agent.provider_session_id,
            permission_mode=turn.claude_permission_mode,
        )
        # Unset CLAUDECODE so peanuts can spawn claude even if launched from a Claude Code terminal.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(turn.cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_BUFFER_LIMIT,
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Claude CLI binary '{self.binary}' was not found in PATH. "
                "Install Claude Code CLI and log in with your subscription."
            ) from exc

        stdout_lines: list[str] = []
        stderr_task = asyncio.create_task(proc.stderr.read()) if proc.stderr is not None else None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + turn.timeout_seconds

        try:
            if proc.stdout is not None:
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError
                    raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                    if not raw_line:
                        break
                    decoded = raw_line.decode("utf-8", errors="replace")
                    stdout_lines.append(decoded)
                    if on_progress is not None:
                        stripped = decoded.strip()
                        if stripped.startswith("{"):
                            try:
                                ev = json.loads(stripped)
                                if isinstance(ev, dict):
                                    for progress_line in extract_claude_progress(ev):
                                        await on_progress(progress_line)
                            except Exception:
                                pass
            remaining = max(0.1, deadline - loop.time())
            await asyncio.wait_for(proc.wait(), timeout=remaining)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            if stderr_task is not None:
                stderr_task.cancel()
            raise RunnerError(f"Claude CLI timed out after {turn.timeout_seconds}s") from exc

        stderr_b = await stderr_task if stderr_task is not None else b""
        stdout = "".join(stdout_lines)
        stderr = stderr_b.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            detail = stderr or stdout.strip() or f"exit code {proc.returncode}"
            raise RunnerError(f"Claude CLI failed ({proc.returncode}): {detail}")

        return parse_claude_json(stdout, fallback_session_id=turn.agent.provider_session_id)
