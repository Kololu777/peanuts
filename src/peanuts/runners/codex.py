"""Subscription-authenticated OpenAI Codex CLI runner (`codex exec --json [resume <thread_id>]`)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from peanuts.models import ProviderName
from peanuts.runners.base import ProgressCallback, RunnerError, TurnInput, TurnOutput

_STREAM_BUFFER_LIMIT = 10 * 1024 * 1024  # 10 MB per JSONL line


def _truncate(text: str, max_len: int = 200) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"


def build_codex_command(
    binary: str,
    prompt: str,
    thread_id: str | None,
    *,
    sandbox: str | None = None,
    network_access: bool = True,
) -> list[str]:
    """Build non-interactive Codex CLI arguments."""
    cmd = [binary, "exec"]
    if sandbox:
        cmd.extend(["--sandbox", sandbox])
        if sandbox == "workspace-write" and network_access:
            cmd.extend(["-c", "sandbox_workspace_write.network_access=true"])
    cmd.append("--json")
    if thread_id:
        cmd.extend(["resume", thread_id, prompt])
    else:
        cmd.append(prompt)
    return cmd


def extract_codex_progress(event: dict[str, Any]) -> list[str]:
    """Extract human-readable progress summaries from a single Codex JSONL event."""
    ev_type = event.get("type")
    if ev_type not in {"item.started", "item.completed"}:
        return []
    item = event.get("item")
    if not isinstance(item, dict):
        return []
    item_type = str(item.get("type") or "")
    if item_type == "reasoning":
        txt = item.get("text")
        if isinstance(txt, str) and txt.strip():
            return [f"Thinking: {_truncate(txt, 160)}"]
    elif item_type == "command_execution":
        cmd = item.get("command")
        status = "Running" if ev_type == "item.started" else "Ran"
        if isinstance(cmd, str) and cmd.strip():
            return [f"{status} command: {_truncate(cmd, 150)}"]
    elif item_type == "file_change":
        changes = item.get("changes")
        if isinstance(changes, list) and changes:
            paths = [str(c.get("path", "")) for c in changes if isinstance(c, dict) and c.get("path")]
            if paths:
                return [f"File change: {_truncate(', '.join(paths), 150)}"]
    return []


def parse_codex_jsonl(stdout: str, fallback_thread_id: str | None = None) -> TurnOutput:
    """Parse JSONL event stream emitted by `codex exec --json`."""
    thread_id: str | None = fallback_thread_id
    messages: list[str] = []
    errors: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        ev_type = event.get("type")
        if ev_type == "thread.started":
            tid = event.get("thread_id")
            if isinstance(tid, str) and tid:
                thread_id = tid
        elif ev_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    messages.append(text.strip())
        elif ev_type in {"error", "turn.failed"}:
            err = event.get("message") or event.get("error") or str(event)
            errors.append(str(err))

    if errors and not messages:
        raise RunnerError(f"Codex CLI reported error: {'; '.join(errors)}")

    if messages:
        return TurnOutput(
            text="\n\n".join(messages),
            provider_session_id=thread_id,
            raw_stdout=stdout,
        )

    fallback_text = stdout.strip()
    if not fallback_text:
        raise RunnerError("Codex CLI returned empty stdout")
    return TurnOutput(text=fallback_text, provider_session_id=thread_id, raw_stdout=stdout)


class CodexRunner:
    """Runs turns via the local subscription-authenticated `codex` CLI with live progress streaming."""

    provider: ProviderName = "codex"

    def __init__(self, binary: str = "codex") -> None:
        self.binary = binary

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        cmd = build_codex_command(
            self.binary,
            turn.prompt,
            turn.agent.provider_session_id,
            sandbox=turn.codex_sandbox,
            network_access=turn.codex_network_access,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(turn.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_BUFFER_LIMIT,
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Codex CLI binary '{self.binary}' was not found in PATH. "
                "Install OpenAI Codex CLI and log in with your subscription."
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
                                    for progress_line in extract_codex_progress(ev):
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
            raise RunnerError(f"Codex CLI timed out after {turn.timeout_seconds}s") from exc

        stderr_b = await stderr_task if stderr_task is not None else b""
        stdout = "".join(stdout_lines)
        stderr = stderr_b.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            detail = stderr or stdout.strip() or f"exit code {proc.returncode}"
            raise RunnerError(f"Codex CLI failed ({proc.returncode}): {detail}")

        return parse_codex_jsonl(stdout, fallback_thread_id=turn.agent.provider_session_id)
