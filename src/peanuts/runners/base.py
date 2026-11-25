"""Base abstractions for CLI and mock agent runners."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from peanuts.models import AgentRecord, ProviderName

ProgressCallback = Callable[[str], Awaitable[None]]


class RunnerError(RuntimeError):
    """Raised when an underlying CLI agent runner fails or times out."""


class TurnInput(BaseModel):
    """Input passed to an AgentRunner for a single CLI turn."""

    agent: AgentRecord
    prompt: str
    cwd: Path
    timeout_seconds: float = 300.0
    claude_permission_mode: str | None = None
    codex_sandbox: str | None = None
    codex_network_access: bool = True


class TurnOutput(BaseModel):
    """Output parsed from a single CLI agent turn."""

    text: str
    provider_session_id: str | None = None
    raw_stdout: str = ""


class AgentRunner(Protocol):
    """Pluggable runner interface for Claude CLI, Codex CLI, and Mock."""

    provider: ProviderName

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        """Execute a single non-interactive turn and optionally stream progress lines."""
        ...
