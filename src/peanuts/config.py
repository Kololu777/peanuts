"""Configuration models for Peanuts."""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from peanuts.models import ProviderAvailability, ProviderName


class PeanutsConfig(BaseModel):
    """Runtime configuration for the Peanuts server and orchestrator."""

    db_path: Path = Field(default=Path(".peanuts/peanuts.db"))
    cwd: Path = Field(default_factory=Path.cwd)
    root_provider: ProviderName | None = Field(default=None)
    default_child_provider: ProviderName | None = Field(default=None)
    max_depth: int = Field(default=2, ge=0, le=6)
    max_questions_per_turn: int = Field(default=3, ge=1, le=20)
    turn_timeout_seconds: float = Field(default=300.0, gt=0.0)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8765, ge=1, le=65535)
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )
    claude_bin: str = Field(default="claude")
    codex_bin: str = Field(default="codex")
    claude_permission_mode: str = Field(default="auto")
    codex_sandbox: str = Field(default="workspace-write")
    codex_network_access: bool = Field(default=True)
    enable_worktrees: bool = Field(default=True)
    auto_merge: bool = Field(default=True)

    def detect_providers(self) -> list[ProviderAvailability]:
        """Inspect installed subscription-authenticated CLI agents."""
        claude_path = shutil.which(self.claude_bin)
        codex_path = shutil.which(self.codex_bin)
        return [
            ProviderAvailability(name="claude", available=claude_path is not None, binary=claude_path),
            ProviderAvailability(name="codex", available=codex_path is not None, binary=codex_path),
            ProviderAvailability(name="mock", available=True, binary="builtin"),
        ]

    def resolve_root_provider(self) -> ProviderName:
        """Choose the root agent provider (explicit config, else first available CLI, else mock)."""
        if self.root_provider is not None:
            return self.root_provider
        for item in self.detect_providers():
            if item.available and item.name != "mock":
                return item.name
        return "mock"

    def resolve_child_provider(self) -> ProviderName:
        """Choose default child provider if a delegation omits one."""
        if self.default_child_provider is not None:
            return self.default_child_provider
        return self.resolve_root_provider()
