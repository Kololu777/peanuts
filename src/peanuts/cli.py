"""CLI entrypoints for Peanuts (`pipx install` / `uvx` compatible)."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn

from peanuts.config import PeanutsConfig
from peanuts.git_ops import BranchLock, BranchLockError, detect_git_context
from peanuts.models import ProviderName
from peanuts.server import create_app
from peanuts.tui import PeanutsTUI

app = typer.Typer(
    name="peanuts",
    help="Parent-child CLI agent mediator (Claude Code / Codex) with isolated git worktrees, Integrator merge queue, FastAPI backend, and Textual TUI.",
    no_args_is_help=False,
)


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_for_health(url: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/api/v1/health", timeout=0.5)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise RuntimeError(f"Embedded Peanuts server did not start within {timeout}s at {url}")


def _build_config(
    *,
    db: Path,
    provider: ProviderName | None,
    child_provider: ProviderName | None,
    max_depth: int,
    max_questions: int,
    host: str,
    port: int,
    claude_permission_mode: str,
    codex_sandbox: str,
    codex_network_access: bool,
    enable_worktrees: bool,
    auto_merge: bool,
) -> PeanutsConfig:
    return PeanutsConfig(
        db_path=db,
        cwd=Path.cwd(),
        root_provider=provider,
        default_child_provider=child_provider,
        max_depth=max_depth,
        max_questions_per_turn=max_questions,
        host=host,
        port=port,
        claude_permission_mode=claude_permission_mode,
        codex_sandbox=codex_sandbox,
        codex_network_access=codex_network_access,
        enable_worktrees=enable_worktrees,
        auto_merge=auto_merge,
    )


def _acquire_branch_lock(cwd: Path, server_url: str | None = None) -> BranchLock:
    lock = BranchLock(detect_git_context(cwd))
    try:
        lock.acquire(server_url=server_url)
    except BranchLockError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    return lock


@app.callback(invoke_without_command=True)
def default_command(
    ctx: typer.Context,
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path")] = Path(".peanuts/peanuts.db"),
    provider: Annotated[ProviderName | None, typer.Option("--provider", help="Root agent provider (claude|codex|mock)")] = None,
    child_provider: Annotated[ProviderName | None, typer.Option("--child-provider", help="Default child provider")] = None,
    max_depth: Annotated[int, typer.Option("--max-depth", help="Max recursive delegation depth")] = 2,
    claude_permission_mode: Annotated[str, typer.Option("--claude-permission-mode", help="Claude CLI permission mode for workers/integrator")] = "auto",
    codex_sandbox: Annotated[str, typer.Option("--codex-sandbox", help="Codex CLI sandbox mode for workers/integrator")] = "workspace-write",
    worktrees: Annotated[bool, typer.Option("--worktrees/--no-worktrees", help="Isolate child workers in git worktrees")] = True,
    auto_merge: Annotated[bool, typer.Option("--auto-merge/--no-auto-merge", help="Auto-queue finished child branches to Integrator")] = True,
    server_url: Annotated[str | None, typer.Option("--server-url", help="Connect to an existing peanuts server")] = None,
) -> None:
    """Launch the TUI (default subcommand when running `peanuts`)."""
    if ctx.invoked_subcommand is not None:
        return
    tui(
        db=db,
        provider=provider,
        child_provider=child_provider,
        max_depth=max_depth,
        claude_permission_mode=claude_permission_mode,
        codex_sandbox=codex_sandbox,
        worktrees=worktrees,
        auto_merge=auto_merge,
        server_url=server_url,
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port")] = 8765,
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path")] = Path(".peanuts/peanuts.db"),
    provider: Annotated[ProviderName | None, typer.Option("--provider", help="Root agent provider")] = None,
    child_provider: Annotated[ProviderName | None, typer.Option("--child-provider", help="Default child provider")] = None,
    max_depth: Annotated[int, typer.Option("--max-depth", help="Max recursive delegation depth")] = 2,
    max_questions: Annotated[int, typer.Option("--max-questions", help="Max ask_human loops per turn")] = 3,
    claude_permission_mode: Annotated[str, typer.Option("--claude-permission-mode", help="Claude CLI permission mode for workers/integrator")] = "auto",
    codex_sandbox: Annotated[str, typer.Option("--codex-sandbox", help="Codex CLI sandbox mode for workers/integrator")] = "workspace-write",
    worktrees: Annotated[bool, typer.Option("--worktrees/--no-worktrees", help="Isolate child workers in git worktrees")] = True,
    auto_merge: Annotated[bool, typer.Option("--auto-merge/--no-auto-merge", help="Auto-queue finished child branches to Integrator")] = True,
) -> None:
    """Run the FastAPI REST + WebSocket backend (for TUI and Web+React)."""
    url = f"http://{host}:{port}"
    lock = _acquire_branch_lock(Path.cwd(), server_url=url)
    try:
        cfg = _build_config(
            db=db,
            provider=provider,
            child_provider=child_provider,
            max_depth=max_depth,
            max_questions=max_questions,
            host=host,
            port=port,
            claude_permission_mode=claude_permission_mode,
            codex_sandbox=codex_sandbox,
            codex_network_access=True,
            enable_worktrees=worktrees,
            auto_merge=auto_merge,
        )
        fastapi_app = create_app(cfg)
        typer.echo(f"Peanuts API server listening on {url} (OpenAPI: {url}/openapi.json)")
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    finally:
        lock.release()


@app.command()
def tui(
    db: Annotated[Path, typer.Option("--db", help="SQLite DB path")] = Path(".peanuts/peanuts.db"),
    provider: Annotated[ProviderName | None, typer.Option("--provider", help="Root agent provider (claude|codex|mock)")] = None,
    child_provider: Annotated[ProviderName | None, typer.Option("--child-provider", help="Default child provider")] = None,
    max_depth: Annotated[int, typer.Option("--max-depth", help="Max recursive delegation depth")] = 2,
    claude_permission_mode: Annotated[str, typer.Option("--claude-permission-mode", help="Claude CLI permission mode for workers/integrator")] = "auto",
    codex_sandbox: Annotated[str, typer.Option("--codex-sandbox", help="Codex CLI sandbox mode for workers/integrator")] = "workspace-write",
    worktrees: Annotated[bool, typer.Option("--worktrees/--no-worktrees", help="Isolate child workers in git worktrees")] = True,
    auto_merge: Annotated[bool, typer.Option("--auto-merge/--no-auto-merge", help="Auto-queue finished child branches to Integrator")] = True,
    server_url: Annotated[str | None, typer.Option("--server-url", help="Connect to an existing peanuts server")] = None,
) -> None:
    """Launch the Textual TUI (auto-starts an embedded FastAPI server if --server-url is omitted)."""
    if server_url is not None:
        PeanutsTUI(server_url=server_url).run()
        return

    host = "127.0.0.1"
    port = _find_free_port(host)
    url = f"http://{host}:{port}"
    lock = _acquire_branch_lock(Path.cwd(), server_url=url)
    cfg = _build_config(
        db=db,
        provider=provider,
        child_provider=child_provider,
        max_depth=max_depth,
        max_questions=3,
        host=host,
        port=port,
        claude_permission_mode=claude_permission_mode,
        codex_sandbox=codex_sandbox,
        codex_network_access=True,
        enable_worktrees=worktrees,
        auto_merge=auto_merge,
    )
    fastapi_app = create_app(cfg)
    uv_config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uv_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        _wait_for_health(url)
        PeanutsTUI(server_url=url).run()
    finally:
        server.should_exit = True
        thread.join(timeout=3.0)
        lock.release()


@app.command()
def providers() -> None:
    """Show installed subscription CLI providers (claude, codex, mock)."""
    cfg = PeanutsConfig()
    for item in cfg.detect_providers():
        status = "AVAILABLE" if item.available else "NOT FOUND"
        typer.echo(f"{item.name:<8} {status:<10} ({item.binary or '-'})")


@app.command()
def openapi(
    out: Annotated[Path | None, typer.Option("--out", help="Optional output file path for OpenAPI JSON")] = None,
) -> None:
    """Export the OpenAPI schema JSON (for React / openapi-typescript codegen)."""
    cfg = PeanutsConfig(db_path=Path(":memory:"), root_provider="mock", enable_worktrees=False)
    fastapi_app = create_app(cfg)
    schema = fastapi_app.openapi()
    text = json.dumps(schema, indent=2, ensure_ascii=False)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote OpenAPI schema to {out}")
    else:
        typer.echo(text)


def main() -> None:
    """Console script entrypoint for pipx / uvx / uv tool."""
    app()


if __name__ == "__main__":
    main()
