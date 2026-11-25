"""Git repository detection, OS-level branch exclusion locks (flock), and worktree/merge operations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Literal


class GitError(RuntimeError):
    """Raised when a git operation fails."""


class BranchLockError(RuntimeError):
    """Raised when another peanuts instance already owns the lock for (repo, branch)."""

    def __init__(self, message: str, existing: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.existing = existing or {}


@dataclass(frozen=True)
class GitContext:
    """Resolved git metadata for the workspace where peanuts was launched."""

    is_git_repo: bool
    cwd: Path
    repo_root: Path | None = None
    git_common_dir: Path | None = None
    git_dir: Path | None = None
    base_branch: str | None = None

    @property
    def supports_worktrees(self) -> bool:
        return self.is_git_repo and self.repo_root is not None and self.base_branch is not None


@dataclass(frozen=True)
class MergeAttemptResult:
    outcome: Literal["merged", "conflict", "error"]
    conflicted_files: list[str] = field(default_factory=list)
    summary: str = ""


def _run_git(args: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return proc


def detect_git_context(cwd: Path) -> GitContext:
    """Inspect whether `cwd` is inside a git repository and resolve its canonical common dir + branch."""
    resolved_cwd = cwd.resolve()
    top = _run_git(["rev-parse", "--show-toplevel"], resolved_cwd, check=False)
    if top.returncode != 0:
        return GitContext(is_git_repo=False, cwd=resolved_cwd)

    repo_root = Path(top.stdout.strip()).resolve()
    common_raw = _run_git(["rev-parse", "--git-common-dir"], repo_root).stdout.strip()
    git_dir_raw = _run_git(["rev-parse", "--git-dir"], repo_root).stdout.strip()
    git_common_dir = (repo_root / common_raw).resolve()
    git_dir = (repo_root / git_dir_raw).resolve()

    branch_proc = _run_git(["symbolic-ref", "--short", "HEAD"], repo_root, check=False)
    base_branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else None
    if not base_branch:
        base_branch = None

    return GitContext(
        is_git_repo=True,
        cwd=resolved_cwd,
        repo_root=repo_root,
        git_common_dir=git_common_dir,
        git_dir=git_dir,
        base_branch=base_branch,
    )


def _slugify(text: str, max_len: int = 24) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip()).strip("-.").lower()
    if not slug:
        slug = "task"
    return slug[:max_len].rstrip("-.") or "task"


class BranchLock:
    """OS-level advisory exclusive lock preventing two peanuts instances on the same (repo, branch)."""

    def __init__(self, git_ctx: GitContext) -> None:
        self.git_ctx = git_ctx
        self.lock_path = self._resolve_lock_path(git_ctx)
        self._fp: IO[str] | None = None
        self._server_url: str | None = None

    @staticmethod
    def _resolve_lock_path(git_ctx: GitContext) -> Path:
        if git_ctx.is_git_repo and git_ctx.git_common_dir is not None:
            branch_key = git_ctx.base_branch or "__detached__"
            digest = hashlib.sha256(branch_key.encode("utf-8")).hexdigest()[:16]
            return git_ctx.git_common_dir / "peanuts-locks" / f"{digest}.lock"
        digest = hashlib.sha256(str(git_ctx.cwd).encode("utf-8")).hexdigest()[:16]
        return git_ctx.cwd / ".peanuts" / "locks" / f"{digest}.lock"

    def acquire(self, server_url: str | None = None) -> None:
        """Acquire non-blocking exclusive flock; raises BranchLockError if already held."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fp = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fp.seek(0)
            raw = fp.read().strip()
            fp.close()
            existing: dict[str, str] = {}
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        existing = {str(k): str(v) for k, v in parsed.items()}
                except json.JSONDecodeError:
                    pass
            branch_label = self.git_ctx.base_branch or str(self.git_ctx.cwd)
            pid = existing.get("pid", "unknown")
            url = existing.get("server_url")
            hint = f"\nAttach to it with: peanuts tui --server-url {url}" if url else ""
            raise BranchLockError(
                f"Another peanuts instance is already running on '{branch_label}' (pid={pid}).{hint}",
                existing=existing,
            ) from exc

        self._fp = fp
        self._server_url = server_url
        self._write_metadata()

    def update_server_url(self, server_url: str) -> None:
        self._server_url = server_url
        if self._fp is not None:
            self._write_metadata()

    def _write_metadata(self) -> None:
        if self._fp is None:
            return
        payload = {
            "pid": str(os.getpid()),
            "branch": self.git_ctx.base_branch or "__nogit__",
            "cwd": str(self.git_ctx.cwd),
            "server_url": self._server_url or "",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._fp.seek(0)
        self._fp.truncate()
        self._fp.write(json.dumps(payload, ensure_ascii=False))
        self._fp.flush()

    def release(self) -> None:
        if self._fp is not None:
            try:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
            finally:
                self._fp.close()
                self._fp = None


class WorktreeManager:
    """Manages isolated child git worktrees and Single-Writer merges into the parent's branch."""

    def __init__(self, git_ctx: GitContext) -> None:
        if (
            not git_ctx.supports_worktrees
            or git_ctx.repo_root is None
            or git_ctx.git_common_dir is None
            or git_ctx.base_branch is None
        ):
            raise GitError("WorktreeManager requires a git repository on a named branch")
        self.git_ctx = git_ctx
        self.repo_root: Path = git_ctx.repo_root
        self.base_branch: str = git_ctx.base_branch
        # Store worktrees inside .git/peanuts-worktrees so they never dirty `git status`
        self.worktrees_dir = git_ctx.git_common_dir / "peanuts-worktrees"

    def create_worktree(self, agent_id: str, theme: str) -> tuple[Path, str]:
        """Create an isolated git worktree and branch forked from the parent's base branch."""
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        short = agent_id.split("_")[-1][:8]
        branch_name = f"peanuts/{_slugify(theme)}-{short}"
        wt_path = self.worktrees_dir / agent_id

        if wt_path.exists():
            return wt_path, branch_name

        # If branch already exists from an earlier session, reuse it cleanly; else branch off base_branch
        exists_proc = _run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], self.repo_root, check=False)
        if exists_proc.returncode == 0:
            _run_git(["worktree", "add", str(wt_path), branch_name], self.repo_root)
        else:
            _run_git(["worktree", "add", "-b", branch_name, str(wt_path), self.base_branch], self.repo_root)
        return wt_path, branch_name

    def is_dirty(self, path: Path | None = None) -> bool:
        target = path or self.repo_root
        proc = _run_git(["status", "--porcelain", "--", ".", ":!.peanuts"], target)
        return bool(proc.stdout.strip())

    def has_commits_ahead(self, branch_name: str) -> bool:
        proc = _run_git(["rev-list", "--count", f"{self.base_branch}..{branch_name}"], self.repo_root, check=False)
        if proc.returncode != 0:
            return False
        try:
            return int(proc.stdout.strip()) > 0
        except ValueError:
            return False

    def auto_commit_worktree(self, worktree_path: Path, message: str) -> bool:
        """Stage and commit any uncommitted edits in a child worktree. Returns True if a commit was created."""
        if not worktree_path.exists():
            return False
        if not self.is_dirty(worktree_path):
            return False
        _run_git(["add", "-A", "--", ".", ":!.peanuts"], worktree_path)
        if not self.is_dirty(worktree_path):
            return False
        _run_git(
            [
                "-c",
                "user.name=Peanuts Agent",
                "-c",
                "user.email=peanuts@local",
                "commit",
                "-m",
                message,
            ],
            worktree_path,
        )
        return True

    def is_merge_in_progress(self) -> bool:
        if self.git_ctx.git_dir is None:
            return False
        return (self.git_ctx.git_dir / "MERGE_HEAD").exists()

    def list_conflicted_files(self) -> list[str]:
        proc = _run_git(["diff", "--name-only", "--diff-filter=U"], self.repo_root, check=False)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    def diff_stat(self, branch_name: str) -> str:
        proc = _run_git(["diff", "--stat", f"{self.base_branch}...{branch_name}"], self.repo_root, check=False)
        return proc.stdout.strip()

    def try_merge(self, branch_name: str, message: str) -> MergeAttemptResult:
        """Attempt `git merge --no-ff` of `branch_name` into the parent's current branch."""
        if self.is_dirty(self.repo_root):
            raise GitError("Refusing to merge into dirty base working tree")

        proc = _run_git(
            [
                "-c",
                "user.name=Peanuts Integrator",
                "-c",
                "user.email=peanuts@local",
                "merge",
                "--no-ff",
                "-m",
                message,
                branch_name,
            ],
            self.repo_root,
            check=False,
        )
        if proc.returncode == 0:
            return MergeAttemptResult(outcome="merged", summary=proc.stdout.strip() or f"Merged {branch_name}")

        conflicts = self.list_conflicted_files()
        if conflicts or self.is_merge_in_progress():
            return MergeAttemptResult(
                outcome="conflict",
                conflicted_files=conflicts,
                summary=proc.stdout.strip() or proc.stderr.strip(),
            )

        return MergeAttemptResult(
            outcome="error",
            summary=proc.stderr.strip() or proc.stdout.strip() or f"git merge exited {proc.returncode}",
        )

    def finalize_resolved_merge(self, message: str) -> bool:
        """Commit a resolved merge if no unmerged files remain. Returns True if finalized."""
        if not self.is_merge_in_progress():
            return True
        if self.list_conflicted_files():
            return False
        _run_git(["add", "-A"], self.repo_root)
        _run_git(
            [
                "-c",
                "user.name=Peanuts Integrator",
                "-c",
                "user.email=peanuts@local",
                "commit",
                "-m",
                message,
            ],
            self.repo_root,
        )
        return True

    def abort_merge(self) -> None:
        """Abort an in-progress merge to restore a clean parent working tree."""
        if self.is_merge_in_progress():
            _run_git(["merge", "--abort"], self.repo_root, check=False)
