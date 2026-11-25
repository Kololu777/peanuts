"""Integration tests for BranchLock (flock), isolated git worktrees, and Single-Writer Integrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from peanuts.config import PeanutsConfig
from peanuts.git_ops import BranchLock, BranchLockError, detect_git_context
from peanuts.models import MergeJobStatus, ProviderName
from peanuts.orchestrator import Orchestrator
from peanuts.runners.base import ProgressCallback, TurnInput, TurnOutput
from peanuts.state_machine import check_invariants, check_merge_queue_invariants


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_git_repo(repo: Path, branch: str = "feat/demo") -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", branch)
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo / "shared.txt").write_text("line 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial commit")


def test_branch_lock_rejects_same_branch_and_allows_different_branch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, branch="main")

    # Symlink to same repo must still collide on the same (git_common_dir, branch)
    symlink_repo = tmp_path / "repo-link"
    symlink_repo.symlink_to(repo)

    lock1 = BranchLock(detect_git_context(repo))
    lock1.acquire(server_url="http://127.0.0.1:8765")
    try:
        lock2 = BranchLock(detect_git_context(symlink_repo))
        with pytest.raises(BranchLockError, match="already running on 'main'"):
            lock2.acquire()

        # A separate worktree on a DIFFERENT branch must be allowed
        other_wt = tmp_path / "other-wt"
        _git(repo, "worktree", "add", "-b", "feat/other", str(other_wt), "main")
        lock_other = BranchLock(detect_git_context(other_wt))
        lock_other.acquire(server_url="http://127.0.0.1:8766")
        lock_other.release()
    finally:
        lock1.release()

    # Once released, acquiring again succeeds
    lock3 = BranchLock(detect_git_context(repo))
    lock3.acquire()
    lock3.release()


class _WorktreeEditingRunner:
    """Deterministic runner that edits files inside each worker's isolated git worktree."""

    provider: ProviderName = "mock"

    def __init__(self, *, resolve_conflicts: bool = True) -> None:
        self.resolve_conflicts = resolve_conflicts
        self.seen_cwds: dict[str, Path] = {}

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        agent = turn.agent
        self.seen_cwds[agent.id] = turn.cwd
        user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()

        # Dedicated Integrator turn: either conflict resolution OR root-dispatched instruction
        if agent.role_kind == "integrator":
            if "Stage and commit wip" in turn.prompt:
                _git(turn.cwd, "add", "human_wip.txt")
                _git(turn.cwd, "commit", "-m", "commit wip via integrator instruction")
                return TurnOutput(text="Committed human_wip.txt cleanly.", provider_session_id="int-1")
            if not self.resolve_conflicts:
                return TurnOutput(text="Unable to resolve conflict automatically.", provider_session_id="int-1")
            target = turn.cwd / "shared.txt"
            target.write_text("line 1\nresolved cleanly by integrator\n", encoding="utf-8")
            _git(turn.cwd, "add", "shared.txt")
            return TurnOutput(text="Resolved shared.txt cleanly.", provider_session_id="int-1")

        # Root mediator delegates themes or dispatches integrate actions
        if agent.role_kind == "root":
            if user_input.startswith("Child agent update") or user_input.startswith("Integrator "):
                return TurnOutput(
                    text=f"```peanuts-action\n{json.dumps({'action': 'reply', 'message': user_input})}\n```",
                    provider_session_id="root-1",
                )
            if "commit_wip_via_integrate" in user_input:
                payload = {
                    "action": "integrate",
                    "mode": "instruction",
                    "instruction": "Stage and commit wip files on base branch",
                    "message": "Dispatched commit instruction to Integrator.",
                }
                return TurnOutput(
                    text=f"```peanuts-action\n{json.dumps(payload)}\n```",
                    provider_session_id="root-1",
                )
            if "two_files" in user_input:
                payload = {
                    "action": "delegate",
                    "message": "Splitting auth & docs into isolated worktrees",
                    "tasks": [
                        {"theme": "auth", "provider": "mock", "prompt": "write_file:auth.txt:auth ok"},
                        {"theme": "docs", "provider": "mock", "prompt": "write_file:docs.txt:docs ok"},
                    ],
                }
                return TurnOutput(
                    text=f"```peanuts-action\n{json.dumps(payload)}\n```",
                    provider_session_id="root-1",
                )
            if "edit_shared" in user_input:
                payload = {
                    "action": "delegate",
                    "message": "Editing shared.txt in child worktree",
                    "tasks": [
                        {"theme": "shared-edit", "provider": "mock", "prompt": "write_file:shared.txt:child version\n"},
                    ],
                }
                return TurnOutput(
                    text=f"```peanuts-action\n{json.dumps(payload)}\n```",
                    provider_session_id="root-1",
                )
            return TurnOutput(text="root ok", provider_session_id="root-1")

        # Worker edits a file inside its isolated worktree cwd
        if user_input.startswith("write_file:"):
            _, filename, content = user_input.split(":", 2)
            (turn.cwd / filename).write_text(content, encoding="utf-8")
            if on_progress is not None:
                await on_progress(f"Edited {filename} in {turn.cwd.name}")
            return TurnOutput(text=f"Updated {filename}", provider_session_id=f"w-{agent.id}")

        return TurnOutput(text="done", provider_session_id=f"w-{agent.id}")


async def test_parallel_worktrees_auto_commit_and_single_integrator_clean_merge(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, branch="feat/demo")

    runner = _WorktreeEditingRunner()
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=True,
    )
    orch = Orchestrator(cfg, runners={"mock": runner})
    try:
        assert orch.worktree_mgr is not None
        assert orch.git_ctx.base_branch == "feat/demo"

        await orch.send_human_message("two_files")
        await orch.wait_until_quiescent()

        # Both files created in separate child worktrees were merged into parent's feat/demo branch!
        assert (repo / "auth.txt").read_text(encoding="utf-8") == "auth ok"
        assert (repo / "docs.txt").read_text(encoding="utf-8") == "docs ok"

        workers = [a for a in orch.repo.list_agents() if a.role_kind == "worker"]
        assert len(workers) == 2
        assert workers[0].worktree_path != workers[1].worktree_path
        assert workers[0].branch_name != workers[1].branch_name
        assert runner.seen_cwds[workers[0].id] != repo
        assert runner.seen_cwds[workers[1].id] != repo

        jobs = orch.repo.list_merge_jobs()
        assert len(jobs) == 2
        assert all(j.status == MergeJobStatus.MERGED for j in jobs)

        check_invariants(
            orch.repo.list_agents(),
            orch.repo.list_delegations(),
            orch.repo.list_questions(only_open=False),
            max_depth=cfg.max_depth,
            max_questions_per_turn=cfg.max_questions_per_turn,
        )
        check_merge_queue_invariants(
            orch.repo.list_agents(),
            jobs,
            base_dirty=orch.worktree_mgr.is_dirty(repo),
            base_merge_in_progress=orch.worktree_mgr.is_merge_in_progress(),
        )
    finally:
        await orch.shutdown()


async def test_integrator_pauses_on_dirty_base_and_resumes_after_clean(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo, branch="feat/dirty")

    # Human leaves uncommitted edits on parent's branch
    (repo / "human_wip.txt").write_text("do not overwrite me\n", encoding="utf-8")

    runner = _WorktreeEditingRunner()
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=True,
    )
    orch = Orchestrator(cfg, runners={"mock": runner})
    try:
        await orch.send_human_message("two_files")
        await orch.wait_until_quiescent()

        # Merge jobs MUST pause in WAITING_CLEAN_BASE and ask human instead of touching dirty tree
        jobs = orch.repo.list_merge_jobs()
        assert len(jobs) == 2
        assert any(j.status == MergeJobStatus.WAITING_CLEAN_BASE for j in jobs)
        open_qs = orch.repo.list_questions(only_open=True)
        assert len(open_qs) == 1
        assert open_qs[0].agent_id == "integrator"
        assert not (repo / "auth.txt").exists()

        # Human commits their WIP and answers `/answer ok`
        _git(repo, "add", "human_wip.txt")
        _git(repo, "commit", "-m", "human wip")
        await orch.answer_question(open_qs[0].id, "ok")
        await orch.wait_until_quiescent()

        # Both queued child branches now merge cleanly!
        assert (repo / "auth.txt").read_text(encoding="utf-8") == "auth ok"
        assert (repo / "docs.txt").read_text(encoding="utf-8") == "docs ok"
        assert all(j.status == MergeJobStatus.MERGED for j in orch.repo.list_merge_jobs())
    finally:
        await orch.shutdown()


async def test_integrator_ai_resolves_conflict_and_aborts_cleanly_when_unresolved(tmp_path: Path) -> None:
    # Case 1: Integrator AI resolves conflict successfully
    repo1 = tmp_path / "repo_resolve"
    _init_git_repo(repo1, branch="main")
    runner1 = _WorktreeEditingRunner(resolve_conflicts=True)
    cfg1 = PeanutsConfig(
        db_path=repo1 / ".peanuts" / "peanuts.db",
        cwd=repo1,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=False,  # Hold merge until we create a conflicting commit on main
    )
    orch1 = Orchestrator(cfg1, runners={"mock": runner1})
    try:
        await orch1.send_human_message("edit_shared")
        await orch1.wait_until_quiescent()
        worker = next(a for a in orch1.repo.list_agents() if a.role_kind == "worker")

        # Create conflicting commit on main
        (repo1 / "shared.txt").write_text("main conflicting version\n", encoding="utf-8")
        _git(repo1, "add", "shared.txt")
        _git(repo1, "commit", "-m", "main edit")

        await orch1.enqueue_merge_for_agent(worker.id)
        await orch1.wait_until_quiescent()

        jobs1 = orch1.repo.list_merge_jobs()
        assert len(jobs1) == 1
        assert jobs1[0].status == MergeJobStatus.MERGED
        assert "resolved cleanly by integrator" in (repo1 / "shared.txt").read_text(encoding="utf-8")
        assert orch1.worktree_mgr is not None and not orch1.worktree_mgr.is_merge_in_progress()
    finally:
        await orch1.shutdown()

    # Case 2: Integrator cannot resolve -> runs `git merge --abort`, base branch remains 100% clean
    repo2 = tmp_path / "repo_abort"
    _init_git_repo(repo2, branch="main")
    runner2 = _WorktreeEditingRunner(resolve_conflicts=False)
    cfg2 = PeanutsConfig(
        db_path=repo2 / ".peanuts" / "peanuts.db",
        cwd=repo2,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=False,
    )
    orch2 = Orchestrator(cfg2, runners={"mock": runner2})
    try:
        await orch2.send_human_message("edit_shared")
        await orch2.wait_until_quiescent()
        worker2 = next(a for a in orch2.repo.list_agents() if a.role_kind == "worker")

        (repo2 / "shared.txt").write_text("main conflicting version\n", encoding="utf-8")
        _git(repo2, "add", "shared.txt")
        _git(repo2, "commit", "-m", "main edit")

        await orch2.enqueue_merge_for_agent(worker2.id)
        await orch2.wait_until_quiescent()

        jobs2 = orch2.repo.list_merge_jobs()
        assert len(jobs2) == 1
        assert jobs2[0].status == MergeJobStatus.ABORTED
        assert orch2.worktree_mgr is not None
        assert not orch2.worktree_mgr.is_merge_in_progress()
        assert not orch2.worktree_mgr.is_dirty(repo2)
        assert (repo2 / "shared.txt").read_text(encoding="utf-8") == "main conflicting version\n"
    finally:
        await orch2.shutdown()


async def test_root_integrate_instruction_cleans_dirty_base_supersedes_question_and_resumes_merge(
    tmp_path: Path,
) -> None:
    """Tests the exact TLA+-verified path:
    dirty base pauses BRANCH_MERGE (opening WAITING_HUMAN question) ->
    root emits `integrate(mode='instruction')` ->
    Integrator atomically supersedes stale question, commits WIP, and auto-resumes paused merges.
    """
    repo = tmp_path / "repo_inst"
    _init_git_repo(repo, branch="main")
    (repo / "human_wip.txt").write_text("initial wip\n", encoding="utf-8")

    runner = _WorktreeEditingRunner()
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=True,
    )
    orch = Orchestrator(cfg, runners={"mock": runner})
    try:
        # 1. Child work pauses on dirty base with an open Integrator question
        await orch.send_human_message("two_files")
        await orch.wait_until_quiescent()
        assert len(orch.repo.list_questions(only_open=True)) == 1
        assert any(j.status == MergeJobStatus.WAITING_CLEAN_BASE for j in orch.repo.list_merge_jobs())

        # 2. Human asks root to commit WIP via `integrate(mode="instruction")`
        await orch.send_human_message("commit_wip_via_integrate")
        await orch.wait_until_quiescent()

        # 3. Stale dirty-base question is closed, WIP is committed, and both paused child branches merged!
        assert orch.repo.list_questions(only_open=True) == []
        assert (repo / "auth.txt").read_text(encoding="utf-8") == "auth ok"
        assert (repo / "docs.txt").read_text(encoding="utf-8") == "docs ok"
        assert orch.worktree_mgr is not None
        assert not orch.worktree_mgr.is_dirty(repo)
        assert not orch.worktree_mgr.is_merge_in_progress()
        assert all(j.status == MergeJobStatus.MERGED for j in orch.repo.list_merge_jobs())

        check_invariants(
            orch.repo.list_agents(),
            orch.repo.list_delegations(),
            orch.repo.list_questions(only_open=False),
            max_depth=cfg.max_depth,
            max_questions_per_turn=cfg.max_questions_per_turn,
        )
        check_merge_queue_invariants(
            orch.repo.list_agents(),
            orch.repo.list_merge_jobs(),
            base_dirty=orch.worktree_mgr.is_dirty(repo),
            base_merge_in_progress=orch.worktree_mgr.is_merge_in_progress(),
        )
    finally:
        await orch.shutdown()
