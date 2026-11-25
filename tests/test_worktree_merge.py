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


async def test_instruction_job_failures_still_dirty_and_clean_base_start(tmp_path: Path) -> None:
    """Covers TLA+ InstructionFailedOrStillDirty (runner error, leaked MERGE_HEAD, still dirty) + clean-base start."""
    from peanuts.runners.base import RunnerError

    mode = "clean_ok"

    class _InstructionScenarioRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            agent = turn.agent
            user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()
            if agent.role_kind == "integrator":
                if mode == "fail":
                    raise RunnerError("instruction runner failed")
                if mode == "leave_merge_head":
                    # Simulate a misbehaving instruction that leaves an unresolved conflict / MERGE_HEAD
                    _git(turn.cwd, "merge", "--no-commit", "--no-ff", "conflict-src")
                    return TurnOutput(text="left merge uncommitted", provider_session_id="int-mh")
                if mode == "still_dirty":
                    # Succeeds without cleaning human_wip.txt
                    return TurnOutput(text="checked git status only", provider_session_id="int-sd")
                # clean_ok: create and commit a file cleanly on base branch
                (turn.cwd / "housekeeping.txt").write_text("clean instruction\n", encoding="utf-8")
                _git(turn.cwd, "add", "housekeeping.txt")
                _git(turn.cwd, "commit", "-m", "housekeeping commit")
                return TurnOutput(text="housekeeping done", provider_session_id="int-ok")

            if agent.role_kind == "root":
                if user_input.startswith("Integrator ") or user_input.startswith("Child agent update"):
                    return TurnOutput(text="root noted", provider_session_id="root-1")
                if user_input == "run_instruction":
                    payload = {
                        "action": "integrate",
                        "mode": "instruction",
                        "instruction": "perform git housekeeping",
                        "message": "dispatched instruction",
                    }
                    return TurnOutput(
                        text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="root-1"
                    )
                if user_input == "spawn_worker":
                    payload = {
                        "action": "delegate",
                        "tasks": [{"theme": "w1", "provider": "mock", "prompt": "edit-w1"}],
                    }
                    return TurnOutput(
                        text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="root-1"
                    )
            (turn.cwd / "w1.txt").write_text("from w1\n", encoding="utf-8")
            return TurnOutput(text="w1 edited", provider_session_id="w1-s")

    # 1. Clean-base IDLE -> RUNNING_INSTRUCTION -> MERGED
    repo = tmp_path / "repo_inst_cases"
    _init_git_repo(repo, branch="main")
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=True,
    )
    orch = Orchestrator(cfg, runners={"mock": _InstructionScenarioRunner()})
    try:
        assert orch.worktree_mgr is not None
        mode = "clean_ok"
        await orch.send_human_message("run_instruction")
        await orch.wait_until_quiescent()
        assert (repo / "housekeeping.txt").read_text(encoding="utf-8") == "clean instruction\n"
        assert orch.repo.list_merge_jobs()[-1].status == MergeJobStatus.MERGED
        check_merge_queue_invariants(
            orch.repo.list_agents(),
            orch.repo.list_merge_jobs(),
            base_dirty=orch.worktree_mgr.is_dirty(repo),
            base_merge_in_progress=orch.worktree_mgr.is_merge_in_progress(),
        )

        # 2. RunnerError during RUNNING_INSTRUCTION -> ABORTED, integrator returns IDLE
        mode = "fail"
        await orch.send_human_message("run_instruction")
        await orch.wait_until_quiescent()
        failed_job = orch.repo.list_merge_jobs()[-1]
        assert failed_job.status == MergeJobStatus.ABORTED
        assert failed_job.error is not None and "instruction runner failed" in failed_job.error
        integrator = orch.repo.get_agent("integrator")
        assert integrator is not None and integrator.status.value == "IDLE"
        check_merge_queue_invariants(
            orch.repo.list_agents(),
            orch.repo.list_merge_jobs(),
            base_dirty=orch.worktree_mgr.is_dirty(repo),
            base_merge_in_progress=orch.worktree_mgr.is_merge_in_progress(),
        )

        # 3. Instruction leaves MERGE_HEAD -> auto git merge --abort + ABORTED (CleanBaseWhenIntegratorIdle)
        _git(repo, "checkout", "-b", "conflict-src")
        (repo / "extra.txt").write_text("side\n", encoding="utf-8")
        _git(repo, "add", "extra.txt")
        _git(repo, "commit", "-m", "side commit")
        _git(repo, "checkout", "main")
        mode = "leave_merge_head"
        await orch.send_human_message("run_instruction")
        await orch.wait_until_quiescent()
        mh_job = orch.repo.list_merge_jobs()[-1]
        assert mh_job.status == MergeJobStatus.ABORTED
        assert not orch.worktree_mgr.is_merge_in_progress()
        check_merge_queue_invariants(
            orch.repo.list_agents(),
            orch.repo.list_merge_jobs(),
            base_dirty=orch.worktree_mgr.is_dirty(repo),
            base_merge_in_progress=orch.worktree_mgr.is_merge_in_progress(),
        )

        # 4. Instruction succeeds without cleaning dirty base -> paused BRANCH_MERGE stays WAITING_CLEAN_BASE
        (repo / "human_wip.txt").write_text("still dirty\n", encoding="utf-8")
        await orch.send_human_message("spawn_worker")
        await orch.wait_until_quiescent()
        branch_job = next(j for j in orch.repo.list_merge_jobs() if j.kind == "branch_merge")
        assert branch_job.status == MergeJobStatus.WAITING_CLEAN_BASE

        mode = "still_dirty"
        await orch.send_human_message("run_instruction")
        await orch.wait_until_quiescent()
        branch_job_after = orch.repo.get_merge_job(branch_job.id)
        assert branch_job_after is not None and branch_job_after.status == MergeJobStatus.WAITING_CLEAN_BASE
        assert not (repo / "w1.txt").exists()
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


async def test_root_integrate_merge_resume_modes_and_rejection_guards(tmp_path: Path) -> None:
    """Covers IntegrateAction(mode='merge'), IntegrateAction(mode='resume'), non-root rejection, and empty instruction."""

    class _IntegrateModesRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            agent = turn.agent
            user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()
            if agent.role_kind == "root":
                if user_input.startswith("Child agent update") or user_input.startswith("Integrator "):
                    return TurnOutput(text="root ack", provider_session_id="r1")
                if user_input == "delegate_worker":
                    payload = {
                        "action": "delegate",
                        "tasks": [{"theme": "feat-x", "provider": "mock", "prompt": "do-edit"}],
                    }
                    return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
                if user_input == "delegate_bad_integrate":
                    payload = {
                        "action": "delegate",
                        "tasks": [{"theme": "rogue", "provider": "mock", "prompt": "try-integrate"}],
                    }
                    return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
                if user_input == "empty_instruction":
                    payload = {"action": "integrate", "mode": "instruction", "instruction": "   "}
                    return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
                if user_input.startswith("merge_worker:"):
                    worker_id = user_input.split(":", 1)[1]
                    payload = {"action": "integrate", "mode": "merge", "agent_id": worker_id}
                    return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
                if user_input == "resume_queue":
                    payload = {"action": "integrate", "mode": "resume"}
                    return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
                return TurnOutput(text="ok", provider_session_id="r1")

            if user_input == "try-integrate":
                payload = {"action": "integrate", "mode": "instruction", "instruction": "git status"}
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="rogue-s")
            (turn.cwd / "feat_x.txt").write_text("feature x\n", encoding="utf-8")
            return TurnOutput(text="edited feat_x", provider_session_id="fx-s")

    repo = tmp_path / "repo_modes"
    _init_git_repo(repo, branch="main")
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=False,
    )
    orch = Orchestrator(cfg, runners={"mock": _IntegrateModesRunner()})
    try:
        assert orch.worktree_mgr is not None
        # 1. Worker attempting IntegrateAction is rejected
        await orch.send_human_message("delegate_bad_integrate")
        await orch.wait_until_quiescent()
        assert orch.repo.list_merge_jobs() == []
        rogue_del = next(d for d in orch.repo.list_delegations() if d.theme == "rogue")
        assert rogue_del.result is not None and "integrate rejected" in rogue_del.result

        # 2. Root empty instruction is rejected
        await orch.send_human_message("empty_instruction")
        await orch.wait_until_quiescent()
        assert orch.repo.list_merge_jobs() == []

        # 3. Worker edits worktree; base is dirty -> Root triggers mode='merge' -> pauses WAITING_CLEAN_BASE
        await orch.send_human_message("delegate_worker")
        await orch.wait_until_quiescent()
        worker = orch.repo.find_child_by_theme("root", "feat-x")
        assert worker is not None

        (repo / "wip.txt").write_text("dirty\n", encoding="utf-8")
        await orch.send_human_message(f"merge_worker:{worker.id}")
        await orch.wait_until_quiescent()
        jobs = orch.repo.list_merge_jobs()
        assert len(jobs) == 1 and jobs[0].status == MergeJobStatus.WAITING_CLEAN_BASE
        assert len(orch.repo.list_questions(only_open=True)) == 1

        # 4. Human cleans base and Root emits IntegrateAction(mode='resume') -> supersedes question & merges!
        _git(repo, "add", "wip.txt")
        _git(repo, "commit", "-m", "clean wip")
        await orch.send_human_message("resume_queue")
        await orch.wait_until_quiescent()

        assert orch.repo.list_questions(only_open=True) == []
        assert (repo / "feat_x.txt").read_text(encoding="utf-8") == "feature x\n"
        assert orch.repo.list_merge_jobs()[0].status == MergeJobStatus.MERGED
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


async def test_resume_merge_queue_command_after_manual_clean_maintains_invariants(tmp_path: Path) -> None:
    """Covers /resume-merges (resume_merge_queue) after human cleans dirty base while Integrator was WAITING_HUMAN."""
    repo = tmp_path / "repo_resume_cmd"
    _init_git_repo(repo, branch="main")
    (repo / "human_wip.txt").write_text("dirty wip\n", encoding="utf-8")

    runner = _WorktreeEditingRunner()
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=True,
        max_questions_per_turn=1,
    )
    orch = Orchestrator(cfg, runners={"mock": runner})
    try:
        assert orch.worktree_mgr is not None
        await orch.send_human_message("two_files")
        await orch.wait_until_quiescent()

        assert len(orch.repo.list_questions(only_open=True)) == 1
        assert any(j.status == MergeJobStatus.WAITING_CLEAN_BASE for j in orch.repo.list_merge_jobs())

        # Human cleans working tree manually and triggers `/resume-merges` instead of `/answer ok`
        _git(repo, "add", "human_wip.txt")
        _git(repo, "commit", "-m", "clean wip")
        await orch.resume_merge_queue()
        await orch.wait_until_quiescent()

        assert orch.repo.list_questions(only_open=True) == []
        assert (repo / "auth.txt").read_text(encoding="utf-8") == "auth ok"
        assert (repo / "docs.txt").read_text(encoding="utf-8") == "docs ok"
        assert all(j.status == MergeJobStatus.MERGED for j in orch.repo.list_merge_jobs())

        # Second dirty-base cycle must still succeed (regression test: Integrator question_count resets on IDLE)
        (repo / "human_wip_2.txt").write_text("second dirty wip\n", encoding="utf-8")
        await orch.send_human_message("edit_shared")
        await orch.wait_until_quiescent()
        open_qs_2 = orch.repo.list_questions(only_open=True)
        assert len(open_qs_2) == 1 and open_qs_2[0].agent_id == "integrator"

        _git(repo, "add", "human_wip_2.txt")
        _git(repo, "commit", "-m", "clean wip 2")
        await orch.resume_merge_queue()
        await orch.wait_until_quiescent()

        assert orch.repo.list_questions(only_open=True) == []
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


async def test_crash_recovery_aborts_interrupted_merge_jobs_and_clears_merge_head(tmp_path: Path) -> None:
    """Crash recovery aborts MERGING/CONFLICT_RESOLVING/RUNNING_INSTRUCTION jobs and cleans MERGE_HEAD."""
    repo = tmp_path / "repo_crash_merge"
    _init_git_repo(repo, branch="main")
    cfg = PeanutsConfig(
        db_path=repo / ".peanuts" / "peanuts.db",
        cwd=repo,
        root_provider="mock",
        default_child_provider="mock",
        enable_worktrees=True,
        auto_merge=False,
    )
    orch1 = Orchestrator(cfg)
    assert orch1.worktree_mgr is not None
    # Create an active MERGING job in DB + a real uncommitted MERGE_HEAD in git
    _git(repo, "checkout", "-b", "side-branch")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "side.txt")
    _git(repo, "commit", "-m", "side")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-commit", "--no-ff", "side-branch")
    assert orch1.worktree_mgr.is_merge_in_progress()

    job, _ = orch1.repo.enqueue_merge_job(
        worker_agent_id="root",
        theme="side",
        source_branch="side-branch",
        target_branch="main",
    )
    orch1.repo.transition_merge_job(job.id, MergeJobStatus.MERGING)
    orch1.repo.close()  # Simulate abrupt crash without calling shutdown()

    orch2 = Orchestrator(cfg)
    try:
        assert orch2.worktree_mgr is not None
        assert not orch2.worktree_mgr.is_merge_in_progress()
        recovered = orch2.repo.get_merge_job(job.id)
        assert recovered is not None and recovered.status == MergeJobStatus.ABORTED
        check_merge_queue_invariants(
            orch2.repo.list_agents(),
            orch2.repo.list_merge_jobs(),
            base_dirty=orch2.worktree_mgr.is_dirty(repo),
            base_merge_in_progress=orch2.worktree_mgr.is_merge_in_progress(),
        )
    finally:
        await orch2.shutdown()


def test_single_integrator_writer_db_lock_and_duplicate_enqueue_dedup(tmp_path: Path) -> None:
    """Covers DB-level SingleIntegratorWriter exclusion and duplicate branch enqueue deduplication."""
    from peanuts.db import PeanutsRepository
    from peanuts.state_machine import StateTransitionError

    repo = PeanutsRepository(tmp_path / "mq_lock.db")
    try:
        repo.ensure_root_agent("mock")
        j1, _ = repo.enqueue_merge_job(
            worker_agent_id="root",
            theme="a",
            source_branch="peanuts/a-1",
            target_branch="main",
        )
        # Enqueueing the same source_branch while QUEUED deduplicates cleanly
        j1_dup, dup_evs = repo.enqueue_merge_job(
            worker_agent_id="root",
            theme="a",
            source_branch="peanuts/a-1",
            target_branch="main",
        )
        assert j1_dup.id == j1.id
        assert dup_evs == []

        j2, _ = repo.enqueue_instruction_job(instruction="git status", target_branch="main")
        repo.transition_merge_job(j1.id, MergeJobStatus.MERGING)

        # Second job cannot become active while j1 is MERGING
        with pytest.raises(StateTransitionError, match="SingleIntegratorWriter"):
            repo.transition_merge_job(j2.id, MergeJobStatus.RUNNING_INSTRUCTION)
    finally:
        repo.close()


def test_stateful_merge_queue_tla_model_exploration(tmp_path: Path) -> None:
    """Randomized deterministic exploration of all PeanutsMergeQueue.tla actions with per-step invariant checks."""
    import random

    from peanuts.db import PeanutsRepository
    from peanuts.models import AgentStatus

    rng = random.Random(20261126)
    repo = PeanutsRepository(tmp_path / "stateful_mq.db")
    try:
        repo.ensure_root_agent("mock")
        repo.ensure_integrator_agent("mock", worktree_path=str(tmp_path), branch_name="main")
        base_dirty = False
        base_merge_in_progress = False
        job_seq = 0

        def verify() -> None:
            check_invariants(
                repo.list_agents(),
                repo.list_delegations(),
                repo.list_questions(only_open=False),
                max_depth=2,
                max_questions_per_turn=2,
            )
            check_merge_queue_invariants(
                repo.list_agents(),
                repo.list_merge_jobs(),
                base_dirty=base_dirty,
                base_merge_in_progress=base_merge_in_progress,
            )

        verify()
        action_counts: dict[str, int] = {}

        for step in range(200):
            integrator = repo.get_agent("integrator")
            assert integrator is not None
            jobs = repo.list_merge_jobs()
            queued_branch = [j for j in jobs if j.kind == "branch_merge" and j.status == MergeJobStatus.QUEUED]
            queued_inst = [j for j in jobs if j.kind == "instruction" and j.status == MergeJobStatus.QUEUED]
            waiting_jobs = [j for j in jobs if j.status == MergeJobStatus.WAITING_CLEAN_BASE]
            merging_jobs = [j for j in jobs if j.status == MergeJobStatus.MERGING]
            conflict_jobs = [j for j in jobs if j.status == MergeJobStatus.CONFLICT_RESOLVING]
            inst_running = [j for j in jobs if j.status == MergeJobStatus.RUNNING_INSTRUCTION]

            candidates: list[str] = ["enqueue_branch", "enqueue_instruction"]
            if not base_dirty and integrator.status != AgentStatus.RUNNING and not base_merge_in_progress:
                candidates.append("human_edits_base")
            if base_dirty and integrator.status != AgentStatus.RUNNING:
                candidates.append("human_cleans_base")
            if queued_branch and base_dirty and integrator.status != AgentStatus.RUNNING:
                candidates.append("pause_dirty")
            if waiting_jobs and not base_dirty:
                candidates.append("resume_clean")
            if (
                queued_branch
                and not base_dirty
                and integrator.status == AgentStatus.IDLE
                and not base_merge_in_progress
            ):
                candidates.append("start_merge")
            if (
                queued_inst
                and integrator.status in {AgentStatus.IDLE, AgentStatus.WAITING_HUMAN}
                and not base_merge_in_progress
            ):
                candidates.append("start_instruction")
            if merging_jobs:
                candidates.extend(["merge_clean_success", "merge_hits_conflict"])
            if conflict_jobs:
                candidates.extend(["conflict_resolved", "conflict_aborted"])
            if inst_running:
                candidates.extend(["inst_success_clean", "inst_failed_or_dirty"])

            choice = rng.choice(candidates)
            action_counts[choice] = action_counts.get(choice, 0) + 1

            if choice == "enqueue_branch":
                job_seq += 1
                repo.enqueue_merge_job(
                    worker_agent_id="root",
                    theme=f"w{job_seq}",
                    source_branch=f"peanuts/w-{job_seq}",
                    target_branch="main",
                )
            elif choice == "enqueue_instruction":
                repo.enqueue_instruction_job(instruction=f"inst-{step}", target_branch="main")
            elif choice == "human_edits_base":
                base_dirty = True
            elif choice == "human_cleans_base":
                base_dirty = False
                repo.supersede_integrator_dirty_questions()
            elif choice == "pause_dirty":
                j = rng.choice(queued_branch)
                repo.transition_merge_job(j.id, MergeJobStatus.WAITING_CLEAN_BASE)
                open_int_qs = [q for q in repo.list_questions(only_open=True) if q.agent_id == "integrator"]
                if not open_int_qs and integrator.status == AgentStatus.IDLE:
                    repo.set_integrator_status(AgentStatus.RUNNING)
                    repo.ask_human("integrator", "dirty base?", max_questions_per_turn=2)
            elif choice == "resume_clean":
                repo.resume_waiting_merge_jobs()
            elif choice == "start_merge":
                j = rng.choice(queued_branch)
                repo.set_integrator_status(AgentStatus.RUNNING)
                repo.transition_merge_job(j.id, MergeJobStatus.MERGING)
                base_merge_in_progress = True
            elif choice == "start_instruction":
                j = rng.choice(queued_inst)
                repo.supersede_integrator_dirty_questions()
                repo.set_integrator_status(AgentStatus.RUNNING)
                repo.transition_merge_job(j.id, MergeJobStatus.RUNNING_INSTRUCTION)
            elif choice == "merge_clean_success":
                j = merging_jobs[0]
                repo.transition_merge_job(j.id, MergeJobStatus.MERGED)
                repo.set_integrator_status(AgentStatus.IDLE)
                base_merge_in_progress = False
            elif choice == "merge_hits_conflict":
                j = merging_jobs[0]
                repo.transition_merge_job(j.id, MergeJobStatus.CONFLICT_RESOLVING, conflicted_files=["shared.txt"])
            elif choice == "conflict_resolved":
                j = conflict_jobs[0]
                repo.transition_merge_job(j.id, MergeJobStatus.MERGED, conflicted_files=[])
                repo.set_integrator_status(AgentStatus.IDLE)
                base_merge_in_progress = False
            elif choice == "conflict_aborted":
                j = conflict_jobs[0]
                repo.transition_merge_job(j.id, MergeJobStatus.ABORTED, error="aborted")
                repo.set_integrator_status(AgentStatus.IDLE)
                base_merge_in_progress = False
            elif choice == "inst_success_clean":
                j = inst_running[0]
                repo.transition_merge_job(j.id, MergeJobStatus.MERGED)
                repo.set_integrator_status(AgentStatus.IDLE)
                base_dirty = False
            elif choice == "inst_failed_or_dirty":
                j = inst_running[0]
                repo.transition_merge_job(j.id, MergeJobStatus.ABORTED, error="failed")
                repo.set_integrator_status(AgentStatus.IDLE)
                base_dirty = True

            verify()

        for expected in (
            "pause_dirty",
            "resume_clean",
            "start_merge",
            "start_instruction",
            "merge_clean_success",
            "merge_hits_conflict",
            "conflict_resolved",
            "conflict_aborted",
            "inst_success_clean",
            "inst_failed_or_dirty",
        ):
            assert action_counts.get(expected, 0) > 0, f"MergeQueue action {expected} was never exercised"
    finally:
        repo.close()
