"""Integration tests for Orchestrator: multi-theme fan-out, recursion, Q&A, failures, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peanuts.config import PeanutsConfig
from peanuts.models import AgentStatus, DelegationStatus, MessageRole, ProviderName
from peanuts.orchestrator import Orchestrator
from peanuts.runners.base import ProgressCallback, TurnInput, TurnOutput
from peanuts.state_machine import check_invariants


def _assert_invariants(orch: Orchestrator) -> None:
    check_invariants(
        orch.repo.list_agents(),
        orch.repo.list_delegations(),
        orch.repo.list_questions(only_open=False),
        max_depth=orch.config.max_depth,
        max_questions_per_turn=orch.config.max_questions_per_turn,
    )


async def test_multi_theme_parallel_fanout_and_multiturn_reuse(tmp_path: Path) -> None:
    cfg = PeanutsConfig(
        db_path=tmp_path / "peanuts.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg)
    try:
        await orch.send_human_message("auth: fix jwt expiry; docs: update api guide")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        agents = orch.repo.list_agents()
        themes = {a.theme for a in agents}
        assert {"root", "auth", "docs"}.issubset(themes)
        assert all(a.status == AgentStatus.IDLE for a in agents)
        assert all(a.provider_session_id is not None for a in agents)

        delegations = orch.repo.list_delegations()
        assert len(delegations) == 2
        assert all(d.status == DelegationStatus.DONE for d in delegations)

        # Turn 2 on same theme 'auth' should reuse existing child agent & its persisted session_id
        auth_child_before = orch.repo.find_child_by_theme("root", "auth")
        assert auth_child_before is not None
        prev_session_id = auth_child_before.provider_session_id

        await orch.send_human_message("auth: also add refresh token test")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        auth_children = [a for a in orch.repo.list_children("root") if a.theme == "auth"]
        assert len(auth_children) == 1
        assert auth_children[0].id == auth_child_before.id
        assert auth_children[0].provider_session_id == prev_session_id
        # Child agent recorded live progress messages viewable when clicking that child in the UI
        child_msgs = orch.repo.list_messages(agent_id=auth_children[0].id)
        assert any(m.role == MessageRole.PROGRESS for m in child_msgs)
    finally:
        await orch.shutdown()


async def test_recursive_child_to_grandchild_delegation(tmp_path: Path) -> None:
    cfg = PeanutsConfig(
        db_path=tmp_path / "peanuts.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg)
    try:
        await orch.send_human_message("recursive: implement feature and verify with subagent")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        agents = orch.repo.list_agents()
        depths = sorted(a.depth for a in agents)
        assert depths == [0, 1, 2]
        assert all(a.status == AgentStatus.IDLE for a in agents)
        assert all(d.status == DelegationStatus.DONE for d in orch.repo.list_delegations())
    finally:
        await orch.shutdown()


async def test_child_ask_human_mediation_loop(tmp_path: Path) -> None:
    cfg = PeanutsConfig(
        db_path=tmp_path / "peanuts.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg)
    try:
        await orch.send_human_message("db: 確認 マイグレーション実行")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        open_qs = orch.repo.list_questions(only_open=True)
        assert len(open_qs) == 1
        q = open_qs[0]
        assert q.theme == "db"

        root = orch.repo.get_root_agent()
        # Non-blocking root stays IDLE even while child waits for human clarification
        assert root.status == AgentStatus.IDLE
        child = orch.repo.get_agent(q.agent_id)
        assert child is not None and child.status == AgentStatus.WAITING_HUMAN

        # Human answers -> child resumes and completes -> parent wakes and synthesizes
        await orch.answer_question(q.id, "はい、進めてください")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        assert orch.repo.list_questions(only_open=True) == []
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE
        child_after = orch.repo.get_agent(q.agent_id)
        assert child_after is not None and child_after.status == AgentStatus.IDLE
    finally:
        await orch.shutdown()


async def test_child_failure_unblocks_parent_and_allows_retry(tmp_path: Path) -> None:
    cfg = PeanutsConfig(
        db_path=tmp_path / "peanuts.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
    )
    orch = Orchestrator(cfg)
    try:
        await orch.send_human_message("broken: force_fail test")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        # Parent must NOT stay stuck in WAITING_CHILD when child fails
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE
        children = orch.repo.list_children("root")
        assert len(children) == 1
        assert children[0].status == AgentStatus.FAILED

        # Retrying resets child FAILED -> IDLE
        retried = await orch.retry_agent(children[0].id)
        assert retried.status == AgentStatus.IDLE
        _assert_invariants(orch)
    finally:
        await orch.shutdown()


class _AlwaysAskRunner:
    """Adversarial runner that always emits ask_human to test TLC's MaxQuestionsPerTurn bound."""

    provider: ProviderName = "mock"

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        payload = {"action": "ask_human", "question": "Need more info?"}
        return TurnOutput(
            text=f"```peanuts-action\n{json.dumps(payload)}\n```",
            provider_session_id="sess-loop",
        )


async def test_question_budget_prevents_infinite_ask_human_livelock(tmp_path: Path) -> None:
    cfg = PeanutsConfig(
        db_path=tmp_path / "peanuts.db",
        cwd=tmp_path,
        root_provider="mock",
        max_questions_per_turn=1,
    )
    orch = Orchestrator(cfg, runners={"mock": _AlwaysAskRunner()})
    try:
        await orch.send_human_message("start")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        qs = orch.repo.list_questions(only_open=True)
        assert len(qs) == 1

        # After answering once, the 2nd ask_human attempt exceeds budget=1 and degrades cleanly to reply
        await orch.answer_question(qs[0].id, "Here is the answer")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        assert orch.repo.list_questions(only_open=True) == []
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE
    finally:
        await orch.shutdown()


async def test_sqlite_crash_recovery_resets_orphaned_running_agent(tmp_path: Path) -> None:
    db_path = tmp_path / "crash.db"
    cfg = PeanutsConfig(db_path=db_path, cwd=tmp_path, root_provider="mock")
    orch1 = Orchestrator(cfg)
    # Simulate an abrupt crash mid-turn by leaving root RUNNING in DB
    orch1.repo.enqueue_human_message("root", "hello")
    started = orch1.repo.start_inbox_turn("root")
    assert started is not None
    assert orch1.repo.get_root_agent().status == AgentStatus.RUNNING
    await orch1.shutdown()

    # Re-opening Orchestrator recovers the orphaned RUNNING agent to FAILED
    orch2 = Orchestrator(cfg)
    try:
        assert orch2.repo.get_root_agent().status == AgentStatus.FAILED
        _assert_invariants(orch2)
    finally:
        await orch2.shutdown()


async def test_explicit_root_provider_overrides_existing_db(tmp_path: Path) -> None:
    db_path = tmp_path / "override.db"
    orch1 = Orchestrator(PeanutsConfig(db_path=db_path, cwd=tmp_path, root_provider="mock"))
    await orch1.send_human_message("direct: hi")
    await orch1.wait_until_quiescent()
    assert orch1.repo.get_root_agent().provider == "mock"
    assert orch1.repo.get_root_agent().provider_session_id is not None
    await orch1.shutdown()

    orch2 = Orchestrator(PeanutsConfig(db_path=db_path, cwd=tmp_path, root_provider="claude"))
    try:
        root = orch2.repo.get_root_agent()
        assert root.provider == "claude"
        assert root.provider_session_id is None
    finally:
        await orch2.shutdown()


async def test_nonblocking_root_spawns_second_child_while_first_is_running(tmp_path: Path) -> None:
    """Verify the asyncio event-loop mediator accepts Theme B while Theme A's child is still RUNNING."""
    import asyncio

    cfg = PeanutsConfig(
        db_path=tmp_path / "nonblocking.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg)
    try:
        # Start a slow child on theme 'alpha'
        await orch.send_human_message("alpha: slow_child background task")
        # Wait briefly until root delegated to alpha and alpha is actively RUNNING
        for _ in range(30):
            dels = orch.repo.list_delegations()
            if dels and dels[0].status == DelegationStatus.IN_PROGRESS:
                break
            await asyncio.sleep(0.01)

        dels_mid = orch.repo.list_delegations()
        assert len(dels_mid) == 1
        assert dels_mid[0].status == DelegationStatus.IN_PROGRESS
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE

        # While alpha is STILL RUNNING, send theme 'beta' -> root spawns beta immediately!
        await orch.send_human_message("beta: urgent second theme")
        for _ in range(30):
            dels = orch.repo.list_delegations()
            if len(dels) >= 2:
                break
            await asyncio.sleep(0.01)

        dels_both = orch.repo.list_delegations()
        assert len(dels_both) == 2
        assert dels_both[0].status == DelegationStatus.IN_PROGRESS  # alpha still running while beta started!
        _assert_invariants(orch)

        await orch.wait_until_quiescent()
        _assert_invariants(orch)
        assert all(d.status == DelegationStatus.DONE for d in orch.repo.list_delegations())
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE
    finally:
        await orch.shutdown()


class _HallucinatingMockDelegator:
    """Simulates a real provider (claude) mistakenly emitting provider='mock' in a delegate action."""

    provider: ProviderName = "claude"

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        if turn.agent.depth == 0 and not turn.prompt.split("[INPUT]\n", 1)[1].startswith("Child agent update"):
            payload = {
                "action": "delegate",
                "message": "delegating",
                "tasks": [{"theme": "react", "provider": "mock", "prompt": "explain useState"}],
            }
            return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="c-root")
        return TurnOutput(text="real claude answer", provider_session_id=f"c-{turn.agent.id}")


async def test_real_mode_never_spawns_mock_child_even_if_llm_requests_mock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from peanuts.models import ProviderAvailability

    def _fake_detect(_self: PeanutsConfig) -> list[ProviderAvailability]:
        return [
            ProviderAvailability(name="claude", available=True, binary="/usr/bin/claude"),
            ProviderAvailability(name="codex", available=False, binary=None),
            ProviderAvailability(name="mock", available=True, binary="builtin"),
        ]

    monkeypatch.setattr(PeanutsConfig, "detect_providers", _fake_detect)
    cfg = PeanutsConfig(db_path=tmp_path / "no_mock.db", cwd=tmp_path, root_provider="claude")
    fake_runner = _HallucinatingMockDelegator()
    orch = Orchestrator(cfg, runners={"claude": fake_runner, "mock": fake_runner})
    try:
        assert orch.available_providers() == ["claude"]
        await orch.send_human_message("useState教えて")
        await orch.wait_until_quiescent()
        children = orch.repo.list_children("root")
        assert len(children) == 1
        assert children[0].provider == "claude"
    finally:
        await orch.shutdown()
