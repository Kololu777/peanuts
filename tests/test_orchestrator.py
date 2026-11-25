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


async def test_human_direct_delegate_routes_human_to_root_to_child_and_reports_back(tmp_path: Path) -> None:
    """Verifies TLA+ HumanDirectDelegate: Human -> Root (logged) -> deterministic Root->child -> Root report."""
    import asyncio

    from peanuts.state_machine import StateTransitionError

    cfg = PeanutsConfig(
        db_path=tmp_path / "direct_del.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg)
    try:
        # Spawn child 'auth' first
        await orch.send_human_message("auth: initial task")
        await orch.wait_until_quiescent()
        auth = orch.repo.find_child_by_theme("root", "auth")
        assert auth is not None
        prev_sess = auth.provider_session_id

        # Now invoke direct_delegate_to_child (mirrors `@#01 continue auth`)
        delegation, human_msg, route_msg = await orch.direct_delegate_to_child(
            auth.id,
            "continue auth work",
            display_text="@#01 continue auth work",
        )
        assert delegation.caller_id == "root"
        assert delegation.callee_id == auth.id
        assert human_msg.agent_id == "root"
        assert route_msg.agent_id == "root"

        # Trying to direct-delegate again while child is busy MUST raise StateTransitionError
        with pytest.raises(StateTransitionError):
            await orch.direct_delegate_to_child(auth.id, "duplicate while busy")

        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        # Child reused its session AND root received the child's completion report!
        auth_after = orch.repo.get_agent(auth.id)
        assert auth_after is not None and auth_after.provider_session_id == prev_sess
        root_msgs = orch.repo.list_messages(agent_id="root")
        assert any("@#01 continue auth work" in m.content for m in root_msgs)
        assert any("continue auth work" in m.content and m.role.value == "agent" for m in root_msgs)
        assert asyncio is not None
    finally:
        await orch.shutdown()


async def test_cancel_pending_and_needs_human_delegations_notify_root_and_wake_parent(tmp_path: Path) -> None:
    """Covers TLA+ CancelPendingDelegation plus NEEDS_HUMAN -> CANCELLED and waiting parent wake-up."""
    from peanuts.models import QuestionStatus

    cfg = PeanutsConfig(
        db_path=tmp_path / "cancel.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg)
    try:
        # Case 1: PENDING -> CANCELLED on a root-owned delegation notifies Root and settles cleanly
        orch.repo.enqueue_human_message("root", "seed")
        assert orch.repo.start_inbox_turn("root") is not None
        _, created, _ = orch.repo.delegate_batch(
            "root",
            [("billing", "pending task", "mock", None)],
            max_depth=cfg.max_depth,
        )
        pending_del = created[0]
        assert pending_del.status == DelegationStatus.PENDING
        _assert_invariants(orch)

        cancelled = await orch.cancel_delegation(pending_del.id)
        assert cancelled.status == DelegationStatus.CANCELLED
        await orch.wait_until_quiescent()
        _assert_invariants(orch)
        root_msgs = orch.repo.list_messages(agent_id="root")
        assert any("CANCELLED" in m.content for m in root_msgs)

        # Case 2: NEEDS_HUMAN -> CANCELLED closes OPEN question, returns callee to IDLE, notifies Root
        await orch.send_human_message("db: 確認 マイグレーション")
        await orch.wait_until_quiescent()
        open_qs = orch.repo.list_questions(only_open=True)
        assert len(open_qs) == 1
        q = open_qs[0]
        assert q.delegation_id is not None
        _assert_invariants(orch)

        cancelled_nh = await orch.cancel_delegation(q.delegation_id)
        assert cancelled_nh.status == DelegationStatus.CANCELLED
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        assert orch.repo.list_questions(only_open=True) == []
        q_after = orch.repo.get_question(q.id)
        assert q_after is not None and q_after.status == QuestionStatus.CANCELLED
        callee_after = orch.repo.get_agent(q.agent_id)
        assert callee_after is not None and callee_after.status == AgentStatus.IDLE

        # Case 3: Non-root WAITING_CHILD wakes via ResumeAfterChildren when its PENDING grandchild is CANCELLED
        _, root_dels, _ = (
            orch.repo.delegate_batch(
                "root",
                [("parent-c", "work", "mock", None)],
                max_depth=cfg.max_depth,
            )
            if orch.repo.start_inbox_turn("root")
            else (None, [], [])
        )
        if not root_dels:
            orch.repo.enqueue_human_message("root", "seed-parent")
            assert orch.repo.start_inbox_turn("root") is not None
            _, root_dels, _ = orch.repo.delegate_batch(
                "root",
                [("parent-c", "work", "mock", None)],
                max_depth=cfg.max_depth,
            )
        started_child = orch.repo.start_delegation(root_dels[0].id)
        assert started_child is not None
        child_agent, _, _ = started_child
        _, grand_dels, _ = orch.repo.delegate_batch(
            child_agent.id,
            [("grand-c", "subtask", "mock", None)],
            max_depth=cfg.max_depth,
        )
        mid_child = orch.repo.get_agent(child_agent.id)
        assert mid_child is not None and mid_child.status == AgentStatus.WAITING_CHILD
        _assert_invariants(orch)

        cancelled_grand = await orch.cancel_delegation(grand_dels[0].id)
        assert cancelled_grand.status == DelegationStatus.CANCELLED
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        final_child = orch.repo.get_agent(child_agent.id)
        assert final_child is not None and final_child.status == AgentStatus.IDLE
        final_root_del = orch.repo.get_delegation(root_dels[0].id)
        assert final_root_del is not None and final_root_del.status == DelegationStatus.DONE
    finally:
        await orch.shutdown()


async def test_non_root_parallel_grandchild_fanout_waits_for_all_before_resuming(tmp_path: Path) -> None:
    """Covers TLA+ DelegateTwo on a non-root child: WAITING_CHILD must not resume until ALL grandchildren settle."""
    import asyncio

    release_g2 = asyncio.Event()
    g1_done = asyncio.Event()
    g2_started = asyncio.Event()

    class _ParallelGrandchildRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            agent = turn.agent
            user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()
            if agent.depth == 0:
                if user_input.startswith("Child agent update"):
                    return TurnOutput(text=f"root-report: {user_input}", provider_session_id="r1")
                payload = {
                    "action": "delegate",
                    "message": "spawn parent-worker",
                    "tasks": [{"theme": "parent-w", "provider": "mock", "prompt": "fanout-two"}],
                }
                return TurnOutput(
                    text=f"```peanuts-action\n{json.dumps(payload)}\n```",
                    provider_session_id="r1",
                )
            if agent.depth == 1:
                if user_input.startswith("All delegated child tasks have settled."):
                    return TurnOutput(text=f"parent-merged: {user_input}", provider_session_id="c1")
                payload = {
                    "action": "delegate",
                    "message": "fanout 2 grandchildren",
                    "tasks": [
                        {"theme": "g1", "provider": "mock", "prompt": "fast-g1"},
                        {"theme": "g2", "provider": "mock", "prompt": "slow-g2"},
                    ],
                }
                return TurnOutput(
                    text=f"```peanuts-action\n{json.dumps(payload)}\n```",
                    provider_session_id="c1",
                )
            if agent.theme == "g1":
                g1_done.set()
                return TurnOutput(text="g1 finished", provider_session_id="g1-s")
            g2_started.set()
            await release_g2.wait()
            return TurnOutput(text="g2 finished", provider_session_id="g2-s")

    cfg = PeanutsConfig(
        db_path=tmp_path / "fanout_grand.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg, runners={"mock": _ParallelGrandchildRunner()})
    try:
        await orch.send_human_message("start parallel grandchild")
        await asyncio.wait_for(g1_done.wait(), timeout=5.0)
        await asyncio.wait_for(g2_started.wait(), timeout=5.0)

        # Wait until g1 delegation is recorded DONE while g2 is still IN_PROGRESS
        for _ in range(50):
            dels = orch.repo.list_delegations()
            g1_del = next((d for d in dels if d.theme == "g1"), None)
            g2_del = next((d for d in dels if d.theme == "g2"), None)
            if g1_del and g2_del and g1_del.status == DelegationStatus.DONE:
                break
            await asyncio.sleep(0.01)

        parent_worker = orch.repo.find_child_by_theme("root", "parent-w")
        assert parent_worker is not None
        # Parent worker MUST remain WAITING_CHILD while g2 is still IN_PROGRESS
        assert parent_worker.status == AgentStatus.WAITING_CHILD
        _assert_invariants(orch)

        release_g2.set()
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        assert all(d.status == DelegationStatus.DONE for d in orch.repo.list_delegations())
        assert all(a.status == AgentStatus.IDLE for a in orch.repo.list_agents())
        parent_msgs = orch.repo.list_messages(agent_id=parent_worker.id)
        assert any("g1 finished" in m.content and "g2 finished" in m.content for m in parent_msgs)
    finally:
        release_g2.set()
        await orch.shutdown()


async def test_grandchild_failure_wakes_waiting_child_parent(tmp_path: Path) -> None:
    """Covers TLA+ FailTurn at depth=2 -> non-root WAITING_CHILD parent wakes via ResumeAfterChildren."""
    from peanuts.runners.base import RunnerError

    class _FailingGrandchildRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            agent = turn.agent
            user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()
            if agent.depth == 0:
                if user_input.startswith("Child agent update"):
                    return TurnOutput(text=f"root-got: {user_input}", provider_session_id="r1")
                payload = {
                    "action": "delegate",
                    "tasks": [{"theme": "mid", "provider": "mock", "prompt": "run-grand"}],
                }
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
            if agent.depth == 1:
                if user_input.startswith("All delegated child tasks have settled."):
                    return TurnOutput(text=f"mid-handled-failure: {user_input}", provider_session_id="m1")
                payload = {
                    "action": "delegate",
                    "tasks": [{"theme": "leaf", "provider": "mock", "prompt": "boom"}],
                }
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="m1")
            raise RunnerError("grandchild exploded")

    cfg = PeanutsConfig(
        db_path=tmp_path / "grand_fail.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg, runners={"mock": _FailingGrandchildRunner()})
    try:
        await orch.send_human_message("trigger grandchild failure")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        mid = orch.repo.find_child_by_theme("root", "mid")
        assert mid is not None and mid.status == AgentStatus.IDLE
        leaf = orch.repo.find_child_by_theme(mid.id, "leaf")
        assert leaf is not None and leaf.status == AgentStatus.FAILED
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE

        dels = {d.theme: d for d in orch.repo.list_delegations()}
        assert dels["leaf"].status == DelegationStatus.FAILED
        assert dels["mid"].status == DelegationStatus.DONE
    finally:
        await orch.shutdown()


async def test_grandchild_ask_human_three_level_mediation_loop(tmp_path: Path) -> None:
    """Covers TLA+ AskHuman + HumanAnswer at depth=2 while depth=1 waits in WAITING_CHILD and root stays IDLE."""

    class _GrandchildAskRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            agent = turn.agent
            user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()
            if agent.depth == 0:
                if user_input.startswith("Child agent update"):
                    return TurnOutput(text=f"root-final: {user_input}", provider_session_id="r1")
                payload = {
                    "action": "delegate",
                    "tasks": [{"theme": "mid-q", "provider": "mock", "prompt": "go"}],
                }
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
            if agent.depth == 1:
                if user_input.startswith("All delegated child tasks have settled."):
                    return TurnOutput(text=f"mid-done: {user_input}", provider_session_id="m1")
                payload = {
                    "action": "delegate",
                    "tasks": [{"theme": "leaf-q", "provider": "mock", "prompt": "ask"}],
                }
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="m1")
            if user_input.startswith("Human answered your question"):
                return TurnOutput(text=f"leaf-finished: {user_input}", provider_session_id="l1")
            payload = {"action": "ask_human", "question": "Grandchild needs approval?"}
            return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="l1")

    cfg = PeanutsConfig(
        db_path=tmp_path / "grand_ask.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg, runners={"mock": _GrandchildAskRunner()})
    try:
        await orch.send_human_message("start 3-level question")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        # Verify intermediate 3-level state: root=IDLE, mid=WAITING_CHILD, leaf=WAITING_HUMAN
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE
        mid = orch.repo.find_child_by_theme("root", "mid-q")
        assert mid is not None and mid.status == AgentStatus.WAITING_CHILD
        leaf = orch.repo.find_child_by_theme(mid.id, "leaf-q")
        assert leaf is not None and leaf.status == AgentStatus.WAITING_HUMAN

        open_qs = orch.repo.list_questions(only_open=True)
        assert len(open_qs) == 1 and open_qs[0].agent_id == leaf.id

        await orch.answer_question(open_qs[0].id, "approved")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        assert orch.repo.list_questions(only_open=True) == []
        assert all(a.status == AgentStatus.IDLE for a in orch.repo.list_agents())
        assert all(d.status == DelegationStatus.DONE for d in orch.repo.list_delegations())
    finally:
        await orch.shutdown()


async def test_depth_limit_and_child_question_budget_degrade_safely(tmp_path: Path) -> None:
    """Covers DepthBound fallback at depth == max_depth and MaxQuestionsPerTurn fallback on a child worker."""

    class _BoundaryRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            agent = turn.agent
            user_input = turn.prompt.split("[INPUT]\n", 1)[-1].strip()
            if agent.depth == 0:
                if user_input.startswith("Child agent update"):
                    return TurnOutput(text="root ack", provider_session_id="r1")
                if "test-depth" in user_input:
                    payload = {
                        "action": "delegate",
                        "tasks": [{"theme": "deep-worker", "provider": "mock", "prompt": "try-delegate"}],
                    }
                else:
                    payload = {
                        "action": "delegate",
                        "tasks": [{"theme": "ask-worker", "provider": "mock", "prompt": "loop-ask"}],
                    }
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="r1")
            if agent.theme == "deep-worker":
                payload = {
                    "action": "delegate",
                    "message": "trying to exceed max_depth",
                    "tasks": [{"theme": "forbidden-grand", "provider": "mock", "prompt": "should not spawn"}],
                }
                return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="d1")
            # Child always asks human -> 2nd ask must degrade to reply and settle delegation DONE
            payload = {"action": "ask_human", "question": "Still unsure?"}
            return TurnOutput(text=f"```peanuts-action\n{json.dumps(payload)}\n```", provider_session_id="a1")

    cfg = PeanutsConfig(
        db_path=tmp_path / "bounds.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=1,
        max_questions_per_turn=1,
    )
    orch = Orchestrator(cfg, runners={"mock": _BoundaryRunner()})
    try:
        # 1. Depth limit fallback
        await orch.send_human_message("test-depth")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)
        assert max(a.depth for a in orch.repo.list_agents()) == 1
        deep_del = next(d for d in orch.repo.list_delegations() if d.theme == "deep-worker")
        assert deep_del.status == DelegationStatus.DONE
        assert deep_del.result is not None and "Depth limit 1 reached" in deep_del.result

        # 2. Child question budget exhaustion fallback
        await orch.send_human_message("test-child-ask")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)
        open_qs = orch.repo.list_questions(only_open=True)
        assert len(open_qs) == 1
        await orch.answer_question(open_qs[0].id, "go ahead")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        assert orch.repo.list_questions(only_open=True) == []
        ask_del = next(d for d in orch.repo.list_delegations() if d.theme == "ask-worker")
        assert ask_del.status == DelegationStatus.DONE
        assert ask_del.result is not None and "Question budget (1) exhausted" in ask_del.result
        ask_worker = orch.repo.find_child_by_theme("root", "ask-worker")
        assert ask_worker is not None and ask_worker.question_count == 0
    finally:
        await orch.shutdown()


async def test_root_failure_retry_and_direct_delegate_and_answer_guards(tmp_path: Path) -> None:
    """Covers Root FailTurn -> ResetFailed -> recovery, invalid retry, double answer, and direct-delegate guards."""
    from peanuts.runners.base import RunnerError
    from peanuts.state_machine import StateTransitionError

    fail_root = True

    class _FlakyRootRunner:
        provider: ProviderName = "mock"

        async def run_turn(self, turn: TurnInput, on_progress: ProgressCallback | None = None) -> TurnOutput:
            nonlocal fail_root
            if turn.agent.depth == 0 and fail_root:
                fail_root = False
                raise RunnerError("root transient error")
            return TurnOutput(text="recovered ok", provider_session_id="r-ok")

    cfg = PeanutsConfig(
        db_path=tmp_path / "guards.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    orch = Orchestrator(cfg, runners={"mock": _FlakyRootRunner()})
    try:
        # Root fails on first turn -> FAILED
        await orch.send_human_message("first try")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)
        assert orch.repo.get_root_agent().status == AgentStatus.FAILED

        # Retry transitions Root FAILED -> IDLE
        retried = await orch.retry_agent("root")
        assert retried.status == AgentStatus.IDLE
        _assert_invariants(orch)

        # Retrying an already-IDLE agent must raise StateTransitionError
        with pytest.raises(StateTransitionError):
            await orch.retry_agent("root")

        # Subsequent message succeeds
        await orch.send_human_message("second try")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE

        # Direct-delegate to root itself must fail
        with pytest.raises(StateTransitionError):
            await orch.direct_delegate_to_child("root", "cannot direct delegate root")

        # Set up a child in WAITING_HUMAN and verify direct_delegate + double answer guards
        orch.repo.enqueue_human_message("root", "seed")
        assert orch.repo.start_inbox_turn("root") is not None
        _, dels, _ = orch.repo.delegate_batch("root", [("worker-a", "do", "mock", None)], max_depth=2)
        started = orch.repo.start_delegation(dels[0].id)
        assert started is not None
        child, _, _ = started
        _, question, _ = orch.repo.ask_human(child.id, "confirm?", max_questions_per_turn=2)
        _assert_invariants(orch)

        with pytest.raises(StateTransitionError):
            await orch.direct_delegate_to_child(child.id, "busy waiting human")

        await orch.answer_question(question.id, "yes")
        await orch.wait_until_quiescent()
        _assert_invariants(orch)

        # Double answer on an already-ANSWERED question must raise StateTransitionError
        with pytest.raises(StateTransitionError):
            await orch.answer_question(question.id, "duplicate answer")

        # Direct-delegate to FAILED child must raise StateTransitionError
        orch.repo.enqueue_human_message(child.id, "queued item")
        # Non-empty inbox also blocks HumanDirectDelegate (TLA+ inboxCount[c] = 0)
        with pytest.raises(StateTransitionError):
            await orch.direct_delegate_to_child(child.id, "inbox not empty")
        assert orch.repo.start_inbox_turn(child.id) is not None
        orch.repo.fail_turn(child.id, "manual fail")
        with pytest.raises(StateTransitionError):
            await orch.direct_delegate_to_child(child.id, "child is failed")
        await orch.retry_agent(child.id)

        # Direct-delegate to a grandchild (depth=2) satisfies depth[root] < depth[grandchild]
        orch.repo.enqueue_human_message(child.id, "make grand")
        assert orch.repo.start_inbox_turn(child.id) is not None
        _, gdels, _ = orch.repo.delegate_batch(child.id, [("grand-a", "g", "mock", None)], max_depth=2)
        await orch.wait_until_quiescent()
        grand = orch.repo.find_child_by_theme(child.id, "grand-a")
        assert grand is not None and grand.depth == 2
        g_del, _, _ = await orch.direct_delegate_to_child(grand.id, "direct to grand")
        assert g_del.caller_id == "root" and g_del.callee_id == grand.id
        await orch.wait_until_quiescent()
        _assert_invariants(orch)
    finally:
        await orch.shutdown()


async def test_sqlite_crash_recovery_fails_child_in_progress_delegation_and_notifies_root(tmp_path: Path) -> None:
    """Crash while a child is RUNNING an IN_PROGRESS delegation marks both FAILED and notifies Root."""
    db_path = tmp_path / "crash_child.db"
    cfg = PeanutsConfig(db_path=db_path, cwd=tmp_path, root_provider="mock", default_child_provider="mock")
    orch1 = Orchestrator(cfg)
    orch1.repo.enqueue_human_message("root", "seed")
    assert orch1.repo.start_inbox_turn("root") is not None
    _, dels, _ = orch1.repo.delegate_batch("root", [("worker-crash", "task", "mock", None)], max_depth=2)
    started = orch1.repo.start_delegation(dels[0].id)
    assert started is not None
    child, active_del, _ = started
    assert child.status == AgentStatus.RUNNING
    assert active_del.status == DelegationStatus.IN_PROGRESS
    await orch1.shutdown()

    orch2 = Orchestrator(cfg)
    try:
        recovered_child = orch2.repo.get_agent(child.id)
        recovered_del = orch2.repo.get_delegation(active_del.id)
        assert recovered_child is not None and recovered_child.status == AgentStatus.FAILED
        assert recovered_del is not None and recovered_del.status == DelegationStatus.FAILED
        _assert_invariants(orch2)

        # Root received the failure notification in its inbox and can process it cleanly
        await orch2.wait_until_quiescent()
        _assert_invariants(orch2)
        root_msgs = orch2.repo.list_messages(agent_id="root")
        assert any("Interrupted by process restart" in m.content for m in root_msgs)
    finally:
        await orch2.shutdown()


def test_stateful_orchestrator_tla_model_exploration(tmp_path: Path) -> None:
    """Randomized deterministic exploration of all PeanutsOrchestrator.tla actions with per-step invariant checks."""
    import random

    from peanuts.db import PeanutsRepository

    rng = random.Random(20261125)
    repo = PeanutsRepository(tmp_path / "stateful_orch.db")
    max_depth = 2
    max_q = 2
    try:
        repo.ensure_root_agent("mock")

        def verify() -> None:
            check_invariants(
                repo.list_agents(),
                repo.list_delegations(),
                repo.list_questions(only_open=False),
                max_depth=max_depth,
                max_questions_per_turn=max_q,
            )

        verify()
        action_counts: dict[str, int] = {}

        for step in range(250):
            agents = repo.list_agents()
            open_qs = repo.list_questions(only_open=True)
            candidates: list[str] = ["human_send"]

            idle_with_inbox = repo.list_idle_agents_with_inbox()
            if idle_with_inbox:
                candidates.append("start_turn")

            pending_dels = repo.list_pending_delegations()
            startable_dels = [
                d
                for d in pending_dels
                if (callee := repo.get_agent(d.callee_id)) is not None and callee.status == AgentStatus.IDLE
            ]
            if startable_dels:
                candidates.append("start_delegation")
            if pending_dels:
                candidates.append("cancel_pending")

            running_agents = [a for a in agents if a.status == AgentStatus.RUNNING]
            if running_agents:
                candidates.extend(["finish_reply", "fail_turn"])
                if any(a.depth < max_depth for a in running_agents):
                    candidates.extend(["delegate_one", "delegate_two"])
                if any(a.question_count < max_q for a in running_agents):
                    candidates.append("ask_human")

            if open_qs:
                candidates.append("human_answer")

            failed_agents = [a for a in agents if a.status == AgentStatus.FAILED]
            if failed_agents:
                candidates.append("reset_failed")

            waiting_ready = repo.list_waiting_parents_ready_to_resume()
            if waiting_ready:
                candidates.append("resume_after_children")

            # Eligible direct-delegate targets (TLA+ HumanDirectDelegate)
            direct_targets = [
                a
                for a in agents
                if a.role_kind == "worker"
                and a.depth >= 1
                and a.status == AgentStatus.IDLE
                and repo.get_active_incoming_delegation(a.id) is None
                and a.id not in {x.id for x in idle_with_inbox}
            ]
            if direct_targets:
                candidates.append("human_direct_delegate")

            choice = rng.choice(candidates)
            action_counts[choice] = action_counts.get(choice, 0) + 1

            if choice == "human_send":
                repo.enqueue_human_message("root", f"msg-{step}")
            elif choice == "human_direct_delegate":
                target = rng.choice(direct_targets)
                repo.direct_delegate_from_root(target.id, f"direct-{step}")
            elif choice == "start_turn":
                a = rng.choice(idle_with_inbox)
                repo.start_inbox_turn(a.id)
            elif choice == "start_delegation":
                d = rng.choice(startable_dels)
                repo.start_delegation(d.id)
            elif choice == "cancel_pending":
                d = rng.choice(pending_dels)
                repo.cancel_delegation(d.id)
            elif choice == "finish_reply":
                a = rng.choice(running_agents)
                repo.finish_turn_reply(a.id, f"reply-{step}")
            elif choice == "fail_turn":
                a = rng.choice(running_agents)
                repo.fail_turn(a.id, f"err-{step}")
            elif choice == "delegate_one":
                a = rng.choice([x for x in running_agents if x.depth < max_depth])
                theme = rng.choice(["t1", "t2", f"t-{step % 5}"])
                repo.delegate_batch(a.id, [(theme, f"p-{step}", "mock", None)], max_depth=max_depth)
            elif choice == "delegate_two":
                a = rng.choice([x for x in running_agents if x.depth < max_depth])
                repo.delegate_batch(
                    a.id,
                    [("fan-a", f"pa-{step}", "mock", None), ("fan-b", f"pb-{step}", "mock", None)],
                    max_depth=max_depth,
                )
            elif choice == "ask_human":
                a = rng.choice([x for x in running_agents if x.question_count < max_q])
                repo.ask_human(a.id, f"q-{step}?", max_questions_per_turn=max_q)
            elif choice == "human_answer":
                q = rng.choice(open_qs)
                repo.answer_question(q.id, f"ans-{step}")
            elif choice == "reset_failed":
                a = rng.choice(failed_agents)
                repo.reset_failed(a.id)
            elif choice == "resume_after_children":
                a = rng.choice(waiting_ready)
                repo.resume_parent_after_children(a.id)

            verify()

        # Verify all major TLA+ action classes were exercised
        for expected_action in (
            "human_send",
            "start_turn",
            "finish_reply",
            "delegate_one",
            "delegate_two",
            "start_delegation",
            "ask_human",
            "human_answer",
            "fail_turn",
            "reset_failed",
            "cancel_pending",
        ):
            assert action_counts.get(expected_action, 0) > 0, f"Action {expected_action} was never exercised"

        # Drain to quiescent state (Liveness check: every active delegation and question settles)
        for _ in range(300):
            progressed = False
            for q in repo.list_questions(only_open=True):
                repo.answer_question(q.id, "drain")
                verify()
                progressed = True
            for a in repo.list_agents():
                if a.status == AgentStatus.FAILED:
                    repo.reset_failed(a.id)
                    verify()
                    progressed = True
                elif a.status == AgentStatus.RUNNING:
                    repo.finish_turn_reply(a.id, "drain-reply")
                    verify()
                    progressed = True
            for p in repo.list_waiting_parents_ready_to_resume():
                if repo.resume_parent_after_children(p.id) is not None:
                    verify()
                    progressed = True
            for d in repo.list_pending_delegations():
                if repo.start_delegation(d.id) is not None:
                    verify()
                    progressed = True
            for a in repo.list_idle_agents_with_inbox():
                if repo.start_inbox_turn(a.id) is not None:
                    verify()
                    progressed = True
            if not progressed:
                break

        assert repo.list_questions(only_open=True) == []
        assert all(a.status == AgentStatus.IDLE for a in repo.list_agents())
        assert all(
            d.status in {DelegationStatus.DONE, DelegationStatus.FAILED, DelegationStatus.CANCELLED}
            for d in repo.list_delegations()
        )
        verify()
    finally:
        repo.close()
