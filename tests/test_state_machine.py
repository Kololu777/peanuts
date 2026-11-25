"""Unit tests for the TLA+-verified state machine rules and invariants."""

from __future__ import annotations

import pytest

from peanuts.models import (
    AgentRecord,
    AgentStatus,
    DelegationRecord,
    DelegationStatus,
    MergeJobRecord,
    MergeJobStatus,
    QuestionRecord,
    QuestionStatus,
)
from peanuts.state_machine import (
    ALLOWED_AGENT_TRANSITIONS,
    ALLOWED_DELEGATION_TRANSITIONS,
    ALLOWED_MERGE_JOB_TRANSITIONS,
    InvariantViolationError,
    StateTransitionError,
    check_invariants,
    check_merge_queue_invariants,
    validate_agent_transition,
    validate_delegation_depth,
    validate_delegation_transition,
    validate_merge_job_transition,
    validate_question_budget,
)


def test_allowed_and_illegal_agent_transitions() -> None:
    validate_agent_transition(AgentStatus.IDLE, AgentStatus.RUNNING)
    validate_agent_transition(AgentStatus.RUNNING, AgentStatus.WAITING_CHILD)
    validate_agent_transition(AgentStatus.RUNNING, AgentStatus.WAITING_HUMAN)
    validate_agent_transition(AgentStatus.RUNNING, AgentStatus.FAILED)
    validate_agent_transition(AgentStatus.WAITING_CHILD, AgentStatus.IDLE)
    validate_agent_transition(AgentStatus.WAITING_HUMAN, AgentStatus.IDLE)
    validate_agent_transition(AgentStatus.FAILED, AgentStatus.IDLE)

    with pytest.raises(StateTransitionError):
        validate_agent_transition(AgentStatus.IDLE, AgentStatus.WAITING_CHILD)
    with pytest.raises(StateTransitionError):
        validate_agent_transition(AgentStatus.WAITING_CHILD, AgentStatus.RUNNING)
    with pytest.raises(StateTransitionError):
        validate_agent_transition(AgentStatus.RUNNING, AgentStatus.RUNNING)


def test_allowed_and_illegal_delegation_transitions() -> None:
    validate_delegation_transition(DelegationStatus.PENDING, DelegationStatus.IN_PROGRESS)
    validate_delegation_transition(DelegationStatus.IN_PROGRESS, DelegationStatus.NEEDS_HUMAN)
    validate_delegation_transition(DelegationStatus.NEEDS_HUMAN, DelegationStatus.IN_PROGRESS)
    validate_delegation_transition(DelegationStatus.IN_PROGRESS, DelegationStatus.DONE)
    validate_delegation_transition(DelegationStatus.IN_PROGRESS, DelegationStatus.FAILED)
    validate_delegation_transition(DelegationStatus.PENDING, DelegationStatus.CANCELLED)

    with pytest.raises(StateTransitionError):
        validate_delegation_transition(DelegationStatus.DONE, DelegationStatus.IN_PROGRESS)
    with pytest.raises(StateTransitionError):
        validate_delegation_transition(DelegationStatus.PENDING, DelegationStatus.DONE)


def test_depth_and_question_bounds() -> None:
    validate_delegation_depth(0, max_depth=2)
    validate_delegation_depth(1, max_depth=2)
    with pytest.raises(StateTransitionError):
        validate_delegation_depth(2, max_depth=2)

    validate_question_budget(0, max_questions_per_turn=2)
    validate_question_budget(1, max_questions_per_turn=2)
    with pytest.raises(StateTransitionError):
        validate_question_budget(2, max_questions_per_turn=2)


def test_check_invariants_detects_violations() -> None:
    root = AgentRecord(id="root", parent_id=None, depth=0, theme="root", provider="mock", status=AgentStatus.IDLE)
    child = AgentRecord(
        id="c1",
        parent_id="root",
        depth=1,
        theme="auth",
        provider="codex",
        status=AgentStatus.WAITING_HUMAN,
        question_count=1,
    )
    delegation = DelegationRecord(
        id="d1",
        caller_id="root",
        callee_id="c1",
        theme="auth",
        prompt="fix login",
        status=DelegationStatus.NEEDS_HUMAN,
    )
    question = QuestionRecord(
        id="q1",
        agent_id="c1",
        delegation_id="d1",
        theme="auth",
        question="Allow breaking migration?",
        status=QuestionStatus.OPEN,
    )

    # Valid snapshot passes (root stays non-blocking IDLE while child is active)
    check_invariants([root, child], [delegation], [question], max_depth=2, max_questions_per_turn=3)

    # Root entering WAITING_CHILD violates RootNeverBlockedOnChildren
    blocked_root = root.model_copy(update={"status": AgentStatus.WAITING_CHILD})
    with pytest.raises(InvariantViolationError):
        check_invariants([blocked_root, child], [delegation], [question], max_depth=2, max_questions_per_turn=3)

    # Non-root caller with active grandchild delegation must be WAITING_CHILD
    grand = AgentRecord(
        id="g1", parent_id="c1", depth=2, theme="auth-sub", provider="codex", status=AgentStatus.RUNNING
    )
    bad_child = child.model_copy(update={"status": AgentStatus.IDLE, "question_count": 0})
    sub_del = DelegationRecord(
        id="d2",
        caller_id="c1",
        callee_id="g1",
        theme="auth-sub",
        prompt="write test",
        status=DelegationStatus.IN_PROGRESS,
    )
    with pytest.raises(InvariantViolationError):
        check_invariants([root, bad_child, grand], [sub_del], [], max_depth=2, max_questions_per_turn=3)

    # WAITING_HUMAN without open question violates WaitingHumanConsistency
    with pytest.raises(InvariantViolationError):
        check_invariants([root, child], [delegation], [], max_depth=2, max_questions_per_turn=3)

    # Depth exceeding max_depth violates DepthBound
    deep_child = child.model_copy(update={"depth": 3})
    with pytest.raises(InvariantViolationError):
        check_invariants([root, deep_child], [], [question], max_depth=2, max_questions_per_turn=3)


def test_exhaustive_transition_matrices_match_tla_specs() -> None:
    """Verify every (src, dst) pair for Agent, Delegation, and MergeJob conforms to TLA+ transition tables."""
    # AgentStatus (5x5 = 25 pairs)
    for src in AgentStatus:
        for dst in AgentStatus:
            if dst in ALLOWED_AGENT_TRANSITIONS[src]:
                validate_agent_transition(src, dst)
            else:
                with pytest.raises(StateTransitionError):
                    validate_agent_transition(src, dst)

    # DelegationStatus (6x6 = 36 pairs, including IN_PROGRESS/NEEDS_HUMAN -> CANCELLED and settled sinks)
    for src in DelegationStatus:
        for dst in DelegationStatus:
            if dst in ALLOWED_DELEGATION_TRANSITIONS[src]:
                validate_delegation_transition(src, dst)
            else:
                with pytest.raises(StateTransitionError):
                    validate_delegation_transition(src, dst)

    # MergeJobStatus (7x7 = 49 pairs from PeanutsMergeQueue.tla)
    for src in MergeJobStatus:
        for dst in MergeJobStatus:
            if dst in ALLOWED_MERGE_JOB_TRANSITIONS[src]:
                validate_merge_job_transition(src, dst)
            else:
                with pytest.raises(StateTransitionError):
                    validate_merge_job_transition(src, dst)


def test_check_invariants_detects_remaining_tla_safety_violations() -> None:
    """Covers StrictTreeHierarchy, AtMostOneActiveIncomingDelegation, QuestionBudget, and WaitingHumanConsistency."""
    root = AgentRecord(id="root", parent_id=None, depth=0, theme="root", provider="mock", status=AgentStatus.IDLE)
    c1 = AgentRecord(id="c1", parent_id="root", depth=1, theme="auth", provider="mock", status=AgentStatus.IDLE)

    # 1. Missing root / multiple roots
    with pytest.raises(InvariantViolationError, match="exactly 1 root"):
        check_invariants([c1], [], [], max_depth=2, max_questions_per_turn=2)
    root2 = root.model_copy(update={"id": "root2"})
    with pytest.raises(InvariantViolationError, match="exactly 1 root"):
        check_invariants([root, root2], [], [], max_depth=2, max_questions_per_turn=2)

    # 2. Root depth != 0
    bad_root = root.model_copy(update={"depth": 1})
    with pytest.raises(InvariantViolationError, match="depth 0"):
        check_invariants([bad_root], [], [], max_depth=2, max_questions_per_turn=2)

    # 3. Self-parent cycle
    self_parent = c1.model_copy(update={"parent_id": "c1"})
    with pytest.raises(InvariantViolationError, match="Self-parent"):
        check_invariants([root, self_parent], [], [], max_depth=2, max_questions_per_turn=2)

    # 4. Orphan agent
    orphan = c1.model_copy(update={"parent_id": "missing"})
    with pytest.raises(InvariantViolationError, match="Orphan"):
        check_invariants([root, orphan], [], [], max_depth=2, max_questions_per_turn=2)

    # 5. Depth gap (StrictTreeHierarchy: agent.depth != parent.depth + 1)
    gapped = c1.model_copy(update={"depth": 2})
    with pytest.raises(InvariantViolationError, match="StrictTreeHierarchy"):
        check_invariants([root, gapped], [], [], max_depth=2, max_questions_per_turn=2)

    # 6. Question budget exceeded
    over_q = c1.model_copy(update={"question_count": 3})
    with pytest.raises(InvariantViolationError, match="Question budget"):
        check_invariants([root, over_q], [], [], max_depth=2, max_questions_per_turn=2)

    # 7. WaitingHumanConsistency reverse direction: IDLE agent with OPEN question
    open_q = QuestionRecord(
        id="q1", agent_id="c1", delegation_id=None, theme="auth", question="stale?", status=QuestionStatus.OPEN
    )
    with pytest.raises(InvariantViolationError, match="WaitingHumanConsistency"):
        check_invariants([root, c1], [], [open_q], max_depth=2, max_questions_per_turn=2)

    # 8. Delegation references unknown agent
    bad_ref_del = DelegationRecord(
        id="d-bad", caller_id="root", callee_id="ghost", theme="x", prompt="p", status=DelegationStatus.PENDING
    )
    with pytest.raises(InvariantViolationError, match="unknown agent"):
        check_invariants([root, c1], [bad_ref_del], [], max_depth=2, max_questions_per_turn=2)

    # 9. Strict depth ordering violated (callee.depth <= caller.depth)
    c2 = AgentRecord(id="c2", parent_id="root", depth=1, theme="docs", provider="mock", status=AgentStatus.IDLE)
    sibling_del = DelegationRecord(
        id="d-sib", caller_id="c1", callee_id="c2", theme="docs", prompt="p", status=DelegationStatus.DONE
    )
    with pytest.raises(InvariantViolationError, match="strict depth ordering"):
        check_invariants([root, c1, c2], [sibling_del], [], max_depth=2, max_questions_per_turn=2)

    # 10. AtMostOneActiveIncomingDelegation violated (2 active delegations targeting c1)
    d1 = DelegationRecord(
        id="d1", caller_id="root", callee_id="c1", theme="auth", prompt="p1", status=DelegationStatus.IN_PROGRESS
    )
    d2 = DelegationRecord(
        id="d2", caller_id="root", callee_id="c1", theme="auth", prompt="p2", status=DelegationStatus.PENDING
    )
    with pytest.raises(InvariantViolationError, match="AtMostOneActiveIncomingDelegation"):
        check_invariants([root, c1], [d1, d2], [], max_depth=2, max_questions_per_turn=2)


def test_check_merge_queue_invariants_detects_all_tla_violations() -> None:
    """Covers SingleIntegratorWriter, NeverMergeIntoDirtyBase, CleanBaseWhenIntegratorIdle, and DisjointWorkerBranches."""
    w1 = AgentRecord(
        id="w1",
        parent_id="root",
        depth=1,
        theme="auth",
        provider="mock",
        role_kind="worker",
        branch_name="peanuts/auth-1",
    )
    w2 = AgentRecord(
        id="w2",
        parent_id="root",
        depth=1,
        theme="docs",
        provider="mock",
        role_kind="worker",
        branch_name="peanuts/docs-2",
    )
    j_merging = MergeJobRecord(
        id="m1",
        kind="branch_merge",
        worker_agent_id="w1",
        theme="auth",
        source_branch="peanuts/auth-1",
        target_branch="main",
        status=MergeJobStatus.MERGING,
    )
    j_conflict = j_merging.model_copy(update={"id": "m2", "status": MergeJobStatus.CONFLICT_RESOLVING})
    j_inst = MergeJobRecord(
        id="i1",
        kind="instruction",
        worker_agent_id="root",
        theme="integrator",
        source_branch="main",
        target_branch="main",
        instruction="git add -A && git commit -m wip",
        status=MergeJobStatus.RUNNING_INSTRUCTION,
    )

    # Valid snapshots pass:
    # - Clean branch merge with MERGE_HEAD active
    check_merge_queue_invariants([w1, w2], [j_merging], base_dirty=False, base_merge_in_progress=True)
    # - Instruction job IS allowed on a dirty base (TLA+ PeanutsMergeQueue.tla)
    check_merge_queue_invariants([w1, w2], [j_inst], base_dirty=True, base_merge_in_progress=False)

    # 1. SingleIntegratorWriter violated (two active jobs simultaneously)
    with pytest.raises(InvariantViolationError, match="SingleIntegratorWriter"):
        check_merge_queue_invariants([w1, w2], [j_merging, j_inst], base_dirty=False, base_merge_in_progress=True)

    # 2. NeverMergeIntoDirtyBase violated (MERGING or CONFLICT_RESOLVING while base_dirty=True)
    with pytest.raises(InvariantViolationError, match="NeverMergeIntoDirtyBase"):
        check_merge_queue_invariants([w1, w2], [j_merging], base_dirty=True, base_merge_in_progress=True)
    with pytest.raises(InvariantViolationError, match="NeverMergeIntoDirtyBase"):
        check_merge_queue_invariants([w1, w2], [j_conflict], base_dirty=True, base_merge_in_progress=True)

    # 3. CleanBaseWhenIntegratorIdle violated (MERGE_HEAD left behind with no active job)
    j_done = j_merging.model_copy(update={"status": MergeJobStatus.MERGED})
    with pytest.raises(InvariantViolationError, match="CleanBaseWhenIntegratorIdle"):
        check_merge_queue_invariants([w1, w2], [j_done], base_dirty=False, base_merge_in_progress=True)

    # 4. DisjointWorkerBranches violated (two workers sharing the same branch)
    w2_dup = w2.model_copy(update={"branch_name": "peanuts/auth-1"})
    with pytest.raises(InvariantViolationError, match="DisjointWorkerBranches"):
        check_merge_queue_invariants([w1, w2_dup], [j_done], base_dirty=False, base_merge_in_progress=False)
