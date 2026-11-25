"""Unit tests for the TLA+-verified state machine rules and invariants."""

from __future__ import annotations

import pytest

from peanuts.models import (
    AgentRecord,
    AgentStatus,
    DelegationRecord,
    DelegationStatus,
    QuestionRecord,
    QuestionStatus,
)
from peanuts.state_machine import (
    InvariantViolationError,
    StateTransitionError,
    check_invariants,
    validate_agent_transition,
    validate_delegation_depth,
    validate_delegation_transition,
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
    grand = AgentRecord(id="g1", parent_id="c1", depth=2, theme="auth-sub", provider="codex", status=AgentStatus.RUNNING)
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
