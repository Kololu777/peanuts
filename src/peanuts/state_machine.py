"""TLA+-aligned state transition rules and runtime invariant verifier.

Every transition rule and invariant here mirrors spec/PeanutsOrchestrator.tla.
"""

from __future__ import annotations

from collections.abc import Sequence

from peanuts.models import (
    ACTIVE_DELEGATION_STATUSES,
    ACTIVE_MERGE_STATUSES,
    AgentRecord,
    AgentStatus,
    DelegationRecord,
    DelegationStatus,
    MergeJobRecord,
    MergeJobStatus,
    QuestionRecord,
    QuestionStatus,
)


class StateTransitionError(ValueError):
    """Raised when an invalid state transition is attempted."""


class InvariantViolationError(AssertionError):
    """Raised when a TLA+ safety invariant is violated."""


ALLOWED_AGENT_TRANSITIONS: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.IDLE: frozenset({AgentStatus.RUNNING}),
    AgentStatus.RUNNING: frozenset(
        {
            AgentStatus.IDLE,
            AgentStatus.WAITING_CHILD,
            AgentStatus.WAITING_HUMAN,
            AgentStatus.FAILED,
        }
    ),
    AgentStatus.WAITING_CHILD: frozenset({AgentStatus.IDLE}),
    AgentStatus.WAITING_HUMAN: frozenset({AgentStatus.IDLE}),
    AgentStatus.FAILED: frozenset({AgentStatus.IDLE}),
}

ALLOWED_DELEGATION_TRANSITIONS: dict[DelegationStatus, frozenset[DelegationStatus]] = {
    DelegationStatus.PENDING: frozenset({DelegationStatus.IN_PROGRESS, DelegationStatus.CANCELLED}),
    DelegationStatus.IN_PROGRESS: frozenset(
        {
            DelegationStatus.NEEDS_HUMAN,
            DelegationStatus.DONE,
            DelegationStatus.FAILED,
            DelegationStatus.CANCELLED,
        }
    ),
    DelegationStatus.NEEDS_HUMAN: frozenset({DelegationStatus.IN_PROGRESS, DelegationStatus.CANCELLED}),
    DelegationStatus.DONE: frozenset(),
    DelegationStatus.FAILED: frozenset(),
    DelegationStatus.CANCELLED: frozenset(),
}


def validate_agent_transition(current: AgentStatus, target: AgentStatus) -> None:
    """Ensure agent status transition conforms to spec/PeanutsOrchestrator.tla."""
    if target not in ALLOWED_AGENT_TRANSITIONS[current]:
        raise StateTransitionError(f"Illegal agent transition: {current.value} -> {target.value}")


def validate_delegation_transition(current: DelegationStatus, target: DelegationStatus) -> None:
    """Ensure delegation status transition conforms to spec/PeanutsOrchestrator.tla."""
    if target not in ALLOWED_DELEGATION_TRANSITIONS[current]:
        raise StateTransitionError(f"Illegal delegation transition: {current.value} -> {target.value}")


def validate_delegation_depth(caller_depth: int, max_depth: int) -> None:
    """Enforce DepthBound invariant: depth[caller] < MaxDepth."""
    if caller_depth >= max_depth:
        raise StateTransitionError(
            f"Max delegation depth ({max_depth}) reached at depth {caller_depth}; cannot spawn child."
        )


def validate_question_budget(current_count: int, max_questions_per_turn: int) -> None:
    """Enforce MaxQuestionsPerTurn bound discovered by TLC liveness checking."""
    if current_count >= max_questions_per_turn:
        raise StateTransitionError(
            f"Agent exceeded max_questions_per_turn ({max_questions_per_turn}); "
            "must reply, delegate, or fail instead of looping on ask_human."
        )


def check_invariants(
    agents: Sequence[AgentRecord],
    delegations: Sequence[DelegationRecord],
    questions: Sequence[QuestionRecord],
    *,
    max_depth: int,
    max_questions_per_turn: int,
) -> None:
    """Check all TLA+ safety invariants against a persisted state snapshot."""
    agent_map: dict[str, AgentRecord] = {a.id: a for a in agents}
    roots = [a for a in agents if a.parent_id is None]
    if len(roots) != 1:
        raise InvariantViolationError(f"Expected exactly 1 root agent, found {len(roots)}")
    root = roots[0]
    if root.depth != 0:
        raise InvariantViolationError(f"Root agent must have depth 0, got {root.depth}")
    if root.status == AgentStatus.WAITING_CHILD:
        raise InvariantViolationError("RootNeverBlockedOnChildren violated: root agent must not enter WAITING_CHILD")

    # DepthBound + StrictTreeHierarchy + QuestionBudget
    for agent in agents:
        if not (0 <= agent.depth <= max_depth):
            raise InvariantViolationError(
                f"DepthBound violated for {agent.id}: depth={agent.depth}, max_depth={max_depth}"
            )
        if not (0 <= agent.question_count <= max_questions_per_turn):
            raise InvariantViolationError(
                f"Question budget violated for {agent.id}: count={agent.question_count}, max={max_questions_per_turn}"
            )
        if agent.parent_id is not None:
            if agent.parent_id == agent.id:
                raise InvariantViolationError(f"Self-parent cycle on {agent.id}")
            parent = agent_map.get(agent.parent_id)
            if parent is None:
                raise InvariantViolationError(f"Orphan agent {agent.id}: missing parent {agent.parent_id}")
            if agent.depth != parent.depth + 1:
                raise InvariantViolationError(
                    f"StrictTreeHierarchy violated: {agent.id}.depth={agent.depth} != {parent.id}.depth+1"
                )

    # WaitingHumanConsistency
    open_q_by_agent: dict[str, int] = {}
    for q in questions:
        if q.status == QuestionStatus.OPEN:
            open_q_by_agent[q.agent_id] = open_q_by_agent.get(q.agent_id, 0) + 1
    for agent in agents:
        has_open_q = open_q_by_agent.get(agent.id, 0) > 0
        if (agent.status == AgentStatus.WAITING_HUMAN) != has_open_q:
            raise InvariantViolationError(
                f"WaitingHumanConsistency violated for {agent.id}: "
                f"status={agent.status.value}, open_questions={open_q_by_agent.get(agent.id, 0)}"
            )

    # ActiveDelegationImpliesWaitingCaller + AtMostOneActiveIncomingDelegation + StrictTreeHierarchy
    active_incoming: dict[str, int] = {}
    for d in delegations:
        caller = agent_map.get(d.caller_id)
        callee = agent_map.get(d.callee_id)
        if caller is None or callee is None:
            raise InvariantViolationError(f"Delegation {d.id} references unknown agent")
        if caller.depth >= callee.depth:
            raise InvariantViolationError(
                f"Delegation {d.id} violates strict depth ordering: {caller.depth} >= {callee.depth}"
            )
        if d.status in ACTIVE_DELEGATION_STATUSES:
            if caller.parent_id is not None and caller.status != AgentStatus.WAITING_CHILD:
                raise InvariantViolationError(
                    f"ChildCallerWaitsForGrandchildren violated for {d.id}: "
                    f"non-root caller {caller.id} has status {caller.status.value}"
                )
            active_incoming[callee.id] = active_incoming.get(callee.id, 0) + 1
            if active_incoming[callee.id] > 1:
                raise InvariantViolationError(
                    f"AtMostOneActiveIncomingDelegation violated for callee {callee.id}"
                )


ALLOWED_MERGE_JOB_TRANSITIONS: dict[MergeJobStatus, frozenset[MergeJobStatus]] = {
    MergeJobStatus.QUEUED: frozenset(
        {
            MergeJobStatus.WAITING_CLEAN_BASE,
            MergeJobStatus.MERGING,
            MergeJobStatus.RUNNING_INSTRUCTION,
            MergeJobStatus.ABORTED,
        }
    ),
    MergeJobStatus.WAITING_CLEAN_BASE: frozenset({MergeJobStatus.QUEUED, MergeJobStatus.ABORTED}),
    MergeJobStatus.MERGING: frozenset(
        {MergeJobStatus.MERGED, MergeJobStatus.CONFLICT_RESOLVING, MergeJobStatus.ABORTED}
    ),
    MergeJobStatus.CONFLICT_RESOLVING: frozenset({MergeJobStatus.MERGED, MergeJobStatus.ABORTED}),
    MergeJobStatus.RUNNING_INSTRUCTION: frozenset({MergeJobStatus.MERGED, MergeJobStatus.ABORTED}),
    MergeJobStatus.MERGED: frozenset(),
    MergeJobStatus.ABORTED: frozenset(),
}


def validate_merge_job_transition(current: MergeJobStatus, target: MergeJobStatus) -> None:
    """Ensure merge job transition conforms to spec/PeanutsMergeQueue.tla."""
    if target not in ALLOWED_MERGE_JOB_TRANSITIONS[current]:
        raise StateTransitionError(f"Illegal merge job transition: {current.value} -> {target.value}")


def check_merge_queue_invariants(
    agents: Sequence[AgentRecord],
    merge_jobs: Sequence[MergeJobRecord],
    *,
    base_dirty: bool = False,
    base_merge_in_progress: bool = False,
) -> None:
    """Verify safety invariants from spec/PeanutsMergeQueue.tla."""
    active_jobs = [j for j in merge_jobs if j.status in ACTIVE_MERGE_STATUSES]
    if len(active_jobs) > 1:
        raise InvariantViolationError(
            f"SingleIntegratorWriter violated: {len(active_jobs)} merge jobs active simultaneously"
        )
    active_branch_merges = [
        j for j in active_jobs if j.status in {MergeJobStatus.MERGING, MergeJobStatus.CONFLICT_RESOLVING}
    ]
    if active_branch_merges and base_dirty:
        raise InvariantViolationError("NeverMergeIntoDirtyBase violated: branch merge active on dirty base")
    if not active_jobs and base_merge_in_progress:
        raise InvariantViolationError("CleanBaseWhenIntegratorIdle violated: MERGE_HEAD left behind")

    # DisjointWorkerBranches across worker agents
    seen_branches: dict[str, str] = {}
    for agent in agents:
        if agent.role_kind == "worker" and agent.branch_name:
            owner = seen_branches.get(agent.branch_name)
            if owner is not None and owner != agent.id:
                raise InvariantViolationError(
                    f"DisjointWorkerBranches violated: branch {agent.branch_name} shared by {owner} and {agent.id}"
                )
            seen_branches[agent.branch_name] = agent.id
