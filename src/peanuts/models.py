"""Pydantic domain models, structured agent actions, and API/WebSocket schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

ProviderName = Literal["claude", "codex", "mock"]


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class AgentStatus(StrEnum):
    """Agent states verified in spec/PeanutsOrchestrator.tla."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    WAITING_CHILD = "WAITING_CHILD"
    WAITING_HUMAN = "WAITING_HUMAN"
    FAILED = "FAILED"


class DelegationStatus(StrEnum):
    """Delegation states verified in spec/PeanutsOrchestrator.tla."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ACTIVE_DELEGATION_STATUSES: frozenset[DelegationStatus] = frozenset(
    {
        DelegationStatus.PENDING,
        DelegationStatus.IN_PROGRESS,
        DelegationStatus.NEEDS_HUMAN,
    }
)

SETTLED_DELEGATION_STATUSES: frozenset[DelegationStatus] = frozenset(
    {
        DelegationStatus.DONE,
        DelegationStatus.FAILED,
        DelegationStatus.CANCELLED,
    }
)


class QuestionStatus(StrEnum):
    OPEN = "OPEN"
    ANSWERED = "ANSWERED"
    CANCELLED = "CANCELLED"


MergeJobKind = Literal["branch_merge", "instruction"]


class MergeJobStatus(StrEnum):
    """Single-Writer Integrator merge queue states verified in spec/PeanutsMergeQueue.tla."""

    QUEUED = "QUEUED"
    WAITING_CLEAN_BASE = "WAITING_CLEAN_BASE"
    MERGING = "MERGING"
    CONFLICT_RESOLVING = "CONFLICT_RESOLVING"
    RUNNING_INSTRUCTION = "RUNNING_INSTRUCTION"
    MERGED = "MERGED"
    ABORTED = "ABORTED"


ACTIVE_MERGE_STATUSES: frozenset[MergeJobStatus] = frozenset(
    {
        MergeJobStatus.MERGING,
        MergeJobStatus.CONFLICT_RESOLVING,
        MergeJobStatus.RUNNING_INSTRUCTION,
    }
)


class MessageRole(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    PROGRESS = "progress"
    SYSTEM = "system"
    DELEGATION = "delegation"
    ANSWER = "answer"


class EventType(StrEnum):
    AGENT_CREATED = "agent_created"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    MESSAGE_ADDED = "message_added"
    DELEGATION_CREATED = "delegation_created"
    DELEGATION_UPDATED = "delegation_updated"
    QUESTION_CREATED = "question_created"
    QUESTION_ANSWERED = "question_answered"
    MERGE_JOB_CREATED = "merge_job_created"
    MERGE_JOB_UPDATED = "merge_job_updated"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Structured Agent Actions (returned by Claude / Codex / Mock turns)
# ---------------------------------------------------------------------------


class ReplyAction(BaseModel):
    action: Literal["reply"] = "reply"
    message: str


class DelegateTaskSpec(BaseModel):
    theme: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1)
    provider: ProviderName | None = None
    agent_id: str | None = None


class DelegateAction(BaseModel):
    action: Literal["delegate"] = "delegate"
    tasks: list[DelegateTaskSpec] = Field(min_length=1)
    message: str = ""


class AskHumanAction(BaseModel):
    action: Literal["ask_human"] = "ask_human"
    question: str = Field(min_length=1)


class IntegrateAction(BaseModel):
    action: Literal["integrate"] = "integrate"
    mode: Literal["merge", "resume", "instruction"] = "instruction"
    agent_id: str | None = None
    branch: str | None = None
    instruction: str = ""
    message: str = ""


AgentAction = Annotated[
    ReplyAction | DelegateAction | AskHumanAction | IntegrateAction,
    Field(discriminator="action"),
]


# ---------------------------------------------------------------------------
# Persisted Domain Records
# ---------------------------------------------------------------------------


class AgentRecord(BaseModel):
    id: str
    parent_id: str | None = None
    depth: int = Field(ge=0)
    theme: str
    provider: ProviderName
    provider_session_id: str | None = None
    role_kind: Literal["root", "worker", "integrator"] = "worker"
    worktree_path: str | None = None
    branch_name: str | None = None
    status: AgentStatus = AgentStatus.IDLE
    question_count: int = Field(default=0, ge=0)
    last_error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MergeJobRecord(BaseModel):
    id: str
    kind: MergeJobKind = "branch_merge"
    worker_agent_id: str
    theme: str
    source_branch: str
    target_branch: str
    instruction: str | None = None
    status: MergeJobStatus = MergeJobStatus.QUEUED
    conflicted_files: list[str] = Field(default_factory=list)
    summary: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MessageRecord(BaseModel):
    id: str
    agent_id: str
    role: MessageRole
    content: str
    theme: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DelegationRecord(BaseModel):
    id: str
    caller_id: str
    callee_id: str
    theme: str
    prompt: str
    status: DelegationStatus = DelegationStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class QuestionRecord(BaseModel):
    id: str
    agent_id: str
    delegation_id: str | None = None
    theme: str
    question: str
    answer: str | None = None
    status: QuestionStatus = QuestionStatus.OPEN
    created_at: datetime = Field(default_factory=utc_now)
    answered_at: datetime | None = None


class InboxItem(BaseModel):
    id: int
    agent_id: str
    kind: Literal["human_message", "human_answer", "children_settled", "child_result"]
    content: str
    theme: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EventRecord(BaseModel):
    seq: int
    type: EventType
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# REST / WebSocket API Schemas (React-ready OpenAPI contract)
# ---------------------------------------------------------------------------


class ProviderAvailability(BaseModel):
    name: ProviderName
    available: bool
    binary: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    root_agent_id: str
    base_branch: str | None = None
    worktrees_enabled: bool = False
    max_depth: int
    max_questions_per_turn: int
    providers: list[ProviderAvailability]


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1)
    target_agent_id: str | None = None
    theme: str | None = None
    provider: ProviderName | None = None


class SendMessageResponse(BaseModel):
    agent_id: str
    message: MessageRecord


class DirectDelegateRequest(BaseModel):
    callee_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    display_text: str | None = None


class DirectDelegateResponse(BaseModel):
    delegation: DelegationRecord
    human_message: MessageRecord
    route_message: MessageRecord


class AnswerQuestionRequest(BaseModel):
    answer: str = Field(min_length=1)


class UpdateProviderRequest(BaseModel):
    provider: ProviderName


class SystemStateResponse(BaseModel):
    root_agent_id: str
    base_branch: str | None = None
    worktrees_enabled: bool = False
    latest_seq: int
    agents: list[AgentRecord]
    delegations: list[DelegationRecord]
    open_questions: list[QuestionRecord]
    merge_jobs: list[MergeJobRecord] = Field(default_factory=list)
    recent_messages: list[MessageRecord]
