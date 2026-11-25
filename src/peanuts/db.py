"""SQLite persistence layer with atomic TLA+-aligned state transitions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from peanuts.models import (
    ACTIVE_DELEGATION_STATUSES,
    ACTIVE_MERGE_STATUSES,
    AgentRecord,
    AgentStatus,
    DelegationRecord,
    DelegationStatus,
    EventRecord,
    EventType,
    InboxItem,
    MergeJobRecord,
    MergeJobStatus,
    MessageRecord,
    MessageRole,
    ProviderName,
    QuestionRecord,
    QuestionStatus,
    utc_now,
)
from peanuts.state_machine import (
    StateTransitionError,
    validate_agent_transition,
    validate_delegation_depth,
    validate_delegation_transition,
    validate_merge_job_transition,
    validate_question_budget,
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES agents(id),
    depth INTEGER NOT NULL,
    theme TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_session_id TEXT,
    role_kind TEXT NOT NULL DEFAULT 'worker',
    worktree_path TEXT,
    branch_name TEXT,
    status TEXT NOT NULL,
    question_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merge_jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'branch_merge',
    worker_agent_id TEXT NOT NULL REFERENCES agents(id),
    theme TEXT NOT NULL,
    source_branch TEXT NOT NULL,
    target_branch TEXT NOT NULL,
    instruction TEXT,
    status TEXT NOT NULL,
    conflicted_files_json TEXT NOT NULL DEFAULT '[]',
    summary TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_merge_jobs_status ON merge_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    theme TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_agent_created ON messages(agent_id, created_at);

CREATE TABLE IF NOT EXISTS delegations (
    id TEXT PRIMARY KEY,
    caller_id TEXT NOT NULL REFERENCES agents(id),
    callee_id TEXT NOT NULL REFERENCES agents(id),
    theme TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_delegations_caller_status ON delegations(caller_id, status);
CREATE INDEX IF NOT EXISTS idx_delegations_callee_status ON delegations(callee_id, status);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    delegation_id TEXT REFERENCES delegations(id),
    theme TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    answered_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);

CREATE TABLE IF NOT EXISTS inbox_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    theme TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inbox_agent ON inbox_items(agent_id, id);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class PeanutsRepository:
    """Thread-safe SQLite repository storing multi-turn agent state and event log."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if str(db_path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.executescript(SCHEMA_SQL)
            self._migrate_schema()
            self._conn.commit()

    def _migrate_schema(self) -> None:
        """Add new columns safely when opening a pre-existing SQLite database."""
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(agents)").fetchall()}
        if "role_kind" not in cols:
            self._conn.execute("ALTER TABLE agents ADD COLUMN role_kind TEXT NOT NULL DEFAULT 'worker'")
            self._conn.execute("UPDATE agents SET role_kind = 'root' WHERE parent_id IS NULL")
        if "worktree_path" not in cols:
            self._conn.execute("ALTER TABLE agents ADD COLUMN worktree_path TEXT")
        if "branch_name" not in cols:
            self._conn.execute("ALTER TABLE agents ADD COLUMN branch_name TEXT")
        mj_cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(merge_jobs)").fetchall()}
        if "kind" not in mj_cols:
            self._conn.execute("ALTER TABLE merge_jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'branch_merge'")
        if "instruction" not in mj_cols:
            self._conn.execute("ALTER TABLE merge_jobs ADD COLUMN instruction TEXT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Row mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_agent(row: sqlite3.Row) -> AgentRecord:
        return AgentRecord(
            id=row["id"],
            parent_id=row["parent_id"],
            depth=row["depth"],
            theme=row["theme"],
            provider=row["provider"],
            provider_session_id=row["provider_session_id"],
            role_kind=row["role_kind"],
            worktree_path=row["worktree_path"],
            branch_name=row["branch_name"],
            status=AgentStatus(row["status"]),
            question_count=row["question_count"],
            last_error=row["last_error"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    @staticmethod
    def _row_to_merge_job(row: sqlite3.Row) -> MergeJobRecord:
        return MergeJobRecord(
            id=row["id"],
            kind=row["kind"],
            worker_agent_id=row["worker_agent_id"],
            theme=row["theme"],
            source_branch=row["source_branch"],
            target_branch=row["target_branch"],
            instruction=row["instruction"],
            status=MergeJobStatus(row["status"]),
            conflicted_files=json.loads(row["conflicted_files_json"]),
            summary=row["summary"],
            error=row["error"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=row["id"],
            agent_id=row["agent_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            theme=row["theme"],
            metadata=json.loads(row["metadata_json"]),
            created_at=_parse_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_delegation(row: sqlite3.Row) -> DelegationRecord:
        return DelegationRecord(
            id=row["id"],
            caller_id=row["caller_id"],
            callee_id=row["callee_id"],
            theme=row["theme"],
            prompt=row["prompt"],
            status=DelegationStatus(row["status"]),
            result=row["result"],
            error=row["error"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    @staticmethod
    def _row_to_question(row: sqlite3.Row) -> QuestionRecord:
        return QuestionRecord(
            id=row["id"],
            agent_id=row["agent_id"],
            delegation_id=row["delegation_id"],
            theme=row["theme"],
            question=row["question"],
            answer=row["answer"],
            status=QuestionStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]),
            answered_at=_parse_dt(row["answered_at"]) if row["answered_at"] else None,
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            seq=row["seq"],
            type=EventType(row["type"]),
            payload=json.loads(row["payload_json"]),
            created_at=_parse_dt(row["created_at"]),
        )

    def _append_event(self, conn: sqlite3.Connection, event_type: EventType, payload: dict[str, Any]) -> EventRecord:
        now = utc_now()
        cur = conn.execute(
            "INSERT INTO events (type, payload_json, created_at) VALUES (?, ?, ?)",
            (event_type.value, json.dumps(payload, default=str), _iso(now)),
        )
        seq = int(cur.lastrowid or 0)
        return EventRecord(seq=seq, type=event_type, payload=payload, created_at=now)

    def _enqueue_root_child_result(
        self,
        conn: sqlite3.Connection,
        delegation: DelegationRecord,
        *,
        status: DelegationStatus,
        outcome: str,
        now: datetime,
    ) -> None:
        """If delegation caller is the non-blocking root mediator, enqueue an async child_result event."""
        caller_row = conn.execute("SELECT parent_id FROM agents WHERE id = ?", (delegation.caller_id,)).fetchone()
        if caller_row is None or caller_row["parent_id"] is not None:
            return
        summary = (
            f"Child agent update — theme='{delegation.theme}' "
            f"(callee={delegation.callee_id}, status={status.value}):\n{outcome}"
        )
        conn.execute(
            "INSERT INTO inbox_items (agent_id, kind, content, theme, created_at) VALUES (?, ?, ?, ?, ?)",
            (delegation.caller_id, "child_result", summary, delegation.theme, _iso(now)),
        )

    def _insert_message(
        self,
        conn: sqlite3.Connection,
        *,
        agent_id: str,
        role: MessageRole,
        content: str,
        theme: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MessageRecord, EventRecord]:
        now = utc_now()
        msg = MessageRecord(
            id=_new_id("msg"),
            agent_id=agent_id,
            role=role,
            content=content,
            theme=theme,
            metadata=metadata or {},
            created_at=now,
        )
        conn.execute(
            """
            INSERT INTO messages (id, agent_id, role, content, theme, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg.id, msg.agent_id, msg.role.value, msg.content, msg.theme, json.dumps(msg.metadata), _iso(now)),
        )
        ev = self._append_event(conn, EventType.MESSAGE_ADDED, msg.model_dump(mode="json"))
        return msg, ev

    # ------------------------------------------------------------------
    # Bootstrap & Queries
    # ------------------------------------------------------------------

    def ensure_root_agent(
        self,
        provider: ProviderName,
        *,
        worktree_path: str | None = None,
        branch_name: str | None = None,
    ) -> tuple[AgentRecord, list[EventRecord]]:
        """Ensure singleton root parent agent (depth=0) exists."""
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM agents WHERE parent_id IS NULL LIMIT 1").fetchone()
            if row is not None:
                now = utc_now()
                if worktree_path is not None or branch_name is not None:
                    conn.execute(
                        "UPDATE agents SET role_kind = 'root', worktree_path = ?, branch_name = ?, updated_at = ? WHERE id = ?",
                        (worktree_path, branch_name, _iso(now), row["id"]),
                    )
                    row = conn.execute("SELECT * FROM agents WHERE id = ?", (row["id"],)).fetchone()
                return self._row_to_agent(row), []
            now = utc_now()
            root = AgentRecord(
                id="root",
                parent_id=None,
                depth=0,
                theme="root",
                provider=provider,
                role_kind="root",
                worktree_path=worktree_path,
                branch_name=branch_name,
                status=AgentStatus.IDLE,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO agents (
                    id, parent_id, depth, theme, provider, provider_session_id,
                    role_kind, worktree_path, branch_name,
                    status, question_count, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    root.id,
                    None,
                    0,
                    root.theme,
                    root.provider,
                    None,
                    root.role_kind,
                    root.worktree_path,
                    root.branch_name,
                    root.status.value,
                    0,
                    None,
                    _iso(now),
                    _iso(now),
                ),
            )
            ev = self._append_event(conn, EventType.AGENT_CREATED, root.model_dump(mode="json"))
            return root, [ev]

    def ensure_integrator_agent(
        self,
        provider: ProviderName,
        *,
        worktree_path: str,
        branch_name: str,
    ) -> tuple[AgentRecord, list[EventRecord]]:
        """Ensure singleton Integrator agent (Single-Writer on parent's base branch) exists."""
        with self._tx() as conn:
            now = utc_now()
            row = conn.execute("SELECT * FROM agents WHERE id = 'integrator' LIMIT 1").fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE agents SET worktree_path = ?, branch_name = ?, updated_at = ? WHERE id = 'integrator'",
                    (worktree_path, branch_name, _iso(now)),
                )
                updated = self._row_to_agent(
                    conn.execute("SELECT * FROM agents WHERE id = 'integrator'").fetchone()
                )
                return updated, []
            agent = AgentRecord(
                id="integrator",
                parent_id="root",
                depth=1,
                theme="integrator",
                provider=provider,
                role_kind="integrator",
                worktree_path=worktree_path,
                branch_name=branch_name,
                status=AgentStatus.IDLE,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO agents (
                    id, parent_id, depth, theme, provider, provider_session_id,
                    role_kind, worktree_path, branch_name,
                    status, question_count, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent.id,
                    agent.parent_id,
                    agent.depth,
                    agent.theme,
                    agent.provider,
                    None,
                    agent.role_kind,
                    agent.worktree_path,
                    agent.branch_name,
                    agent.status.value,
                    0,
                    None,
                    _iso(now),
                    _iso(now),
                ),
            )
            ev = self._append_event(conn, EventType.AGENT_CREATED, agent.model_dump(mode="json"))
            return agent, [ev]

    def set_agent_worktree(self, agent_id: str, worktree_path: str, branch_name: str) -> AgentRecord:
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE agents SET worktree_path = ?, branch_name = ?, updated_at = ? WHERE id = ?",
                (worktree_path, branch_name, _iso(now), agent_id),
            )
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise KeyError(f"Agent not found: {agent_id}")
            return self._row_to_agent(row)

    def recover_interrupted_turns(self) -> list[EventRecord]:
        """Recover from an abrupt crash where agents were left RUNNING in SQLite."""
        events: list[EventRecord] = []
        now = utc_now()
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM agents WHERE status = ?",
                (AgentStatus.RUNNING.value,),
            ).fetchall()
            for row in rows:
                agent = self._row_to_agent(row)
                conn.execute(
                    "UPDATE agents SET status = ?, last_error = ?, question_count = 0, updated_at = ? WHERE id = ?",
                    (AgentStatus.FAILED.value, "Interrupted by process restart", _iso(now), agent.id),
                )
                events.append(
                    self._append_event(
                        conn,
                        EventType.AGENT_STATUS_CHANGED,
                        {"agent_id": agent.id, "status": AgentStatus.FAILED.value, "error": "Interrupted by restart"},
                    )
                )
                del_rows = conn.execute(
                    "SELECT * FROM delegations WHERE callee_id = ? AND status = ?",
                    (agent.id, DelegationStatus.IN_PROGRESS.value),
                ).fetchall()
                for drow in del_rows:
                    delegation = self._row_to_delegation(drow)
                    conn.execute(
                        "UPDATE delegations SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                        (DelegationStatus.FAILED.value, "Interrupted by process restart", _iso(now), delegation.id),
                    )
                    self._enqueue_root_child_result(
                        conn,
                        delegation,
                        status=DelegationStatus.FAILED,
                        outcome="Interrupted by process restart",
                        now=now,
                    )
                    events.append(
                        self._append_event(
                            conn,
                            EventType.DELEGATION_UPDATED,
                            {"delegation_id": delegation.id, "status": DelegationStatus.FAILED.value},
                        )
                    )
        return events

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            return self._row_to_agent(row) if row else None

    def get_root_agent(self) -> AgentRecord:
        with self._lock:
            row = self._conn.execute("SELECT * FROM agents WHERE parent_id IS NULL LIMIT 1").fetchone()
            if row is None:
                raise RuntimeError("Root agent has not been initialized")
            return self._row_to_agent(row)

    def list_agents(self) -> list[AgentRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM agents ORDER BY depth ASC, created_at ASC").fetchall()
            return [self._row_to_agent(r) for r in rows]

    def list_children(self, parent_id: str) -> list[AgentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agents WHERE parent_id = ? ORDER BY created_at ASC",
                (parent_id,),
            ).fetchall()
            return [self._row_to_agent(r) for r in rows]

    def find_child_by_theme(self, parent_id: str, theme: str) -> AgentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM agents WHERE parent_id = ? AND lower(theme) = lower(?) LIMIT 1",
                (parent_id, theme),
            ).fetchone()
            return self._row_to_agent(row) if row else None

    def list_messages(self, agent_id: str | None = None, limit: int = 100) -> list[MessageRecord]:
        with self._lock:
            if agent_id is not None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?
                    ) ORDER BY created_at ASC
                    """,
                    (agent_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM (SELECT * FROM messages ORDER BY created_at DESC LIMIT ?) ORDER BY created_at ASC",
                    (limit,),
                ).fetchall()
            return [self._row_to_message(r) for r in rows]

    def list_delegations(self, caller_id: str | None = None) -> list[DelegationRecord]:
        with self._lock:
            if caller_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM delegations WHERE caller_id = ? ORDER BY created_at ASC",
                    (caller_id,),
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM delegations ORDER BY created_at ASC").fetchall()
            return [self._row_to_delegation(r) for r in rows]

    def get_delegation(self, delegation_id: str) -> DelegationRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation_id,)).fetchone()
            return self._row_to_delegation(row) if row else None

    def get_active_incoming_delegation(self, agent_id: str) -> DelegationRecord | None:
        placeholders = ",".join("?" for _ in ACTIVE_DELEGATION_STATUSES)
        params = [agent_id, *[s.value for s in ACTIVE_DELEGATION_STATUSES]]
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM delegations WHERE callee_id = ? AND status IN ({placeholders}) LIMIT 1",
                params,
            ).fetchone()
            return self._row_to_delegation(row) if row else None

    def list_active_outgoing_delegations(self, caller_id: str) -> list[DelegationRecord]:
        placeholders = ",".join("?" for _ in ACTIVE_DELEGATION_STATUSES)
        params = [caller_id, *[s.value for s in ACTIVE_DELEGATION_STATUSES]]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM delegations WHERE caller_id = ? AND status IN ({placeholders}) ORDER BY created_at ASC",
                params,
            ).fetchall()
            return [self._row_to_delegation(r) for r in rows]

    def list_pending_delegations(self) -> list[DelegationRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM delegations WHERE status = ? ORDER BY created_at ASC",
                (DelegationStatus.PENDING.value,),
            ).fetchall()
            return [self._row_to_delegation(r) for r in rows]

    def list_questions(self, only_open: bool = False) -> list[QuestionRecord]:
        with self._lock:
            if only_open:
                rows = self._conn.execute(
                    "SELECT * FROM questions WHERE status = ? ORDER BY created_at ASC",
                    (QuestionStatus.OPEN.value,),
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM questions ORDER BY created_at ASC").fetchall()
            return [self._row_to_question(r) for r in rows]

    def get_question(self, question_id: str) -> QuestionRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            return self._row_to_question(row) if row else None

    def list_events_after(self, after_seq: int = 0, limit: int = 500) -> list[EventRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE seq > ? ORDER BY seq ASC LIMIT ?",
                (after_seq, limit),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

    def latest_seq(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM events").fetchone()
            return int(row["m"]) if row else 0

    def list_idle_agents_with_inbox(self) -> list[AgentRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT a.*
                FROM agents a
                JOIN inbox_items i ON i.agent_id = a.id
                WHERE a.status = ?
                ORDER BY a.depth ASC, i.id ASC
                """,
                (AgentStatus.IDLE.value,),
            ).fetchall()
            return [self._row_to_agent(r) for r in rows]

    def list_waiting_parents_ready_to_resume(self) -> list[AgentRecord]:
        placeholders = ",".join("?" for _ in ACTIVE_DELEGATION_STATUSES)
        params = [AgentStatus.WAITING_CHILD.value, *[s.value for s in ACTIVE_DELEGATION_STATUSES]]
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT a.*
                FROM agents a
                WHERE a.parent_id IS NOT NULL
                  AND a.status = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM delegations d
                      WHERE d.caller_id = a.id AND d.status IN ({placeholders})
                  )
                ORDER BY a.depth DESC, a.updated_at ASC
                """,
                params,
            ).fetchall()
            return [self._row_to_agent(r) for r in rows]

    # ------------------------------------------------------------------
    # Atomic TLA+ State Transitions
    # ------------------------------------------------------------------

    def update_agent_provider(self, agent_id: str, provider: ProviderName) -> tuple[AgentRecord, list[EventRecord]]:
        now = utc_now()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise KeyError(f"Agent not found: {agent_id}")
            agent = self._row_to_agent(row)
            if agent.status == AgentStatus.RUNNING:
                raise StateTransitionError("Cannot change provider while agent is RUNNING")
            # Switching CLI provider invalidates the old provider's session_id
            new_session_id = agent.provider_session_id if agent.provider == provider else None
            conn.execute(
                "UPDATE agents SET provider = ?, provider_session_id = ?, updated_at = ? WHERE id = ?",
                (provider, new_session_id, _iso(now), agent_id),
            )
            updated = self._row_to_agent(conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone())
            ev = self._append_event(
                conn,
                EventType.AGENT_STATUS_CHANGED,
                {"agent_id": updated.id, "status": updated.status.value, "provider": updated.provider},
            )
            return updated, [ev]

    def append_progress(self, agent_id: str, text: str) -> tuple[MessageRecord, list[EventRecord]]:
        """Persist a live progress update for a RUNNING agent (no state transition)."""
        with self._tx() as conn:
            row = conn.execute("SELECT theme FROM agents WHERE id = ?", (agent_id,)).fetchone()
            theme = str(row["theme"]) if row is not None else None
            msg, ev = self._insert_message(
                conn,
                agent_id=agent_id,
                role=MessageRole.PROGRESS,
                content=text,
                theme=theme,
            )
            return msg, [ev]

    def enqueue_human_message(
        self,
        agent_id: str,
        content: str,
        *,
        theme: str | None = None,
    ) -> tuple[MessageRecord, list[EventRecord]]:
        """TLA+ HumanSendMessage: persist human message and push to agent inbox."""
        now = utc_now()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                raise KeyError(f"Agent not found: {agent_id}")
            msg, ev = self._insert_message(
                conn,
                agent_id=agent_id,
                role=MessageRole.HUMAN,
                content=content,
                theme=theme,
            )
            conn.execute(
                "INSERT INTO inbox_items (agent_id, kind, content, theme, created_at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, "human_message", content, theme, _iso(now)),
            )
            return msg, [ev]

    def start_inbox_turn(self, agent_id: str) -> tuple[AgentRecord, InboxItem, list[EventRecord]] | None:
        """TLA+ StartTurn: atomically transition IDLE -> RUNNING and pop earliest inbox item."""
        now = utc_now()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row is None:
                return None
            agent = self._row_to_agent(row)
            if agent.status != AgentStatus.IDLE:
                return None
            inbox_row = conn.execute(
                "SELECT * FROM inbox_items WHERE agent_id = ? ORDER BY id ASC LIMIT 1",
                (agent_id,),
            ).fetchone()
            if inbox_row is None:
                return None
            validate_agent_transition(agent.status, AgentStatus.RUNNING)
            conn.execute("DELETE FROM inbox_items WHERE id = ?", (inbox_row["id"],))
            conn.execute(
                "UPDATE agents SET status = ?, last_error = NULL, updated_at = ? WHERE id = ?",
                (AgentStatus.RUNNING.value, _iso(now), agent_id),
            )
            updated = self._row_to_agent(conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone())
            item = InboxItem(
                id=inbox_row["id"],
                agent_id=inbox_row["agent_id"],
                kind=inbox_row["kind"],
                content=inbox_row["content"],
                theme=inbox_row["theme"],
                created_at=_parse_dt(inbox_row["created_at"]),
            )
            ev = self._append_event(
                conn,
                EventType.AGENT_STATUS_CHANGED,
                {"agent_id": agent_id, "status": AgentStatus.RUNNING.value},
            )
            return updated, item, [ev]

    def start_delegation(self, delegation_id: str) -> tuple[AgentRecord, DelegationRecord, list[EventRecord]] | None:
        """TLA+ StartDelegation: atomically PENDING -> IN_PROGRESS and callee IDLE -> RUNNING."""
        now = utc_now()
        with self._tx() as conn:
            drow = conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation_id,)).fetchone()
            if drow is None:
                return None
            delegation = self._row_to_delegation(drow)
            if delegation.status != DelegationStatus.PENDING:
                return None
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (delegation.callee_id,)).fetchone()
            if arow is None:
                return None
            callee = self._row_to_agent(arow)
            if callee.status != AgentStatus.IDLE:
                return None
            validate_delegation_transition(delegation.status, DelegationStatus.IN_PROGRESS)
            validate_agent_transition(callee.status, AgentStatus.RUNNING)

            conn.execute(
                "UPDATE delegations SET status = ?, updated_at = ? WHERE id = ?",
                (DelegationStatus.IN_PROGRESS.value, _iso(now), delegation.id),
            )
            conn.execute(
                "UPDATE agents SET status = ?, question_count = 0, last_error = NULL, updated_at = ? WHERE id = ?",
                (AgentStatus.RUNNING.value, _iso(now), callee.id),
            )
            _, msg_ev = self._insert_message(
                conn,
                agent_id=callee.id,
                role=MessageRole.DELEGATION,
                content=delegation.prompt,
                theme=delegation.theme,
                metadata={"delegation_id": delegation.id, "caller_id": delegation.caller_id},
            )
            updated_agent = self._row_to_agent(
                conn.execute("SELECT * FROM agents WHERE id = ?", (callee.id,)).fetchone()
            )
            updated_del = self._row_to_delegation(
                conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
            )
            ev_agent = self._append_event(
                conn,
                EventType.AGENT_STATUS_CHANGED,
                {"agent_id": callee.id, "status": AgentStatus.RUNNING.value},
            )
            ev_del = self._append_event(
                conn,
                EventType.DELEGATION_UPDATED,
                updated_del.model_dump(mode="json"),
            )
            return updated_agent, updated_del, [msg_ev, ev_agent, ev_del]

    def finish_turn_reply(
        self,
        agent_id: str,
        reply_text: str,
        *,
        provider_session_id: str | None = None,
    ) -> tuple[AgentRecord, MessageRecord, list[EventRecord]]:
        """TLA+ FinishTurnReply: RUNNING -> IDLE, settle incoming IN_PROGRESS delegation to DONE."""
        now = utc_now()
        events: list[EventRecord] = []
        with self._tx() as conn:
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if arow is None:
                raise KeyError(f"Agent not found: {agent_id}")
            agent = self._row_to_agent(arow)
            validate_agent_transition(agent.status, AgentStatus.IDLE)
            session_id = provider_session_id or agent.provider_session_id

            conn.execute(
                """
                UPDATE agents
                SET status = ?, provider_session_id = ?, question_count = 0, last_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (AgentStatus.IDLE.value, session_id, _iso(now), agent_id),
            )
            msg, msg_ev = self._insert_message(
                conn,
                agent_id=agent_id,
                role=MessageRole.AGENT,
                content=reply_text,
                theme=agent.theme,
            )
            events.append(msg_ev)

            # Settle active incoming delegation if any
            drows = conn.execute(
                "SELECT * FROM delegations WHERE callee_id = ? AND status = ?",
                (agent_id, DelegationStatus.IN_PROGRESS.value),
            ).fetchall()
            for drow in drows:
                delegation = self._row_to_delegation(drow)
                validate_delegation_transition(delegation.status, DelegationStatus.DONE)
                conn.execute(
                    "UPDATE delegations SET status = ?, result = ?, updated_at = ? WHERE id = ?",
                    (DelegationStatus.DONE.value, reply_text, _iso(now), delegation.id),
                )
                self._enqueue_root_child_result(
                    conn,
                    delegation,
                    status=DelegationStatus.DONE,
                    outcome=reply_text,
                    now=now,
                )
                updated_del = self._row_to_delegation(
                    conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
                )
                events.append(
                    self._append_event(conn, EventType.DELEGATION_UPDATED, updated_del.model_dump(mode="json"))
                )

            updated_agent = self._row_to_agent(
                conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            )
            events.append(
                self._append_event(
                    conn,
                    EventType.AGENT_STATUS_CHANGED,
                    {
                        "agent_id": agent_id,
                        "status": AgentStatus.IDLE.value,
                        "provider_session_id": session_id,
                    },
                )
            )
            return updated_agent, msg, events

    def delegate_batch(
        self,
        caller_id: str,
        tasks: list[tuple[str, str, ProviderName, str | None]],
        *,
        note: str = "",
        provider_session_id: str | None = None,
        max_depth: int,
    ) -> tuple[AgentRecord, list[DelegationRecord], list[EventRecord]]:
        """TLA+ DelegateOne / DelegateTwo: RUNNING -> WAITING_CHILD + create/reuse child agents."""
        if not tasks:
            raise StateTransitionError("Cannot delegate an empty task list")
        now = utc_now()
        events: list[EventRecord] = []
        created_delegations: list[DelegationRecord] = []

        with self._tx() as conn:
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (caller_id,)).fetchone()
            if arow is None:
                raise KeyError(f"Caller agent not found: {caller_id}")
            caller = self._row_to_agent(arow)
            # Root mediator stays non-blocking (IDLE); non-root children wait for grandchildren (WAITING_CHILD).
            next_caller_status = AgentStatus.IDLE if caller.parent_id is None else AgentStatus.WAITING_CHILD
            validate_agent_transition(caller.status, next_caller_status)
            validate_delegation_depth(caller.depth, max_depth)

            session_id = provider_session_id or caller.provider_session_id
            used_callees: set[str] = set()
            placeholders = ",".join("?" for _ in ACTIVE_DELEGATION_STATUSES)

            def _is_reusable_child(candidate: AgentRecord) -> bool:
                if candidate.role_kind != "worker":
                    return False
                if candidate.parent_id != caller.id or candidate.id in used_callees:
                    return False
                if candidate.status != AgentStatus.IDLE:
                    return False
                active_cnt = conn.execute(
                    f"SELECT COUNT(*) AS c FROM delegations WHERE callee_id = ? AND status IN ({placeholders})",
                    [candidate.id, *[s.value for s in ACTIVE_DELEGATION_STATUSES]],
                ).fetchone()["c"]
                inbox_cnt = conn.execute(
                    "SELECT COUNT(*) AS c FROM inbox_items WHERE agent_id = ?",
                    (candidate.id,),
                ).fetchone()["c"]
                return active_cnt == 0 and inbox_cnt == 0

            for theme, prompt, provider, requested_agent_id in tasks:
                callee: AgentRecord | None = None
                if requested_agent_id is not None:
                    crow = conn.execute("SELECT * FROM agents WHERE id = ?", (requested_agent_id,)).fetchone()
                    if crow is not None:
                        candidate = self._row_to_agent(crow)
                        if _is_reusable_child(candidate):
                            callee = candidate
                if callee is None:
                    # Reuse idle existing worker child with the same theme & provider if available
                    crow = conn.execute(
                        """
                        SELECT * FROM agents
                        WHERE parent_id = ? AND role_kind = 'worker' AND lower(theme) = lower(?) AND provider = ? AND status = ?
                        LIMIT 1
                        """,
                        (caller.id, theme, provider, AgentStatus.IDLE.value),
                    ).fetchone()
                    if crow is not None:
                        candidate = self._row_to_agent(crow)
                        if _is_reusable_child(candidate):
                            callee = candidate

                if callee is None:
                    child_id = _new_id("agt")
                    callee = AgentRecord(
                        id=child_id,
                        parent_id=caller.id,
                        depth=caller.depth + 1,
                        theme=theme,
                        provider=provider,
                        status=AgentStatus.IDLE,
                        created_at=now,
                        updated_at=now,
                    )
                    conn.execute(
                        """
                        INSERT INTO agents (
                            id, parent_id, depth, theme, provider, provider_session_id,
                            status, question_count, last_error, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            callee.id,
                            callee.parent_id,
                            callee.depth,
                            callee.theme,
                            callee.provider,
                            None,
                            callee.status.value,
                            0,
                            None,
                            _iso(now),
                            _iso(now),
                        ),
                    )
                    events.append(
                        self._append_event(conn, EventType.AGENT_CREATED, callee.model_dump(mode="json"))
                    )

                used_callees.add(callee.id)
                del_rec = DelegationRecord(
                    id=_new_id("del"),
                    caller_id=caller.id,
                    callee_id=callee.id,
                    theme=theme,
                    prompt=prompt,
                    status=DelegationStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                )
                conn.execute(
                    """
                    INSERT INTO delegations (
                        id, caller_id, callee_id, theme, prompt, status, result, error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        del_rec.id,
                        del_rec.caller_id,
                        del_rec.callee_id,
                        del_rec.theme,
                        del_rec.prompt,
                        del_rec.status.value,
                        None,
                        None,
                        _iso(now),
                        _iso(now),
                    ),
                )
                created_delegations.append(del_rec)
                events.append(
                    self._append_event(conn, EventType.DELEGATION_CREATED, del_rec.model_dump(mode="json"))
                )

            if note:
                _, note_ev = self._insert_message(
                    conn,
                    agent_id=caller.id,
                    role=MessageRole.AGENT,
                    content=note,
                    theme=caller.theme,
                    metadata={"delegation_ids": [d.id for d in created_delegations]},
                )
                events.append(note_ev)

            conn.execute(
                "UPDATE agents SET status = ?, provider_session_id = ?, updated_at = ? WHERE id = ?",
                (next_caller_status.value, session_id, _iso(now), caller.id),
            )
            updated_caller = self._row_to_agent(
                conn.execute("SELECT * FROM agents WHERE id = ?", (caller.id,)).fetchone()
            )
            events.append(
                self._append_event(
                    conn,
                    EventType.AGENT_STATUS_CHANGED,
                    {
                        "agent_id": caller.id,
                        "status": next_caller_status.value,
                        "provider_session_id": session_id,
                    },
                )
            )
            return updated_caller, created_delegations, events

    def ask_human(
        self,
        agent_id: str,
        question_text: str,
        *,
        provider_session_id: str | None = None,
        max_questions_per_turn: int,
    ) -> tuple[AgentRecord, QuestionRecord, list[EventRecord]]:
        """TLA+ AskHuman: RUNNING -> WAITING_HUMAN (bounded by max_questions_per_turn)."""
        now = utc_now()
        events: list[EventRecord] = []
        with self._tx() as conn:
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if arow is None:
                raise KeyError(f"Agent not found: {agent_id}")
            agent = self._row_to_agent(arow)
            validate_agent_transition(agent.status, AgentStatus.WAITING_HUMAN)
            validate_question_budget(agent.question_count, max_questions_per_turn)

            session_id = provider_session_id or agent.provider_session_id
            drow = conn.execute(
                "SELECT * FROM delegations WHERE callee_id = ? AND status = ? LIMIT 1",
                (agent_id, DelegationStatus.IN_PROGRESS.value),
            ).fetchone()
            delegation_id: str | None = None
            if drow is not None:
                delegation = self._row_to_delegation(drow)
                validate_delegation_transition(delegation.status, DelegationStatus.NEEDS_HUMAN)
                delegation_id = delegation.id
                conn.execute(
                    "UPDATE delegations SET status = ?, updated_at = ? WHERE id = ?",
                    (DelegationStatus.NEEDS_HUMAN.value, _iso(now), delegation.id),
                )
                updated_del = self._row_to_delegation(
                    conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
                )
                events.append(
                    self._append_event(conn, EventType.DELEGATION_UPDATED, updated_del.model_dump(mode="json"))
                )

            q = QuestionRecord(
                id=_new_id("q"),
                agent_id=agent.id,
                delegation_id=delegation_id,
                theme=agent.theme,
                question=question_text,
                status=QuestionStatus.OPEN,
                created_at=now,
            )
            conn.execute(
                """
                INSERT INTO questions (id, agent_id, delegation_id, theme, question, answer, status, created_at, answered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (q.id, q.agent_id, q.delegation_id, q.theme, q.question, None, q.status.value, _iso(now), None),
            )
            conn.execute(
                """
                UPDATE agents
                SET status = ?, provider_session_id = ?, question_count = question_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (AgentStatus.WAITING_HUMAN.value, session_id, _iso(now), agent.id),
            )
            _, msg_ev = self._insert_message(
                conn,
                agent_id=agent.id,
                role=MessageRole.AGENT,
                content=f"[Question for human] {question_text}",
                theme=agent.theme,
                metadata={"question_id": q.id},
            )
            events.append(msg_ev)
            events.append(self._append_event(conn, EventType.QUESTION_CREATED, q.model_dump(mode="json")))
            updated_agent = self._row_to_agent(
                conn.execute("SELECT * FROM agents WHERE id = ?", (agent.id,)).fetchone()
            )
            events.append(
                self._append_event(
                    conn,
                    EventType.AGENT_STATUS_CHANGED,
                    {"agent_id": agent.id, "status": AgentStatus.WAITING_HUMAN.value},
                )
            )
            return updated_agent, q, events

    def answer_question(self, question_id: str, answer_text: str) -> tuple[AgentRecord, QuestionRecord, list[EventRecord]]:
        """TLA+ HumanAnswer: WAITING_HUMAN -> IDLE + push answer to inbox, NEEDS_HUMAN -> IN_PROGRESS."""
        now = utc_now()
        events: list[EventRecord] = []
        with self._tx() as conn:
            qrow = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            if qrow is None:
                raise KeyError(f"Question not found: {question_id}")
            q = self._row_to_question(qrow)
            if q.status != QuestionStatus.OPEN:
                raise StateTransitionError(f"Question {question_id} is already {q.status.value}")

            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (q.agent_id,)).fetchone()
            if arow is None:
                raise KeyError(f"Agent not found: {q.agent_id}")
            agent = self._row_to_agent(arow)
            validate_agent_transition(agent.status, AgentStatus.IDLE)

            conn.execute(
                "UPDATE questions SET answer = ?, status = ?, answered_at = ? WHERE id = ?",
                (answer_text, QuestionStatus.ANSWERED.value, _iso(now), q.id),
            )
            if q.delegation_id is not None:
                drow = conn.execute("SELECT * FROM delegations WHERE id = ?", (q.delegation_id,)).fetchone()
                if drow is not None:
                    delegation = self._row_to_delegation(drow)
                    if delegation.status == DelegationStatus.NEEDS_HUMAN:
                        validate_delegation_transition(delegation.status, DelegationStatus.IN_PROGRESS)
                        conn.execute(
                            "UPDATE delegations SET status = ?, updated_at = ? WHERE id = ?",
                            (DelegationStatus.IN_PROGRESS.value, _iso(now), delegation.id),
                        )
                        updated_del = self._row_to_delegation(
                            conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
                        )
                        events.append(
                            self._append_event(conn, EventType.DELEGATION_UPDATED, updated_del.model_dump(mode="json"))
                        )

            conn.execute(
                "UPDATE agents SET status = ?, updated_at = ? WHERE id = ?",
                (AgentStatus.IDLE.value, _iso(now), agent.id),
            )
            _, msg_ev = self._insert_message(
                conn,
                agent_id=agent.id,
                role=MessageRole.ANSWER,
                content=answer_text,
                theme=agent.theme,
                metadata={"question_id": q.id},
            )
            events.append(msg_ev)
            conn.execute(
                "INSERT INTO inbox_items (agent_id, kind, content, theme, created_at) VALUES (?, ?, ?, ?, ?)",
                (agent.id, "human_answer", f"Human answered your question ({q.question}): {answer_text}", agent.theme, _iso(now)),
            )

            updated_q = self._row_to_question(conn.execute("SELECT * FROM questions WHERE id = ?", (q.id,)).fetchone())
            updated_agent = self._row_to_agent(conn.execute("SELECT * FROM agents WHERE id = ?", (agent.id,)).fetchone())
            events.append(self._append_event(conn, EventType.QUESTION_ANSWERED, updated_q.model_dump(mode="json")))
            events.append(
                self._append_event(
                    conn,
                    EventType.AGENT_STATUS_CHANGED,
                    {"agent_id": agent.id, "status": AgentStatus.IDLE.value},
                )
            )
            return updated_agent, updated_q, events

    def fail_turn(
        self,
        agent_id: str,
        error_text: str,
        *,
        provider_session_id: str | None = None,
    ) -> tuple[AgentRecord, list[EventRecord]]:
        """TLA+ FailTurn: RUNNING -> FAILED and settle active incoming delegation to FAILED."""
        now = utc_now()
        events: list[EventRecord] = []
        with self._tx() as conn:
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if arow is None:
                raise KeyError(f"Agent not found: {agent_id}")
            agent = self._row_to_agent(arow)
            validate_agent_transition(agent.status, AgentStatus.FAILED)
            session_id = provider_session_id or agent.provider_session_id

            conn.execute(
                """
                UPDATE agents
                SET status = ?, provider_session_id = ?, question_count = 0, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (AgentStatus.FAILED.value, session_id, error_text, _iso(now), agent.id),
            )
            _, msg_ev = self._insert_message(
                conn,
                agent_id=agent.id,
                role=MessageRole.SYSTEM,
                content=f"[Error] {error_text}",
                theme=agent.theme,
            )
            events.append(msg_ev)

            drows = conn.execute(
                "SELECT * FROM delegations WHERE callee_id = ? AND status = ?",
                (agent.id, DelegationStatus.IN_PROGRESS.value),
            ).fetchall()
            for drow in drows:
                delegation = self._row_to_delegation(drow)
                validate_delegation_transition(delegation.status, DelegationStatus.FAILED)
                conn.execute(
                    "UPDATE delegations SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                    (DelegationStatus.FAILED.value, error_text, _iso(now), delegation.id),
                )
                self._enqueue_root_child_result(
                    conn,
                    delegation,
                    status=DelegationStatus.FAILED,
                    outcome=error_text,
                    now=now,
                )
                updated_del = self._row_to_delegation(
                    conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
                )
                events.append(
                    self._append_event(conn, EventType.DELEGATION_UPDATED, updated_del.model_dump(mode="json"))
                )

            updated_agent = self._row_to_agent(conn.execute("SELECT * FROM agents WHERE id = ?", (agent.id,)).fetchone())
            events.append(
                self._append_event(
                    conn,
                    EventType.AGENT_STATUS_CHANGED,
                    {"agent_id": agent.id, "status": AgentStatus.FAILED.value, "error": error_text},
                )
            )
            return updated_agent, events

    def reset_failed(self, agent_id: str) -> tuple[AgentRecord, list[EventRecord]]:
        """TLA+ ResetFailed: FAILED -> IDLE."""
        now = utc_now()
        with self._tx() as conn:
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if arow is None:
                raise KeyError(f"Agent not found: {agent_id}")
            agent = self._row_to_agent(arow)
            validate_agent_transition(agent.status, AgentStatus.IDLE)
            conn.execute(
                "UPDATE agents SET status = ?, last_error = NULL, updated_at = ? WHERE id = ?",
                (AgentStatus.IDLE.value, _iso(now), agent.id),
            )
            updated = self._row_to_agent(conn.execute("SELECT * FROM agents WHERE id = ?", (agent.id,)).fetchone())
            ev = self._append_event(
                conn,
                EventType.AGENT_STATUS_CHANGED,
                {"agent_id": agent.id, "status": AgentStatus.IDLE.value},
            )
            return updated, [ev]

    def cancel_delegation(self, delegation_id: str) -> tuple[DelegationRecord, list[EventRecord]]:
        """TLA+ CancelPendingDelegation (also cancels OPEN questions if NEEDS_HUMAN)."""
        now = utc_now()
        events: list[EventRecord] = []
        with self._tx() as conn:
            drow = conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation_id,)).fetchone()
            if drow is None:
                raise KeyError(f"Delegation not found: {delegation_id}")
            delegation = self._row_to_delegation(drow)
            validate_delegation_transition(delegation.status, DelegationStatus.CANCELLED)

            conn.execute(
                "UPDATE delegations SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (DelegationStatus.CANCELLED.value, "Cancelled by user", _iso(now), delegation.id),
            )
            self._enqueue_root_child_result(
                conn,
                delegation,
                status=DelegationStatus.CANCELLED,
                outcome="Cancelled by user",
                now=now,
            )
            # If the callee was waiting on a human question for this delegation, cancel that question and unblock callee
            if delegation.status == DelegationStatus.NEEDS_HUMAN:
                conn.execute(
                    "UPDATE questions SET status = ? WHERE delegation_id = ? AND status = ?",
                    (QuestionStatus.CANCELLED.value, delegation.id, QuestionStatus.OPEN.value),
                )
                conn.execute(
                    "UPDATE agents SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                    (AgentStatus.IDLE.value, _iso(now), delegation.callee_id, AgentStatus.WAITING_HUMAN.value),
                )
                events.append(
                    self._append_event(
                        conn,
                        EventType.AGENT_STATUS_CHANGED,
                        {"agent_id": delegation.callee_id, "status": AgentStatus.IDLE.value},
                    )
                )

            updated_del = self._row_to_delegation(
                conn.execute("SELECT * FROM delegations WHERE id = ?", (delegation.id,)).fetchone()
            )
            events.append(
                self._append_event(conn, EventType.DELEGATION_UPDATED, updated_del.model_dump(mode="json"))
            )
            return updated_del, events

    def resume_parent_after_children(self, parent_id: str) -> tuple[AgentRecord, list[EventRecord]] | None:
        """TLA+ ResumeAfterChildren: coalesced parent wake-up once ALL outgoing delegations settle."""
        now = utc_now()
        with self._tx() as conn:
            arow = conn.execute("SELECT * FROM agents WHERE id = ?", (parent_id,)).fetchone()
            if arow is None:
                return None
            parent = self._row_to_agent(arow)
            if parent.status != AgentStatus.WAITING_CHILD:
                return None

            placeholders = ",".join("?" for _ in ACTIVE_DELEGATION_STATUSES)
            active_cnt = conn.execute(
                f"SELECT COUNT(*) AS c FROM delegations WHERE caller_id = ? AND status IN ({placeholders})",
                [parent.id, *[s.value for s in ACTIVE_DELEGATION_STATUSES]],
            ).fetchone()["c"]
            if active_cnt > 0:
                return None

            validate_agent_transition(parent.status, AgentStatus.IDLE)

            # Gather settled delegations created since parent's last update (or all recent settled ones)
            settled_rows = conn.execute(
                """
                SELECT * FROM delegations
                WHERE caller_id = ? AND updated_at >= ?
                ORDER BY created_at ASC
                """,
                (parent.id, _iso(parent.updated_at)),
            ).fetchall()
            if not settled_rows:
                settled_rows = conn.execute(
                    "SELECT * FROM delegations WHERE caller_id = ? ORDER BY updated_at DESC LIMIT 5",
                    (parent.id,),
                ).fetchall()

            lines = ["All delegated child tasks have settled. Summary of results:"]
            for drow in settled_rows:
                d = self._row_to_delegation(drow)
                outcome = d.result if d.status == DelegationStatus.DONE else f"({d.status.value}) {d.error or ''}"
                lines.append(f"- [{d.theme} / callee={d.callee_id} / status={d.status.value}]: {outcome}")
            summary = "\n".join(lines)

            conn.execute(
                "UPDATE agents SET status = ?, updated_at = ? WHERE id = ?",
                (AgentStatus.IDLE.value, _iso(now), parent.id),
            )
            conn.execute(
                "INSERT INTO inbox_items (agent_id, kind, content, theme, created_at) VALUES (?, ?, ?, ?, ?)",
                (parent.id, "children_settled", summary, parent.theme, _iso(now)),
            )
            updated = self._row_to_agent(conn.execute("SELECT * FROM agents WHERE id = ?", (parent.id,)).fetchone())
            ev = self._append_event(
                conn,
                EventType.AGENT_STATUS_CHANGED,
                {"agent_id": parent.id, "status": AgentStatus.IDLE.value},
            )
            return updated, [ev]

    # ------------------------------------------------------------------
    # Single-Writer Merge Queue (spec/PeanutsMergeQueue.tla)
    # ------------------------------------------------------------------

    def list_merge_jobs(self) -> list[MergeJobRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM merge_jobs ORDER BY created_at ASC").fetchall()
            return [self._row_to_merge_job(r) for r in rows]

    def get_merge_job(self, job_id: str) -> MergeJobRecord | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM merge_jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_merge_job(row) if row else None

    def has_active_merge_job(self) -> bool:
        placeholders = ",".join("?" for _ in ACTIVE_MERGE_STATUSES)
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS c FROM merge_jobs WHERE status IN ({placeholders})",
                [s.value for s in ACTIVE_MERGE_STATUSES],
            ).fetchone()
            return int(row["c"]) > 0 if row else False

    def enqueue_merge_job(
        self,
        *,
        worker_agent_id: str,
        theme: str,
        source_branch: str,
        target_branch: str,
    ) -> tuple[MergeJobRecord, list[EventRecord]]:
        """Enqueue a committed worker branch onto the Single-Writer Integrator queue."""
        now = utc_now()
        with self._tx() as conn:
            # Avoid duplicate queued job for the exact same source_branch
            existing = conn.execute(
                "SELECT * FROM merge_jobs WHERE source_branch = ? AND status IN (?, ?) LIMIT 1",
                (source_branch, MergeJobStatus.QUEUED.value, MergeJobStatus.WAITING_CLEAN_BASE.value),
            ).fetchone()
            if existing is not None:
                return self._row_to_merge_job(existing), []
            job = MergeJobRecord(
                id=_new_id("mrg"),
                kind="branch_merge",
                worker_agent_id=worker_agent_id,
                theme=theme,
                source_branch=source_branch,
                target_branch=target_branch,
                instruction=None,
                status=MergeJobStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO merge_jobs (
                    id, kind, worker_agent_id, theme, source_branch, target_branch,
                    instruction, status, conflicted_files_json, summary, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.kind,
                    job.worker_agent_id,
                    job.theme,
                    job.source_branch,
                    job.target_branch,
                    None,
                    job.status.value,
                    "[]",
                    None,
                    None,
                    _iso(now),
                    _iso(now),
                ),
            )
            ev = self._append_event(conn, EventType.MERGE_JOB_CREATED, job.model_dump(mode="json"))
            return job, [ev]

    def enqueue_instruction_job(
        self,
        *,
        instruction: str,
        target_branch: str,
        theme: str = "integrator",
    ) -> tuple[MergeJobRecord, list[EventRecord]]:
        """Enqueue a root-dispatched git housekeeping instruction onto the Single-Writer queue."""
        now = utc_now()
        with self._tx() as conn:
            job = MergeJobRecord(
                id=_new_id("ins"),
                kind="instruction",
                worker_agent_id="root",
                theme=theme,
                source_branch=target_branch,
                target_branch=target_branch,
                instruction=instruction,
                status=MergeJobStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                """
                INSERT INTO merge_jobs (
                    id, kind, worker_agent_id, theme, source_branch, target_branch,
                    instruction, status, conflicted_files_json, summary, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.kind,
                    job.worker_agent_id,
                    job.theme,
                    job.source_branch,
                    job.target_branch,
                    job.instruction,
                    job.status.value,
                    "[]",
                    None,
                    None,
                    _iso(now),
                    _iso(now),
                ),
            )
            ev = self._append_event(conn, EventType.MERGE_JOB_CREATED, job.model_dump(mode="json"))
            return job, [ev]

    def supersede_integrator_dirty_questions(self) -> list[EventRecord]:
        """Atomically close stale Integrator dirty-base questions and transition WAITING_HUMAN -> IDLE.

        Verified in spec/PeanutsMergeQueue.tla (IntegratorStartInstruction / IntegratorQuestionConsistency).
        """
        events: list[EventRecord] = []
        now = utc_now()
        with self._tx() as conn:
            qrows = conn.execute(
                "SELECT * FROM questions WHERE agent_id = 'integrator' AND status = ?",
                (QuestionStatus.OPEN.value,),
            ).fetchall()
            for qrow in qrows:
                conn.execute(
                    "UPDATE questions SET answer = ?, status = ?, answered_at = ? WHERE id = ?",
                    ("Superseded by root integrate action", QuestionStatus.ANSWERED.value, _iso(now), qrow["id"]),
                )
                updated_q = self._row_to_question(
                    conn.execute("SELECT * FROM questions WHERE id = ?", (qrow["id"],)).fetchone()
                )
                events.append(self._append_event(conn, EventType.QUESTION_ANSWERED, updated_q.model_dump(mode="json")))

            arow = conn.execute("SELECT * FROM agents WHERE id = 'integrator'").fetchone()
            if arow is not None and self._row_to_agent(arow).status == AgentStatus.WAITING_HUMAN:
                validate_agent_transition(AgentStatus.WAITING_HUMAN, AgentStatus.IDLE)
                conn.execute(
                    "UPDATE agents SET status = ?, updated_at = ? WHERE id = 'integrator'",
                    (AgentStatus.IDLE.value, _iso(now)),
                )
                events.append(
                    self._append_event(
                        conn,
                        EventType.AGENT_STATUS_CHANGED,
                        {"agent_id": "integrator", "status": AgentStatus.IDLE.value},
                    )
                )
        return events

    def claim_next_queued_merge_job(self) -> MergeJobRecord | None:
        """Peek the earliest QUEUED merge job if no merge job is currently active."""
        if self.has_active_merge_job():
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM merge_jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                (MergeJobStatus.QUEUED.value,),
            ).fetchone()
            return self._row_to_merge_job(row) if row else None

    def transition_merge_job(
        self,
        job_id: str,
        target_status: MergeJobStatus,
        *,
        conflicted_files: list[str] | None = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> tuple[MergeJobRecord, list[EventRecord]]:
        now = utc_now()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM merge_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Merge job not found: {job_id}")
            job = self._row_to_merge_job(row)
            validate_merge_job_transition(job.status, target_status)
            if target_status in ACTIVE_MERGE_STATUSES and job.status not in ACTIVE_MERGE_STATUSES:
                placeholders = ",".join("?" for _ in ACTIVE_MERGE_STATUSES)
                active_cnt = conn.execute(
                    f"SELECT COUNT(*) AS c FROM merge_jobs WHERE status IN ({placeholders})",
                    [s.value for s in ACTIVE_MERGE_STATUSES],
                ).fetchone()["c"]
                if active_cnt > 0:
                    raise StateTransitionError("SingleIntegratorWriter: another merge job is already active")

            files_json = json.dumps(conflicted_files if conflicted_files is not None else job.conflicted_files)
            conn.execute(
                """
                UPDATE merge_jobs
                SET status = ?, conflicted_files_json = ?, summary = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    target_status.value,
                    files_json,
                    summary if summary is not None else job.summary,
                    error if error is not None else job.error,
                    _iso(now),
                    job_id,
                ),
            )
            updated = self._row_to_merge_job(
                conn.execute("SELECT * FROM merge_jobs WHERE id = ?", (job_id,)).fetchone()
            )
            ev = self._append_event(conn, EventType.MERGE_JOB_UPDATED, updated.model_dump(mode="json"))
            return updated, [ev]

    def resume_waiting_merge_jobs(self) -> list[EventRecord]:
        """Move WAITING_CLEAN_BASE jobs back to QUEUED once the parent's working tree is clean."""
        events: list[EventRecord] = []
        now = utc_now()
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM merge_jobs WHERE status = ? ORDER BY created_at ASC",
                (MergeJobStatus.WAITING_CLEAN_BASE.value,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE merge_jobs SET status = ?, updated_at = ? WHERE id = ?",
                    (MergeJobStatus.QUEUED.value, _iso(now), row["id"]),
                )
                updated = self._row_to_merge_job(
                    conn.execute("SELECT * FROM merge_jobs WHERE id = ?", (row["id"],)).fetchone()
                )
                events.append(self._append_event(conn, EventType.MERGE_JOB_UPDATED, updated.model_dump(mode="json")))
        return events

    def set_integrator_status(self, status: AgentStatus, *, error: str | None = None) -> list[EventRecord]:
        now = utc_now()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = 'integrator'").fetchone()
            if row is None:
                return []
            current = self._row_to_agent(row).status
            if current == status:
                return []
            validate_agent_transition(current, status)
            conn.execute(
                "UPDATE agents SET status = ?, last_error = ?, updated_at = ? WHERE id = 'integrator'",
                (status.value, error, _iso(now)),
            )
            ev = self._append_event(
                conn,
                EventType.AGENT_STATUS_CHANGED,
                {"agent_id": "integrator", "status": status.value},
            )
            return [ev]

    def notify_root_integration(self, content: str, *, theme: str) -> list[EventRecord]:
        """Log Integrator outcome and notify root mediator asynchronously."""
        now = utc_now()
        with self._tx() as conn:
            _, msg_ev = self._insert_message(
                conn,
                agent_id="integrator",
                role=MessageRole.AGENT,
                content=content,
                theme=theme,
            )
            conn.execute(
                "INSERT INTO inbox_items (agent_id, kind, content, theme, created_at) VALUES (?, ?, ?, ?, ?)",
                ("root", "child_result", content, theme, _iso(now)),
            )
            return [msg_ev]

    def recover_interrupted_merge_jobs(self) -> list[EventRecord]:
        """Abort any merge jobs left MERGING/CONFLICT_RESOLVING across a crash."""
        events: list[EventRecord] = []
        now = utc_now()
        placeholders = ",".join("?" for _ in ACTIVE_MERGE_STATUSES)
        with self._tx() as conn:
            rows = conn.execute(
                f"SELECT * FROM merge_jobs WHERE status IN ({placeholders})",
                [s.value for s in ACTIVE_MERGE_STATUSES],
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE merge_jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                    (MergeJobStatus.ABORTED.value, "Interrupted by process restart", _iso(now), row["id"]),
                )
                updated = self._row_to_merge_job(
                    conn.execute("SELECT * FROM merge_jobs WHERE id = ?", (row["id"],)).fetchone()
                )
                events.append(self._append_event(conn, EventType.MERGE_JOB_UPDATED, updated.model_dump(mode="json")))
        return events
