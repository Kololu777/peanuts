"""Pure @file and /command completion engine + TLA+-aligned popup state machine.

Zero UI framework dependencies: shared by FastAPI (`/api/v1/complete`), Textual TUI,
and future Web + React frontends.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CompletionKind = Literal["command", "file", "agent_switch", "agent_mention"]

_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "node_modules",
        ".peanuts",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


class CompletionItem(BaseModel):
    value: str
    label: str
    description: str = ""
    kind: CompletionKind


class CompletionResponse(BaseModel):
    kind: CompletionKind
    query: str
    items: list[CompletionItem] = Field(default_factory=list)


@dataclass(frozen=True)
class CompletionTrigger:
    kind: CompletionKind
    query: str
    start: int
    end: int


SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "コマンド・ショートカット一覧を表示"),
    ("/copy", "最新返答をクリップボードへコピー"),
    ("/copy code", "最新返答の最後のコードブロックをコピー"),
    ("/copy all", "表示中エージェントの会話全体をコピー"),
    ("/answer ", "確認待ちの質問に回答 (/answer <回答>)"),
    ("/merge", "選択中の子エージェントのブランチを統合キューへ投入"),
    ("/resume-merges", "dirty 待機中のマージキューを再開"),
    ("/provider claude", "選択中エージェントを Claude CLI に切替"),
    ("/provider codex", "選択中エージェントを Codex CLI に切替"),
    ("/provider mock", "選択中エージェントを Mock に切替"),
    ("/retry", "FAILED 状態のエージェントを IDLE に復帰"),
    ("/cancel ", "待機中/確認中の委譲をキャンセル (/cancel <id>)"),
    ("/root", "表示・送信先を親Agent（窓口）へ戻す"),
    ("/exit", "TUI を終了"),
    ("/quit", "TUI を終了"),
)


def detect_completion_trigger(text: str, cursor_pos: int) -> CompletionTrigger | None:
    """Detect active trigger token immediately before cursor_pos.

    Clean non-overlapping triggers:
      - `/...`  : slash commands / tools
      - `#...`  : switch viewed agent (`#00`, `#01`, `#02`)
      - `@#...` : direct message to agent (`@#02 <prompt>`)
      - `@...`  : workspace files (`@src/peanuts/tui.py`)
    """
    pos = max(0, min(cursor_pos, len(text)))
    prefix = text[:pos]
    if not prefix:
        return None

    # 1. Slash command at start of input
    if prefix.startswith("/"):
        if prefix.count(" ") <= 1:
            return CompletionTrigger(kind="command", query=prefix[1:], start=0, end=pos)
        return None

    # 2. `#01` agent view switch at start of input
    if prefix.startswith("#"):
        if " " not in prefix:
            return CompletionTrigger(kind="agent_switch", query=prefix[1:], start=0, end=pos)
        return None

    # 3. `@#...` (direct agent mention) or `@...` (workspace file) at token boundary
    token_start = pos
    while token_start > 0 and not prefix[token_start - 1].isspace():
        token_start -= 1
    token = prefix[token_start:pos]
    if token.startswith("@#"):
        return CompletionTrigger(kind="agent_mention", query=token[2:], start=token_start, end=pos)
    if token.startswith("@"):
        return CompletionTrigger(kind="file", query=token[1:], start=token_start, end=pos)

    return None


def apply_completion(text: str, trigger: CompletionTrigger, item: CompletionItem) -> tuple[str, int]:
    """Replace the active trigger slice with the chosen completion item and return (new_text, new_cursor)."""
    before = text[: trigger.start]
    after = text[trigger.end :]
    inserted = item.value
    needs_space = not inserted.endswith(" ") and (not after or not after[0].isspace())
    if item.kind == "agent_switch":
        needs_space = False
    elif item.kind == "command" and inserted in {
        "/help",
        "/copy",
        "/copy code",
        "/copy all",
        "/merge",
        "/resume-merges",
        "/retry",
        "/root",
        "/exit",
        "/quit",
    }:
        needs_space = False
    suffix = " " if needs_space else ""
    new_text = f"{before}{inserted}{suffix}{after}"
    new_cursor = len(before) + len(inserted) + len(suffix)
    return new_text, new_cursor


def _fuzzy_score(query: str, candidate: str) -> int | None:
    """Return lower-is-better rank for fuzzy matching, or None if not matched."""
    q = query.lower().strip()
    if not q:
        return 0
    cand_low = candidate.lower()
    basename = cand_low.rsplit("/", 1)[-1]

    if basename == q or cand_low == q:
        return 1
    if basename.startswith(q):
        return 10 + len(basename)
    if cand_low.startswith(q):
        return 20 + len(cand_low)
    idx_base = basename.find(q)
    if idx_base >= 0:
        return 30 + idx_base + len(basename)
    idx_full = cand_low.find(q)
    if idx_full >= 0:
        return 50 + idx_full + len(cand_low)

    # Ordered subsequence match (e.g. `sm.py` -> `src/peanuts/state_machine.py`)
    qi = 0
    gaps = 0
    last_match = -1
    for ci, ch in enumerate(cand_low):
        if ch == q[qi]:
            if last_match >= 0:
                gaps += ci - last_match - 1
            last_match = ci
            qi += 1
            if qi == len(q):
                return 100 + gaps + len(cand_low)
    return None


def complete_commands(query: str, limit: int = 12) -> list[CompletionItem]:
    """Return ranked slash-command completions."""
    scored: list[tuple[int, str, str]] = []
    for cmd, desc in SLASH_COMMANDS:
        score = _fuzzy_score(query, cmd.lstrip("/"))
        if score is not None:
            scored.append((score, cmd, desc))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [
        CompletionItem(value=cmd, label=cmd.strip(), description=desc, kind="command")
        for _, cmd, desc in scored[:limit]
    ]


def list_workspace_files(cwd: Path, max_files: int = 2000) -> list[str]:
    """List workspace files using `git ls-files` (respects .gitignore), falling back to os.walk."""
    root = cwd.resolve()
    proc = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        files: list[str] = []
        for line in proc.stdout.splitlines():
            rel = line.strip()
            if not rel or rel.startswith(".peanuts/"):
                continue
            files.append(rel)
            if len(files) >= max_files:
                break
        return sorted(set(files))

    # Non-git fallback
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORED_DIRS)
        rel_dir = Path(dirpath).relative_to(root)
        for fname in sorted(filenames):
            rel_path = str(rel_dir / fname) if str(rel_dir) != "." else fname
            found.append(rel_path)
            if len(found) >= max_files:
                return found
    return found


def complete_files(cwd: Path, query: str, limit: int = 12) -> list[CompletionItem]:
    """Return ranked `@file` path completions for the workspace."""
    files = list_workspace_files(cwd)
    scored: list[tuple[int, str]] = []
    for path in files:
        score = _fuzzy_score(query, path)
        if score is not None:
            scored.append((score, path))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [
        CompletionItem(value=f"@{path}", label=f"@{path}", description="", kind="file") for _, path in scored[:limit]
    ]


class CompletionStateMachine:
    """Exact Python implementation of spec/PeanutsCompletion.tla."""

    def __init__(self) -> None:
        self.popup_open: bool = False
        self.mode: Literal["NONE", "COMMAND", "FILE", "AGENT"] = "NONE"
        self.items: list[CompletionItem] = []
        self.selected_index: int = 0  # 0 when closed; 1..len(items) when open (matches TLA+)
        self.latest_req_id: int = 0
        self.invalidated_up_to: int = 0
        self.applied_req_id: int = 0
        self.active_trigger: CompletionTrigger | None = None

    def begin_query(self, trigger: CompletionTrigger) -> int:
        """TLA+ TypeTriggerQuery: allocate a new monotonic request ID."""
        self.latest_req_id += 1
        self.active_trigger = trigger
        return self.latest_req_id

    def clear_trigger(self) -> None:
        """TLA+ ClearTrigger: close popup and invalidate all in-flight queries."""
        self.invalidated_up_to = self.latest_req_id
        self._close_visual()

    def apply_response(self, req_id: int, trigger: CompletionTrigger, items: list[CompletionItem]) -> bool:
        """TLA+ ReceiveFreshResults / ReceiveFreshEmpty / DiscardStaleResponse.

        Returns True if UI state was updated, False if discarded as stale/invalidated.
        """
        if req_id < self.latest_req_id or req_id <= self.invalidated_up_to:
            return False
        self.applied_req_id = req_id
        if not items:
            self._close_visual()
            return True

        prev_open = self.popup_open
        prev_idx = self.selected_index
        self.popup_open = True
        if trigger.kind == "command":
            self.mode = "COMMAND"
        elif trigger.kind == "file":
            self.mode = "FILE"
        else:
            self.mode = "AGENT"
        self.items = list(items)
        self.active_trigger = trigger
        self.selected_index = prev_idx if (prev_open and 1 <= prev_idx <= len(self.items)) else 1
        return True

    def move_down(self) -> None:
        if not self.popup_open or not self.items:
            return
        self.selected_index = self.selected_index + 1 if self.selected_index < len(self.items) else 1

    def move_up(self) -> None:
        if not self.popup_open or not self.items:
            return
        self.selected_index = self.selected_index - 1 if self.selected_index > 1 else len(self.items)

    def accept(self, index_1based: int | None = None) -> tuple[CompletionTrigger, CompletionItem] | None:
        """TLA+ AcceptCompletion: accept selected candidate, close popup, invalidate in-flight queries."""
        if not self.popup_open or not self.items or self.active_trigger is None:
            return None
        idx = index_1based if index_1based is not None else self.selected_index
        if not (1 <= idx <= len(self.items)):
            return None
        chosen = (self.active_trigger, self.items[idx - 1])
        self.invalidated_up_to = self.latest_req_id
        self._close_visual()
        return chosen

    def dismiss(self) -> None:
        """TLA+ DismissEscape: close popup and invalidate in-flight queries."""
        self.invalidated_up_to = self.latest_req_id
        self._close_visual()

    def _close_visual(self) -> None:
        self.popup_open = False
        self.mode = "NONE"
        self.items = []
        self.selected_index = 0
        self.active_trigger = None

    def check_invariants(self) -> None:
        """Runtime assertion of all safety invariants from spec/PeanutsCompletion.tla."""
        if self.popup_open:
            assert self.mode != "NONE", "PopupHasCandidates violated: mode=NONE while open"
            assert len(self.items) > 0, "PopupHasCandidates violated: 0 items while open"
            assert 1 <= self.selected_index <= len(self.items), "SelectedIndexInBounds violated"
            assert self.applied_req_id > self.invalidated_up_to, "MonotonicAppliedResponses violated"
        else:
            assert self.mode == "NONE", "ClosedStateClean violated: mode != NONE"
            assert self.items == [], "ClosedStateClean violated: items not empty"
            assert self.selected_index == 0, "ClosedStateClean violated: selected_index != 0"
        assert self.applied_req_id <= self.latest_req_id
        assert self.invalidated_up_to <= self.latest_req_id
