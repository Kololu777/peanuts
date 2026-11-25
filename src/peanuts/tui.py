"""Textual TUI client for Peanuts with native mouse text selection and Markdown rendering.

Speaks strictly to the FastAPI REST + WebSocket API via PeanutsClient,
so a Web + React UI can replace it without touching any orchestrator logic.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, OptionList, Static, Tree
from textual.widgets import Markdown as TextualMarkdown
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode

from peanuts.client import PeanutsClient
from peanuts.completion import (
    CompletionItem,
    CompletionStateMachine,
    CompletionTrigger,
    apply_completion,
    detect_completion_trigger,
)
from peanuts.models import (
    ACTIVE_DELEGATION_STATUSES,
    AgentRecord,
    AgentStatus,
    DelegationRecord,
    EventType,
    MergeJobRecord,
    MergeJobStatus,
    MessageRecord,
    MessageRole,
    ProviderName,
    QuestionRecord,
    SystemStateResponse,
)

_STATUS_BADGE: dict[AgentStatus, tuple[str, str]] = {
    AgentStatus.IDLE: ("IDLE", "green"),
    AgentStatus.RUNNING: ("RUNNING", "bold yellow"),
    AgentStatus.WAITING_CHILD: ("WAIT_SUB", "cyan"),
    AgentStatus.WAITING_HUMAN: ("WAIT_HUMAN", "bold magenta"),
    AgentStatus.FAILED: ("FAILED", "bold red"),
}

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_DIRECT_AGENT_MENTION_RE = re.compile(r"^@#(\d{1,2})\s+(.+)$", re.DOTALL)


def _fmt_time(dt: datetime) -> str:
    return dt.astimezone().strftime("%H:%M:%S")


def _copy_to_system_clipboard(text: str) -> str | None:
    """Copy text to the OS clipboard when a clipboard utility is available. Returns tool name or None."""
    candidates: list[tuple[str, list[str]]] = []
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        candidates.append(("wl-copy", ["wl-copy"]))
    if os.environ.get("DISPLAY"):
        if shutil.which("xclip"):
            candidates.append(("xclip", ["xclip", "-selection", "clipboard"]))
        if shutil.which("xsel"):
            candidates.append(("xsel", ["xsel", "--clipboard", "--input"]))
    if shutil.which("pbcopy"):
        candidates.append(("pbcopy", ["pbcopy"]))
    if shutil.which("wl-copy") and all(name != "wl-copy" for name, _ in candidates):
        candidates.append(("wl-copy", ["wl-copy"]))

    for name, cmd in candidates:
        try:
            proc = subprocess.run(cmd, input=text, text=True, capture_output=True, timeout=2.0, check=False)
            if proc.returncode == 0:
                return name
        except Exception:
            continue
    return None


def _extract_last_code_block(text: str) -> str | None:
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


class _AgentTree(Tree[str]):
    """Tree without default ▶/▼ prefix so role marks (◆ / ✦ / ▸ / ▹) stay crisp."""

    ICON_NODE = ""
    ICON_NODE_EXPANDED = ""


class PeanutsTUI(App[None]):
    """Copy-friendly parent-child agent mediator TUI with native drag-selection."""

    TITLE = "Peanuts"
    SUB_TITLE = "Parent / Child Agent Mediator"
    CSS = """
    Screen {
        background: $surface;
    }
    #body {
        height: 1fr;
    }
    #sidebar {
        width: 38;
        border-right: solid $panel-lighten-1;
        padding: 1 1 0 1;
        background: $surface-darken-1;
    }
    #sidebar.hidden {
        display: none;
    }
    #sidebar_title {
        height: auto;
        padding-bottom: 1;
    }
    #agent_tree {
        height: 1fr;
        background: transparent;
        padding: 0;
    }
    #active_tasks_box {
        height: auto;
        max-height: 11;
        border-top: solid $panel-lighten-1;
        padding: 1 0;
    }
    #main {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    #view_header {
        height: auto;
        padding: 1 1 0 1;
        border-bottom: solid $panel-lighten-1;
    }
    #chat_log {
        height: 1fr;
        padding: 0 1;
        background: $surface;
    }
    .msg-header {
        margin-top: 1;
        padding-top: 1;
        border-top: solid $panel-lighten-1;
        height: auto;
    }
    .msg-body {
        padding: 0 0 0 2;
        height: auto;
    }
    .msg-markdown {
        padding: 0 0 0 2;
        margin: 0;
        height: auto;
    }
    .msg-progress {
        padding: 0 0 0 2;
        height: auto;
    }
    .msg-notice {
        margin-top: 1;
        padding: 0 1;
        height: auto;
    }
    #questions_box {
        height: auto;
        max-height: 8;
        border-top: solid magenta;
        padding: 0 1;
        margin-top: 1;
        display: none;
    }
    #questions_box.visible {
        display: block;
    }
    #completion_box {
        height: auto;
        max-height: 9;
        border: solid $primary;
        background: $surface-darken-1;
        margin-top: 1;
        display: none;
    }
    #completion_box.visible {
        display: block;
    }
    #prompt_input {
        dock: bottom;
        margin: 1 0 1 0;
    }
    """

    BINDINGS = [
        ("ctrl+y", "copy_last_reply", "Copy Reply"),
        ("ctrl+b", "toggle_sidebar", "Sidebar"),
        ("ctrl+r", "refresh_state", "Refresh"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, server_url: str) -> None:
        super().__init__()
        self.server_url = server_url
        self.client = PeanutsClient(server_url)
        self._state: SystemStateResponse | None = None
        self._selected_agent_id: str | None = None
        self._seen_message_ids: set[str] = set()
        self._displayed_messages: list[MessageRecord] = []
        self._last_tree_fingerprint: tuple[object, ...] | None = None
        self._last_auto_copied: str | None = None
        self._short_id_by_agent: dict[str, str] = {}
        self._agent_by_short_id: dict[str, str] = {}
        self._completion = CompletionStateMachine()
        self._suppress_completion_once: bool = False
        self._ws_task: asyncio.Task[None] | None = None

    def on_mouse_up(self, _event: events.MouseUp) -> None:
        """Auto-copy selected text to clipboard as soon as drag-selection finishes."""
        self.call_after_refresh(self._auto_copy_selection)

    def _auto_copy_selection(self) -> None:
        selected = self.screen.get_selected_text()
        if not selected:
            self._last_auto_copied = None
            return
        if selected == self._last_auto_copied:
            return
        self._last_auto_copied = selected
        self._copy_text(selected, notify=False)

    def _copy_text(self, text: str, *, notify: bool = False) -> None:
        super().copy_to_clipboard(text)
        _copy_to_system_clipboard(text)
        if notify:
            self.notify(f"コピー ({len(text)}文字)", timeout=1.5)

    def copy_to_clipboard(self, text: str) -> None:
        """Mirror Textual's copy into OS clipboard silently (no noisy toast on drag)."""
        self._copy_text(text, notify=False)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static(
                    Text.assemble(
                        ("エージェント\n", "bold"),
                        ("◆親  ✦統合  ▸子  (#00で遷移)", "dim"),
                    ),
                    id="sidebar_title",
                )
                yield _AgentTree("root", id="agent_tree")
                yield Static(Text("稼働中タスク / マージ待ち: なし", style="dim"), id="active_tasks_box")
            with Vertical(id="main"):
                yield Static("", id="view_header")
                yield VerticalScroll(id="chat_log")
                yield Static("", id="questions_box")
                yield OptionList(id="completion_box")
                yield Input(
                    placeholder="◆ 00 root (@ファイル, /コマンド, #01遷移, @#01直接送信)",
                    id="prompt_input",
                )
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_state(reload_messages=True)
        self._ws_task = asyncio.create_task(self._listen_events())
        self.query_one("#prompt_input", Input).focus()

    async def on_unmount(self) -> None:
        if self._ws_task is not None:
            self._ws_task.cancel()
        await self.client.aclose()

    async def action_refresh_state(self) -> None:
        await self._refresh_state(reload_messages=True)

    def action_toggle_sidebar(self) -> None:
        """Hide/show the left sidebar."""
        sidebar = self.query_one("#sidebar", Vertical)
        sidebar.toggle_class("hidden")

    def action_copy_last_reply(self) -> None:
        """Copy the latest agent reply (raw Markdown) to clipboard."""
        self._copy_messages(mode="last")

    def _copy_messages(self, mode: str) -> None:
        agent_replies = [m for m in self._displayed_messages if m.role == MessageRole.AGENT]
        if mode == "last":
            if not agent_replies:
                self.notify("コピーできる返答がまだありません。", severity="warning")
                return
            self._copy_text(agent_replies[-1].content, notify=True)
            return
        if mode == "code":
            for msg in reversed(agent_replies):
                snippet = _extract_last_code_block(msg.content)
                if snippet is not None:
                    self._copy_text(snippet, notify=True)
                    return
            self.notify("最新の返答にコードブロックが見つかりません。", severity="warning")
            return
        if mode == "all":
            if not self._displayed_messages:
                self.notify("コピーできるログがありません。", severity="warning")
                return
            parts = [
                f"[{m.role.value} {_fmt_time(m.created_at)}]\n{m.content}"
                for m in self._displayed_messages
                if m.role != MessageRole.PROGRESS
            ]
            self._copy_text("\n\n---\n\n".join(parts), notify=True)
            return
        self.notify("使い方: /copy | /copy code | /copy all", severity="warning")

    async def _append_notice(self, content: Text | str, *, markdown: bool = False) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        widget: Widget = (
            TextualMarkdown(str(content), classes="msg-markdown") if markdown else Static(content, classes="msg-notice")
        )
        await log.mount(widget)
        log.scroll_end(animate=False)

    async def _listen_events(self) -> None:
        after_seq = self._state.latest_seq if self._state else 0
        while True:
            try:
                async for event in self.client.stream_events(after_seq=after_seq):
                    after_seq = max(after_seq, event.seq)
                    if event.type == EventType.MESSAGE_ADDED:
                        try:
                            msg = MessageRecord.model_validate(event.payload)
                        except Exception:
                            continue
                        if msg.agent_id == self._selected_agent_id:
                            agents = self._state.agents if self._state else []
                            await self._append_messages([msg], agents)
                    else:
                        await self._refresh_state(reload_messages=False)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                await self._append_notice(Text(f"WS reconnect in 1s ({exc})", style="red"))
                await asyncio.sleep(1.0)

    def _rebuild_short_ids(self, agents: list[AgentRecord]) -> None:
        """Assign stable 2-digit short IDs: 00=root, 01=integrator (if any), 02+=workers."""
        ordered: list[AgentRecord] = []
        ordered.extend(a for a in agents if a.role_kind == "root")
        ordered.extend(a for a in agents if a.role_kind == "integrator")
        ordered.extend(a for a in agents if a.role_kind == "worker")
        self._short_id_by_agent = {}
        self._agent_by_short_id = {}
        for idx, agent in enumerate(ordered):
            short = f"{idx:02d}"
            self._short_id_by_agent[agent.id] = short
            self._agent_by_short_id[short] = agent.id
            self._agent_by_short_id[str(idx)] = agent.id

    def _resolve_agent_ref(self, ref: str) -> str | None:
        cleaned = ref.strip().lstrip("/#")
        if cleaned in self._agent_by_short_id:
            return self._agent_by_short_id[cleaned]
        if self._state is not None:
            for a in self._state.agents:
                if a.id == cleaned or a.theme.lower() == cleaned.lower():
                    return a.id
        return None

    async def _refresh_state(self, *, reload_messages: bool) -> None:
        state = await self.client.get_state(message_limit=50)
        self._state = state
        self._rebuild_short_ids(state.agents)
        if self._selected_agent_id is None:
            self._selected_agent_id = state.root_agent_id
        self._render_tree_if_changed(state.agents, state.delegations)
        self._render_active_tasks(state.delegations, state.merge_jobs)
        self._render_questions(state.open_questions)
        self._render_header_and_target(state.agents)
        if reload_messages and self._selected_agent_id is not None:
            await self._load_selected_agent_messages(self._selected_agent_id)

    async def _load_selected_agent_messages(self, agent_id: str) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        agents = self._state.agents if self._state else []
        messages = await self.client.get_agent_messages(agent_id, limit=200)

        await log.remove_children()
        self._seen_message_ids.clear()
        self._displayed_messages.clear()
        if not messages:
            await log.mount(
                Static(
                    Text(
                        "まだメッセージはありません。下の入力欄から依頼を送ってみてください。",
                        style="dim",
                    ),
                    classes="msg-notice",
                )
            )
        else:
            await self._append_messages(messages, agents)

    def _render_tree_if_changed(self, agents: list[AgentRecord], delegations: list[DelegationRecord]) -> None:
        active_root_children = sum(
            1 for d in delegations if d.caller_id == "root" and d.status in ACTIVE_DELEGATION_STATUSES
        )
        fingerprint: tuple[object, ...] = (
            active_root_children,
            tuple((a.id, a.parent_id, a.theme, a.provider, a.status.value) for a in agents),
        )
        if fingerprint == self._last_tree_fingerprint:
            return
        self._last_tree_fingerprint = fingerprint

        tree: Tree[str] = self.query_one("#agent_tree", Tree)
        tree.clear()
        by_parent: dict[str | None, list[AgentRecord]] = {}
        for agent in agents:
            by_parent.setdefault(agent.parent_id, []).append(agent)

        roots = by_parent.get(None, [])
        if not roots:
            return
        root_agent = roots[0]
        tree.root.label = self._format_agent_label(root_agent, active_children=active_root_children)
        tree.root.data = root_agent.id

        def add_children(node: TreeNode[str], parent_id: str) -> None:
            children = by_parent.get(parent_id, [])
            # Show integrator right under root before workers
            children.sort(key=lambda a: (0 if a.role_kind == "integrator" else 1, a.created_at))
            for child in children:
                child_node = node.add(self._format_agent_label(child), data=child.id, expand=True)
                add_children(child_node, child.id)

        add_children(tree.root, root_agent.id)
        tree.root.expand()

    def _format_agent_label(self, agent: AgentRecord, *, active_children: int = 0) -> Text:
        badge_text, badge_style = _STATUS_BADGE.get(agent.status, (agent.status.value, "white"))
        short = self._short_id_by_agent.get(agent.id, "--")
        label = Text()
        if agent.role_kind == "root":
            label.append("◆ ", style="bold green")
            label.append(f"{short} ", style="bold dim")
            label.append("root ", style="bold")
        elif agent.role_kind == "integrator":
            label.append("✦ ", style="bold magenta")
            label.append(f"{short} ", style="bold dim")
            label.append("integrator ", style="bold magenta")
        else:
            mark = "▸ " if agent.depth == 1 else "▹ "
            label.append(mark, style="bold cyan" if agent.depth == 1 else "cyan")
            label.append(f"{short} ", style="bold dim")
            label.append(f"{agent.theme} ", style="bold cyan" if agent.depth == 1 else "cyan")
        label.append(f"[{agent.provider}] ", style="dim")
        label.append(badge_text, style=badge_style)
        if agent.role_kind == "root" and active_children > 0:
            label.append(f" ({active_children})", style="cyan")
        return label

    def _render_active_tasks(
        self,
        delegations: list[DelegationRecord],
        merge_jobs: list[MergeJobRecord],
    ) -> None:
        box = self.query_one("#active_tasks_box", Static)
        active = [d for d in delegations if d.status in ACTIVE_DELEGATION_STATUSES]
        pending_merges = [
            j
            for j in merge_jobs
            if j.status
            in {
                MergeJobStatus.QUEUED,
                MergeJobStatus.WAITING_CLEAN_BASE,
                MergeJobStatus.MERGING,
                MergeJobStatus.CONFLICT_RESOLVING,
                MergeJobStatus.RUNNING_INSTRUCTION,
            }
        ]
        if not active and not pending_merges:
            box.update(Text("稼働中タスク / マージ待ち: なし", style="dim"))
            return
        txt = Text()
        if active:
            txt.append(f"稼働中の子タスク ({len(active)}件)\n", style="bold cyan")
            for d in active[:4]:
                short_prompt = " ".join(d.prompt.split())
                if len(short_prompt) > 24:
                    short_prompt = short_prompt[:23] + "…"
                txt.append(f"• {d.theme} ", style="bold")
                txt.append(f"({d.status.value})\n", style="yellow")
                txt.append(f"  {short_prompt}\n", style="dim")
        if pending_merges:
            txt.append(f"マージキュー ({len(pending_merges)}件)\n", style="bold magenta")
            for j in pending_merges[:3]:
                txt.append(f"• {j.theme} ", style="bold")
                txt.append(f"[{j.status.value}]\n", style="magenta")
        box.update(txt)

    def _render_header_and_target(self, agents: list[AgentRecord]) -> None:
        header = self.query_one("#view_header", Static)
        prompt_input = self.query_one("#prompt_input", Input)
        target_id = self._selected_agent_id or "root"
        agent = next((a for a in agents if a.id == target_id), None)
        base_branch = self._state.base_branch if self._state else None
        branch_part = f"  branch:{base_branch}" if base_branch else ""

        short = self._short_id_by_agent.get(agent.id, "00") if agent else "00"

        if agent is None or agent.role_kind == "root":
            prov = agent.provider if agent else "-"
            status_str, status_style = (
                _STATUS_BADGE.get(agent.status, (agent.status.value, "white")) if agent else ("-", "dim")
            )
            header.update(
                Text.assemble(
                    (f"◆ {short} root ", "bold green"),
                    (f"[{prov}] ", "dim"),
                    (status_str, status_style),
                    (f"{branch_part}   API: {self.server_url}", "dim"),
                )
            )
            prompt_input.placeholder = "◆ 00 root (@ファイル, /コマンド, #01遷移, @#01直接送信)"
        elif agent.role_kind == "integrator":
            status_str, status_style = _STATUS_BADGE.get(agent.status, (agent.status.value, "white"))
            header.update(
                Text.assemble(
                    (f"✦ {short} integrator ", "bold magenta"),
                    (f"[{agent.provider}] ", "dim"),
                    (status_str, status_style),
                    (f"  統合先:{agent.branch_name or '-'}", "dim"),
                )
            )
            prompt_input.placeholder = "✦ integrator 閲覧中 (/resume-merges, #00 で親へ)"
        else:
            status_str, status_style = _STATUS_BADGE.get(agent.status, (agent.status.value, "white"))
            wt_info = f"  branch:{agent.branch_name}" if agent.branch_name else f"  id={agent.id}"
            mark = "▸" if agent.depth == 1 else "▹"
            header.update(
                Text.assemble(
                    (f"{mark} {short} {agent.theme} ", "bold cyan"),
                    (f"[{agent.provider}] ", "dim"),
                    (status_str, status_style),
                    (wt_info, "dim"),
                )
            )
            prompt_input.placeholder = f"{mark} {short} {agent.theme} へ送信 (#00 で親へ, /help)"

    def _render_questions(self, questions: list[QuestionRecord]) -> None:
        box = self.query_one("#questions_box", Static)
        if not questions:
            box.remove_class("visible")
            box.update("")
            return
        txt = Text()
        txt.append("人間への確認待ち (/answer <回答> で返答):\n", style="bold magenta")
        for idx, q in enumerate(questions):
            txt.append(f"  {q.id} ", style="bold")
            txt.append(f"({q.theme}): ", style="cyan")
            txt.append(q.question)
            if idx < len(questions) - 1:
                txt.append("\n")
        box.update(txt)
        box.add_class("visible")

    @staticmethod
    def _build_header_widget(title: str, style: str, ts: str) -> Static:
        header = Text()
        header.append(f"● {title}", style=f"bold {style}")
        header.append(f"  {ts}", style="dim")
        return Static(header, classes="msg-header")

    async def _append_messages(self, messages: Iterable[MessageRecord], agents: list[AgentRecord]) -> None:
        log = self.query_one("#chat_log", VerticalScroll)
        agent_by_id = {a.id: a for a in agents}
        widgets: list[Widget] = []

        for msg in messages:
            if msg.id in self._seen_message_ids:
                continue
            self._seen_message_ids.add(msg.id)
            self._displayed_messages.append(msg)
            agent = agent_by_id.get(msg.agent_id)
            if agent is None:
                agent_label = msg.agent_id
            elif agent.role_kind == "root":
                agent_label = f"親Agent ({agent.provider})"
            elif agent.role_kind == "integrator":
                agent_label = f"助っ人:統合 ({agent.provider})"
            else:
                agent_label = f"{agent.theme} ({agent.provider})"
            ts = _fmt_time(msg.created_at)

            if msg.role == MessageRole.PROGRESS:
                line = Text()
                line.append(f"↳ {ts}  ", style="dim")
                line.append(msg.content, style="yellow")
                widgets.append(Static(line, classes="msg-progress"))
                continue

            if msg.role == MessageRole.HUMAN:
                widgets.append(self._build_header_widget("あなた", "bright_blue", ts))
                widgets.append(Static(Text(msg.content), classes="msg-body"))
                continue

            if msg.role == MessageRole.ANSWER:
                widgets.append(self._build_header_widget("あなたの回答", "magenta", ts))
                widgets.append(Static(Text(msg.content), classes="msg-body"))
                continue

            if msg.role == MessageRole.DELEGATION:
                widgets.append(self._build_header_widget("親からの依頼", "cyan", ts))
                widgets.append(TextualMarkdown(msg.content, classes="msg-markdown"))
                continue

            if msg.role == MessageRole.SYSTEM:
                widgets.append(self._build_header_widget("System / Error", "red", ts))
                widgets.append(Static(Text(msg.content, style="red"), classes="msg-body"))
                continue

            # Native Textual Markdown widget -> full mouse drag-selection + Ctrl+C!
            widgets.append(self._build_header_widget(agent_label, "green", ts))
            widgets.append(TextualMarkdown(msg.content, classes="msg-markdown"))

        if widgets:
            await log.mount_all(widgets)
            log.scroll_end(animate=False)

    async def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        if isinstance(event.node.data, str) and event.node.data != self._selected_agent_id:
            self._selected_agent_id = event.node.data
            if self._state is not None:
                self._render_header_and_target(self._state.agents)
            await self._load_selected_agent_messages(self._selected_agent_id)

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt_input":
            return
        if self._suppress_completion_once:
            self._suppress_completion_once = False
            return
        trigger = detect_completion_trigger(event.value, event.input.cursor_position)
        if trigger is None:
            self._completion.clear_trigger()
            self._render_completion_popup()
            return
        req_id = self._completion.begin_query(trigger)
        asyncio.create_task(self._fetch_completion(req_id, trigger))

    def _agent_switch_completions(self, query: str) -> list[CompletionItem]:
        """Offer `#00`, `#01`, `#02` view-switch items inside `#` autocomplete."""
        if self._state is None:
            return []
        q = query.lower().strip()
        items: list[CompletionItem] = []
        for agent in self._state.agents:
            short = self._short_id_by_agent.get(agent.id)
            if short is None:
                continue
            label = f"#{short}"
            mark = "◆" if agent.role_kind == "root" else ("✦" if agent.role_kind == "integrator" else "▸")
            desc = f"{mark} {agent.theme} [{agent.provider}] {agent.status.value} (遷移)"
            if not q or q in short or q in agent.theme.lower():
                items.append(CompletionItem(value=label, label=label, description=desc, kind="agent_switch"))
        items.sort(key=lambda i: i.value)
        return items

    def _agent_mention_completions(self, query: str) -> list[CompletionItem]:
        """Offer `@#00`, `@#02` direct-dispatch items inside `@#` autocomplete."""
        if self._state is None:
            return []
        q = query.lower().strip()
        items: list[CompletionItem] = []
        for agent in self._state.agents:
            short = self._short_id_by_agent.get(agent.id)
            if short is None or agent.role_kind == "integrator":
                continue
            label = f"@#{short}"
            mark = "◆" if agent.role_kind == "root" else ("▸" if agent.depth == 1 else "▹")
            desc = f"{mark} {agent.theme} [{agent.provider}] {agent.status.value} (親→子 確定委譲)"
            if not q or q in short or q in agent.theme.lower():
                items.append(CompletionItem(value=label, label=label, description=desc, kind="agent_mention"))
        items.sort(key=lambda i: i.value)
        return items

    async def _fetch_completion(self, req_id: int, trigger: CompletionTrigger) -> None:
        if trigger.kind == "agent_switch":
            items = self._agent_switch_completions(trigger.query)[:10]
        elif trigger.kind == "agent_mention":
            items = self._agent_mention_completions(trigger.query)[:10]
        else:
            try:
                resp = await self.client.complete(trigger.kind, trigger.query, limit=10)
            except Exception:
                return
            items = list(resp.items)
        if self._completion.apply_response(req_id, trigger, items):
            self._render_completion_popup()

    def _render_completion_popup(self) -> None:
        box = self.query_one("#completion_box", OptionList)
        if not self._completion.popup_open:
            box.clear_options()
            box.remove_class("visible")
            return
        box.clear_options()
        for item in self._completion.items:
            row = Text()
            style_map = {
                "file": "bold cyan",
                "command": "bold green",
                "agent_switch": "bold magenta",
                "agent_mention": "bold yellow",
            }
            row.append(item.label, style=style_map.get(item.kind, "bold"))
            if item.description:
                row.append(f"  {item.description}", style="dim")
            box.add_option(Option(row, id=item.value))
        box.highlighted = self._completion.selected_index - 1
        box.add_class("visible")

    def _apply_accepted_completion(self, index_1based: int | None = None) -> bool:
        chosen = self._completion.accept(index_1based=index_1based)
        self._render_completion_popup()
        if chosen is None:
            return False
        trigger, item = chosen
        prompt_input = self.query_one("#prompt_input", Input)
        new_text, new_cursor = apply_completion(prompt_input.value, trigger, item)
        self._suppress_completion_once = True
        prompt_input.value = new_text
        prompt_input.cursor_position = new_cursor
        prompt_input.focus()
        return True

    def on_key(self, event: events.Key) -> None:
        """Intercept Up/Down/Tab/Enter/Escape while completion popup is open (TLA+ NoAccidentalSubmitWhileOpen)."""
        if not self._completion.popup_open:
            return
        if event.key == "down":
            self._completion.move_down()
            self._render_completion_popup()
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            self._completion.move_up()
            self._render_completion_popup()
            event.prevent_default()
            event.stop()
        elif event.key in {"tab", "enter"}:
            self._apply_accepted_completion()
            event.prevent_default()
            event.stop()
        elif event.key == "escape":
            self._completion.dismiss()
            self._render_completion_popup()
            event.prevent_default()
            event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "completion_box":
            return
        self._apply_accepted_completion(index_1based=event.option_index + 1)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._completion.popup_open:
            self._apply_accepted_completion()
            return
        text = event.value.strip()
        event.input.value = ""
        self._completion.clear_trigger()
        self._render_completion_popup()
        if not text:
            return
        try:
            await self._handle_command_or_message(text)
        except Exception as exc:
            await self._append_notice(Text(f"Error: {exc}", style="bold red"))

    async def _handle_command_or_message(self, text: str) -> None:
        if text in {"/exit", "/quit"}:
            self.exit()
            return

        # Quick agent view switch by `#00`, `#01`, `#02` (or `#0`, `#1`, `#2`)
        if text.startswith("#") and text[1:].isdigit():
            resolved = self._resolve_agent_ref(text[1:])
            if resolved is None:
                await self._append_notice(Text(f"エージェント {text} が見つかりません。", style="yellow"))
                return
            self._selected_agent_id = resolved
            if self._state is not None:
                self._render_header_and_target(self._state.agents)
            await self._load_selected_agent_messages(resolved)
            return

        if text == "/copy" or text.startswith("/copy "):
            sub = text[len("/copy") :].strip().lower() or "last"
            self._copy_messages(mode=sub)
            return

        if text == "/help":
            help_md = "\n".join(
                [
                    "### コマンド・トリガー・ショートカット",
                    "- `/...` : スラッシュコマンド補完",
                    "- `@...` : ワークスペースのファイル補完（例: `@src/peanuts/tui.py`）",
                    "- `#00` / `#01` / `#02` : エージェント表示切替（`#` で候補補完）",
                    "- `@#02 <指示>` : 指定エージェントへ直接続きを送信（`@#` で候補補完）",
                    "- **マウスドラッグ選択** : 選択して離すだけで自動コピー",
                    "- `Ctrl+Y` / `/copy` : 表示中エージェントの **最新返答** を丸ごとコピー",
                    "- `/copy code` : 最新返答の **最後のコードブロック** だけをコピー",
                    "- `/copy all` : 表示中エージェントの **会話全体** をコピー",
                    "- `Ctrl+B` : 左サイドバーの表示/非表示切替",
                    "- `/answer <回答>` : 確認待ちの質問に回答（ID指定: `/answer <q_id> <回答>`）",
                    "- `/provider <claude|codex|mock>` : 選択中エージェントのCLIプロバイダ切替",
                    "- `/retry [02]` : `FAILED` 状態のエージェントを `IDLE` に復帰",
                    "- `/cancel <delegation_id>` : 待機中/確認中の委譲をキャンセル",
                    "- `/merge [02]` : 子エージェントのブランチを Integrator キューへ投入",
                    "- `/resume-merges` : dirty 待機中のマージキューを再開",
                    "- `/root` / `#00` : 表示/送信先を親Agent（窓口）に戻す",
                    "- `/exit` / `/quit` / `Ctrl+Q` : TUI を終了",
                ]
            )
            await self._append_notice(help_md, markdown=True)
            return

        if text.startswith("/merge"):
            parts = text.split(maxsplit=1)
            raw_target = parts[1].strip() if len(parts) == 2 else (self._selected_agent_id or "")
            target = self._resolve_agent_ref(raw_target) or raw_target
            if not target or target in {"root", "integrator"}:
                await self._append_notice(
                    Text(
                        "マージ対象の子エージェントを選択するか `/merge 02` のように指定してください。",
                        style="yellow",
                    )
                )
                return
            job = await self.client.trigger_merge(target)
            await self._append_notice(
                Text(f"Queued merge job {job.id}: {job.source_branch} -> {job.target_branch}", style="magenta")
            )
            return

        if text == "/resume-merges":
            await self.client.resume_merges()
            await self._append_notice(Text("Resumed waiting merge jobs.", style="magenta"))
            return

        if text == "/root" and self._state is not None:
            self._selected_agent_id = self._state.root_agent_id
            self._render_header_and_target(self._state.agents)
            await self._load_selected_agent_messages(self._selected_agent_id)
            return

        if text.startswith("/answer "):
            rest = text[len("/answer ") :].strip()
            questions = self._state.open_questions if self._state else []
            if not questions:
                await self._append_notice(Text("現在、確認待ちの質問はありません。", style="yellow"))
                return
            parts = rest.split(" ", 1)
            q_ids = {q.id for q in questions}
            if len(parts) == 2 and parts[0] in q_ids:
                qid, answer = parts[0], parts[1].strip()
            else:
                qid, answer = questions[0].id, rest
            await self.client.answer_question(qid, answer)
            return

        if text.startswith("/provider "):
            raw_prov = text[len("/provider ") :].strip()
            provider_map: dict[str, ProviderName] = {
                "claude": "claude",
                "codex": "codex",
                "mock": "mock",
            }
            prov = provider_map.get(raw_prov)
            if prov is None:
                await self._append_notice(Text("Provider must be one of: claude, codex, mock", style="red"))
                return
            target = self._selected_agent_id or (self._state.root_agent_id if self._state else "root")
            await self.client.set_provider(target, prov)
            await self._append_notice(Text(f"Updated {target} provider -> {prov}", style="green"))
            return

        if text.startswith("/retry"):
            parts = text.split(maxsplit=1)
            raw_target = parts[1].strip() if len(parts) == 2 else (self._selected_agent_id or "root")
            target = self._resolve_agent_ref(raw_target) or raw_target
            await self.client.retry_agent(target)
            await self._append_notice(Text(f"Retried {target}", style="green"))
            return

        if text.startswith("/cancel "):
            del_id = text[len("/cancel ") :].strip()
            await self.client.cancel_delegation(del_id)
            await self._append_notice(Text(f"Cancelled delegation {del_id}", style="yellow"))
            return

        # Leading `@#02 <prompt>` -> Human -> Root (logged) -> deterministic Root -> 02 delegation
        mention_match = _DIRECT_AGENT_MENTION_RE.match(text)
        if mention_match is not None:
            short_ref, body = mention_match.group(1), mention_match.group(2).strip()
            resolved = self._resolve_agent_ref(short_ref)
            if resolved is not None and body:
                target_agent = next((a for a in (self._state.agents if self._state else []) if a.id == resolved), None)
                if target_agent is not None and target_agent.role_kind == "integrator":
                    await self._append_notice(
                        Text(
                            "Integrator には親Agent経由（例: 親へマージ依頼）または /resume-merges を使ってください。",
                            style="yellow",
                        )
                    )
                    return
                if target_agent is not None and target_agent.role_kind == "root":
                    await self.client.send_message(body, target_agent_id=resolved)
                    return
                await self.client.direct_delegate(resolved, body, display_text=text)
                return

        target_id = self._selected_agent_id or (self._state.root_agent_id if self._state else None)
        await self.client.send_message(text, target_agent_id=target_id)
