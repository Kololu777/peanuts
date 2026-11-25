"""Textual TUI client for Peanuts with clean Rich Markdown rendering.

Speaks strictly to the FastAPI REST + WebSocket API via PeanutsClient,
so a Web + React UI can replace it without touching any orchestrator logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime

from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static, Tree
from textual.widgets.tree import TreeNode

from peanuts.client import PeanutsClient
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


def _fmt_time(dt: datetime) -> str:
    return dt.astimezone().strftime("%H:%M:%S")


class PeanutsTUI(App[None]):
    """Clean parent-child agent mediator TUI with Markdown panels and per-agent log view."""

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
        padding: 1 1 0 1;
        background: $surface;
    }
    #questions_box {
        height: auto;
        max-height: 8;
        border: round magenta;
        padding: 0 1;
        margin-top: 1;
        display: none;
    }
    #questions_box.visible {
        display: block;
    }
    #target_bar {
        height: auto;
        padding: 0 1;
        margin-top: 1;
    }
    #prompt_input {
        dock: bottom;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
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
        self._last_tree_fingerprint: tuple[object, ...] | None = None
        self._ws_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static(
                    Text.assemble(
                        ("エージェント一覧\n", "bold"),
                        ("クリックで中身・進捗を表示", "dim"),
                    ),
                    id="sidebar_title",
                )
                yield Tree("root", id="agent_tree")
                yield Static(Text("稼働中の子タスク: なし", style="dim"), id="active_tasks_box")
            with Vertical(id="main"):
                yield Static("", id="view_header")
                yield RichLog(id="chat_log", markup=False, wrap=True, auto_scroll=True)
                yield Static("", id="questions_box")
                yield Static("", id="target_bar")
                yield Input(
                    placeholder="メッセージを入力（/help でコマンド一覧・/exit で終了）",
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
                            self._append_messages([msg], agents)
                    else:
                        await self._refresh_state(reload_messages=False)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self.query_one("#chat_log", RichLog).write(Text(f"WS reconnect in 1s ({exc})", style="red"))
                await asyncio.sleep(1.0)

    async def _refresh_state(self, *, reload_messages: bool) -> None:
        state = await self.client.get_state(message_limit=50)
        self._state = state
        if self._selected_agent_id is None:
            self._selected_agent_id = state.root_agent_id
        self._render_tree_if_changed(state.agents, state.delegations)
        self._render_active_tasks(state.delegations, state.merge_jobs)
        self._render_questions(state.open_questions)
        self._render_header_and_target(state.agents)
        if reload_messages and self._selected_agent_id is not None:
            await self._load_selected_agent_messages(self._selected_agent_id)

    async def _load_selected_agent_messages(self, agent_id: str) -> None:
        log = self.query_one("#chat_log", RichLog)
        agents = self._state.agents if self._state else []
        messages = await self.client.get_agent_messages(agent_id, limit=200)

        log.clear()
        self._seen_message_ids.clear()
        if not messages:
            log.write(
                Text(
                    "まだメッセージはありません。下の入力欄から依頼を送ってみてください。",
                    style="dim",
                )
            )
        else:
            self._append_messages(messages, agents)

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
            for child in by_parent.get(parent_id, []):
                child_node = node.add(self._format_agent_label(child), data=child.id, expand=True)
                add_children(child_node, child.id)

        add_children(tree.root, root_agent.id)
        tree.root.expand()

    @staticmethod
    def _format_agent_label(agent: AgentRecord, *, active_children: int = 0) -> Text:
        badge_text, badge_style = _STATUS_BADGE.get(agent.status, (agent.status.value, "white"))
        label = Text()
        if agent.role_kind == "root":
            label.append("親(窓口) ", style="bold")
        elif agent.role_kind == "integrator":
            label.append("助っ人:統合 ", style="bold magenta")
        else:
            prefix = "子" if agent.depth == 1 else "孫"
            label.append(f"{prefix}:{agent.theme} ", style="bold cyan" if agent.depth == 1 else "cyan")
        label.append(f"[{agent.provider}] ", style="dim")
        label.append(badge_text, style=badge_style)
        if agent.role_kind == "root" and active_children > 0:
            label.append(f" (稼働:{active_children})", style="cyan")
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
        bar = self.query_one("#target_bar", Static)
        target_id = self._selected_agent_id or "root"
        agent = next((a for a in agents if a.id == target_id), None)
        base_branch = self._state.base_branch if self._state else None
        branch_part = f"  branch:{base_branch}" if base_branch else ""

        if agent is None or agent.role_kind == "root":
            prov = agent.provider if agent else "-"
            status_str, status_style = (
                _STATUS_BADGE.get(agent.status, (agent.status.value, "white")) if agent else ("-", "dim")
            )
            header.update(
                Text.assemble(
                    ("親Agent（窓口） ", "bold green"),
                    (f"[{prov}] ", "dim"),
                    (status_str, status_style),
                    (f"{branch_part}   API: {self.server_url}", "dim"),
                )
            )
            bar.update(
                Text.assemble(
                    ("送信先: ", "dim"),
                    ("親Agent", "bold green"),
                    (" — テーマごとに独立worktreeの子へ振り分けます", "dim"),
                )
            )
        elif agent.role_kind == "integrator":
            status_str, status_style = _STATUS_BADGE.get(agent.status, (agent.status.value, "white"))
            header.update(
                Text.assemble(
                    ("専用マージ助っ人 (Integrator) ", "bold magenta"),
                    (f"[{agent.provider}] ", "dim"),
                    (status_str, status_style),
                    (f"  統合先:{agent.branch_name or '-'}", "dim"),
                )
            )
            bar.update(
                Text.assemble(
                    ("閲覧中: ", "dim"),
                    ("Integrator", "bold magenta"),
                    (" — マージ＆コンフリクト解消ログ（/resume-merges で待機再開・/root で親へ）", "dim"),
                )
            )
        else:
            status_str, status_style = _STATUS_BADGE.get(agent.status, (agent.status.value, "white"))
            wt_info = f"  branch:{agent.branch_name}" if agent.branch_name else f"  id={agent.id}"
            header.update(
                Text.assemble(
                    (f"子Agent: {agent.theme} ", "bold cyan"),
                    (f"[{agent.provider}] ", "dim"),
                    (status_str, status_style),
                    (wt_info, "dim"),
                )
            )
            bar.update(
                Text.assemble(
                    ("送信先: ", "dim"),
                    (f"{agent.theme}", "bold cyan"),
                    (" — 子のログ閲覧＆直接会話中（/merge で統合キュー投入・/root で親へ）", "dim"),
                )
            )

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

    def _append_messages(self, messages: Iterable[MessageRecord], agents: list[AgentRecord]) -> None:
        log = self.query_one("#chat_log", RichLog)
        agent_by_id = {a.id: a for a in agents}
        for msg in messages:
            if msg.id in self._seen_message_ids:
                continue
            self._seen_message_ids.add(msg.id)
            agent = agent_by_id.get(msg.agent_id)
            if agent is None:
                agent_label = msg.agent_id
            elif agent.depth == 0:
                agent_label = f"親Agent ({agent.provider})"
            else:
                agent_label = f"{agent.theme} ({agent.provider})"
            ts = _fmt_time(msg.created_at)

            if msg.role == MessageRole.PROGRESS:
                line = Text()
                line.append(f"  ↳ {ts}  ", style="dim")
                line.append(msg.content, style="yellow")
                log.write(line)
                continue

            if msg.role == MessageRole.HUMAN:
                log.write(
                    Panel(
                        Text(msg.content),
                        title=f"あなた · {ts}",
                        title_align="left",
                        border_style="bright_blue",
                        padding=(0, 1),
                    )
                )
                continue

            if msg.role == MessageRole.ANSWER:
                log.write(
                    Panel(
                        Text(msg.content),
                        title=f"あなたの回答 · {ts}",
                        title_align="left",
                        border_style="magenta",
                        padding=(0, 1),
                    )
                )
                continue

            if msg.role == MessageRole.DELEGATION:
                log.write(
                    Panel(
                        Markdown(msg.content),
                        title=f"親からの依頼 · {ts}",
                        title_align="left",
                        border_style="cyan",
                        padding=(0, 1),
                    )
                )
                continue

            if msg.role == MessageRole.SYSTEM:
                log.write(
                    Panel(
                        Text(msg.content),
                        title=f"System / Error · {ts}",
                        title_align="left",
                        border_style="red",
                        padding=(0, 1),
                    )
                )
                continue

            # Default: MessageRole.AGENT -> full Rich Markdown rendering (bold, tables, code blocks, lists)
            log.write(
                Panel(
                    Markdown(msg.content),
                    title=f"{agent_label} · {ts}",
                    title_align="left",
                    border_style="green",
                    padding=(0, 1),
                )
            )

    async def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        if isinstance(event.node.data, str) and event.node.data != self._selected_agent_id:
            self._selected_agent_id = event.node.data
            if self._state is not None:
                self._render_header_and_target(self._state.agents)
            await self._load_selected_agent_messages(self._selected_agent_id)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        log = self.query_one("#chat_log", RichLog)
        try:
            await self._handle_command_or_message(text)
        except Exception as exc:
            log.write(Panel(Text(str(exc)), title="Error", title_align="left", border_style="red"))

    async def _handle_command_or_message(self, text: str) -> None:
        log = self.query_one("#chat_log", RichLog)
        if text in {"/exit", "/quit"}:
            self.exit()
            return

        if text == "/help":
            help_md = "\n".join(
                [
                    "### コマンド一覧",
                    "- `<メッセージ>` : 選択中のエージェントへ送信（デフォルトは親Agent）",
                    "- `/answer <回答>` : 確認待ちの質問に回答（ID指定: `/answer <q_id> <回答>`）",
                    "- `/provider <claude|codex|mock>` : 選択中エージェントのCLIプロバイダ切替",
                    "- `/retry [agent_id]` : `FAILED` 状態のエージェントを `IDLE` に復帰",
                    "- `/cancel <delegation_id>` : 待機中/確認中の委譲をキャンセル",
                    "- `/merge [agent_id]` : 子エージェントのブランチを Integrator キューへ投入",
                    "- `/resume-merges` : dirty 待機中のマージキューを再開",
                    "- `/root` : 表示/送信先を親Agent（窓口）に戻す",
                    "- `/exit` / `/quit` / `Ctrl+Q` : TUI を終了",
                ]
            )
            log.write(Rule(style="dim"))
            log.write(Panel(Markdown(help_md), title="Help", title_align="left", border_style="blue", padding=(0, 1)))
            return

        if text.startswith("/merge"):
            parts = text.split(maxsplit=1)
            target = parts[1].strip() if len(parts) == 2 else (self._selected_agent_id or "")
            if not target or target in {"root", "integrator"}:
                log.write(Text("マージ対象の子エージェントを選択するか `/merge <agent_id>` を指定してください。", style="yellow"))
                return
            job = await self.client.trigger_merge(target)
            log.write(Text(f"Queued merge job {job.id}: {job.source_branch} -> {job.target_branch}", style="magenta"))
            return

        if text == "/resume-merges":
            await self.client.resume_merges()
            log.write(Text("Resumed waiting merge jobs.", style="magenta"))
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
                log.write(Text("現在、確認待ちの質問はありません。", style="yellow"))
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
                log.write(Text("Provider must be one of: claude, codex, mock", style="red"))
                return
            target = self._selected_agent_id or (self._state.root_agent_id if self._state else "root")
            await self.client.set_provider(target, prov)
            log.write(Text(f"Updated {target} provider -> {prov}", style="green"))
            return

        if text.startswith("/retry"):
            parts = text.split(maxsplit=1)
            target = parts[1].strip() if len(parts) == 2 else (self._selected_agent_id or "root")
            await self.client.retry_agent(target)
            log.write(Text(f"Retried {target}", style="green"))
            return

        if text.startswith("/cancel "):
            del_id = text[len("/cancel ") :].strip()
            await self.client.cancel_delegation(del_id)
            log.write(Text(f"Cancelled delegation {del_id}", style="yellow"))
            return

        target_id = self._selected_agent_id or (self._state.root_agent_id if self._state else None)
        await self.client.send_message(text, target_agent_id=target_id)
