"""Tests for structured `ask_human` options: parsing, persistence, answer resolution, and TUI selection."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from textual.widgets import Input, OptionList

from peanuts.client import PeanutsClient
from peanuts.config import PeanutsConfig
from peanuts.db import PeanutsRepository
from peanuts.models import (
    AgentRecord,
    AgentStatus,
    AskHumanAction,
    AskHumanOption,
    EventRecord,
    MessageRecord,
    MessageRole,
    ProviderName,
    QuestionRecord,
    SystemStateResponse,
)
from peanuts.orchestrator import Orchestrator
from peanuts.protocol import (
    build_agent_prompt,
    format_question_options,
    parse_agent_action,
    resolve_question_answer,
)
from peanuts.runners.base import ProgressCallback, TurnInput, TurnOutput
from peanuts.tui import PeanutsTUI

_OPTIONS_PAYLOAD: dict[str, Any] = {
    "action": "ask_human",
    "question": "どちらの方式で進めますか?",
    "options": [
        {"label": "A: 既存を拡張", "description": "変更が小さい", "recommended": True},
        {"label": "B: 新規に作る", "description": "自由度が高い"},
    ],
}


def _fenced(payload: dict[str, Any]) -> str:
    return f"```peanuts-action\n{json.dumps(payload, ensure_ascii=False)}\n```"


def test_parse_ask_human_with_options_and_backward_compat() -> None:
    act = parse_agent_action(_fenced(_OPTIONS_PAYLOAD))
    assert isinstance(act, AskHumanAction)
    assert [o.label for o in act.options] == ["A: 既存を拡張", "B: 新規に作る"]
    assert [o.recommended for o in act.options] == [True, False]
    assert act.options[0].description == "変更が小さい"

    legacy = parse_agent_action('{"action": "ask_human", "question": "Proceed?"}')
    assert isinstance(legacy, AskHumanAction)
    assert legacy.options == []


def test_parse_ask_human_tolerates_sloppy_options() -> None:
    sloppy = {
        "action": "ask_human",
        "question": "Pick one",
        "options": ["plain string", {"label": "  ok  ", "recommended": "yes", "description": None}, {"nolabel": 1}, 42],
    }
    act = parse_agent_action(_fenced(sloppy))
    assert isinstance(act, AskHumanAction)
    assert [o.label for o in act.options] == ["plain string", "ok"]
    assert all(not o.recommended for o in act.options)

    not_a_list = parse_agent_action(_fenced({"action": "ask_human", "question": "Pick", "options": "A or B"}))
    assert isinstance(not_a_list, AskHumanAction)
    assert not_a_list.options == []


def test_format_and_resolve_question_options() -> None:
    options = [AskHumanOption.model_validate(o) for o in _OPTIONS_PAYLOAD["options"]]
    assert format_question_options(options) == (
        "1. **A: 既存を拡張** (おすすめ) — 変更が小さい\n2. **B: 新規に作る** — 自由度が高い"
    )
    assert format_question_options([]) == ""

    q = QuestionRecord(id="q1", agent_id="root", theme="root", question="?", options=options)
    assert resolve_question_answer(q, "2") == "B: 新規に作る"
    assert resolve_question_answer(q, " 1 ") == "A: 既存を拡張"
    assert resolve_question_answer(q, "3") == "3"
    assert resolve_question_answer(q, "どちらでもない") == "どちらでもない"
    plain_q = QuestionRecord(id="q2", agent_id="root", theme="root", question="?")
    assert resolve_question_answer(plain_q, "1") == "1"


def test_prompt_documents_options_only_when_questions_remain() -> None:
    root = AgentRecord(id="root", theme="root", provider="mock", depth=0, role_kind="root")
    with_budget = build_agent_prompt(
        agent=root,
        user_input="hi",
        existing_children=[],
        available_providers=["mock"],
        max_depth=2,
        remaining_questions=2,
    )
    assert '"options"' in with_budget and '"recommended": true' in with_budget
    without_budget = build_agent_prompt(
        agent=root,
        user_input="hi",
        existing_children=[],
        available_providers=["mock"],
        max_depth=2,
        remaining_questions=0,
    )
    assert "ask_human" not in without_budget


class _AskWithOptionsRunner:
    """Asks one structured-choice question, then echoes back whatever input it is resumed with."""

    provider: ProviderName = "mock"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        self.inputs.append(turn.prompt)
        if len(self.inputs) == 1:
            return TurnOutput(text=_fenced(_OPTIONS_PAYLOAD), provider_session_id="sess-opt")
        return TurnOutput(text="done", provider_session_id="sess-opt")


async def test_orchestrator_persists_options_and_roundtrips_answer(tmp_path: Path) -> None:
    cfg = PeanutsConfig(db_path=tmp_path / "peanuts.db", cwd=tmp_path, root_provider="mock", enable_worktrees=False)
    runner = _AskWithOptionsRunner()
    orch = Orchestrator(cfg, runners={"mock": runner})
    try:
        await orch.send_human_message("start")
        await orch.wait_until_quiescent()

        (q,) = orch.repo.list_questions(only_open=True)
        assert [o.label for o in q.options] == ["A: 既存を拡張", "B: 新規に作る"]
        assert q.options[0].recommended is True
        assert orch.get_state().open_questions[0].options == q.options

        question_msgs = [m for m in orch.repo.list_messages("root") if m.metadata.get("question_id") == q.id]
        assert "1. **A: 既存を拡張** (おすすめ) — 変更が小さい" in question_msgs[0].content

        await orch.answer_question(q.id, resolve_question_answer(q, "2"))
        await orch.wait_until_quiescent()

        assert orch.repo.list_questions(only_open=True) == []
        assert orch.repo.get_root_agent().status == AgentStatus.IDLE
        assert "B: 新規に作る" in runner.inputs[-1]
    finally:
        await orch.shutdown()


def test_existing_database_gains_options_column(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    PeanutsRepository(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE questions DROP COLUMN options_json")
    conn.execute(
        "INSERT INTO agents (id, theme, provider, depth, status, created_at, updated_at) "
        "VALUES ('root', 'root', 'mock', 0, 'WAITING_HUMAN', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO questions (id, agent_id, theme, question, status, created_at) "
        "VALUES ('q_old', 'root', 'root', 'legacy?', 'OPEN', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    repo = PeanutsRepository(db_path)
    try:
        (q,) = repo.list_questions(only_open=True)
        assert q.id == "q_old" and q.options == []
    finally:
        repo.close()


class _FakeClient:
    """Minimal in-memory stand-in for PeanutsClient used to drive the TUI without a server."""

    def __init__(self, questions: list[QuestionRecord]) -> None:
        self.questions = questions
        self.answers: list[tuple[str, str]] = []
        self.sent: list[str] = []

    async def get_state(self, message_limit: int = 100) -> SystemStateResponse:
        root = AgentRecord(
            id="root", theme="root", provider="mock", depth=0, role_kind="root", status=AgentStatus.WAITING_HUMAN
        )
        return SystemStateResponse(
            root_agent_id="root",
            latest_seq=0,
            agents=[root],
            delegations=[],
            open_questions=list(self.questions),
            recent_messages=[],
        )

    async def get_agent_messages(self, agent_id: str, limit: int = 100) -> list[MessageRecord]:
        return [MessageRecord(id="m1", agent_id="root", role=MessageRole.AGENT, content="hello")]

    async def stream_events(self, after_seq: int = 0) -> AsyncIterator[EventRecord]:
        await asyncio.Event().wait()
        yield cast(EventRecord, None)  # pragma: no cover - never reached

    async def answer_question(self, question_id: str, answer: str) -> None:
        self.answers.append((question_id, answer))
        self.questions = [q for q in self.questions if q.id != question_id]

    async def send_message(self, content: str, *, target_agent_id: str | None = None) -> None:
        self.sent.append(content)

    async def aclose(self) -> None:
        return None


def _choice_question(qid: str = "q_choice") -> QuestionRecord:
    return QuestionRecord(
        id=qid,
        agent_id="root",
        theme="root",
        question=_OPTIONS_PAYLOAD["question"],
        options=[AskHumanOption.model_validate(o) for o in _OPTIONS_PAYLOAD["options"]],
    )


async def _make_app(fake: _FakeClient) -> PeanutsTUI:
    app = PeanutsTUI("http://127.0.0.1:1")
    await app.client.aclose()
    app.client = cast(PeanutsClient, fake)
    return app


async def test_tui_arrow_keys_select_option_and_enter_answers() -> None:
    fake = _FakeClient([_choice_question()])
    app = await _make_app(fake)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        box = app.query_one("#question_options", OptionList)
        assert box.has_class("visible")
        assert box.option_count == 2
        assert "★おすすめ" in str(box.get_option_at_index(0).prompt)
        assert box.highlighted is None

        # Enter on an empty prompt without an explicit selection must not answer anything.
        await pilot.press("enter")
        assert fake.answers == []

        # First arrow press lands on the recommended option; the next moves on; Enter confirms.
        await pilot.press("down")
        assert box.highlighted == 0
        await pilot.press("down")
        assert box.highlighted == 1
        await pilot.press("enter")
        await pilot.pause()
        assert fake.answers == [("q_choice", "B: 新規に作る")]

        await app.action_refresh_state()
        await pilot.pause()
        assert not box.has_class("visible")
        assert box.option_count == 0


async def test_tui_answer_command_accepts_number_and_free_text() -> None:
    fake = _FakeClient([_choice_question("q_a"), _choice_question("q_b")])
    app = await _make_app(fake)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        prompt_input = app.query_one("#prompt_input", Input)

        # Typing text disables arrow selection so normal editing is never hijacked.
        await pilot.press("x", "down")
        assert app.query_one("#question_options", OptionList).highlighted is None
        prompt_input.value = ""

        prompt_input.value = "/answer 1"
        await pilot.press("enter")
        await pilot.pause()
        assert fake.answers == [("q_a", "A: 既存を拡張")]

        await app.action_refresh_state()
        prompt_input.value = "/answer q_b どちらも違うので C で"
        await pilot.press("enter")
        await pilot.pause()
        assert fake.answers[-1] == ("q_b", "どちらも違うので C で")


async def test_tui_plain_question_has_no_options_list() -> None:
    plain = QuestionRecord(id="q_plain", agent_id="root", theme="root", question="Proceed?")
    fake = _FakeClient([plain])
    app = await _make_app(fake)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not app.query_one("#question_options", OptionList).has_class("visible")
        await pilot.press("down", "enter")
        assert fake.answers == []
        app.query_one("#prompt_input", Input).value = "/answer 1"
        await pilot.press("enter")
        await pilot.pause()
        assert fake.answers == [("q_plain", "1")]
