"""Unit, invariant, API, and Textual pilot E2E tests for @file and /command completion."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import uvicorn
from fastapi.testclient import TestClient
from textual.widgets import Input

from peanuts.cli import _find_free_port, _wait_for_health
from peanuts.completion import (
    CompletionItem,
    CompletionStateMachine,
    CompletionTrigger,
    apply_completion,
    complete_commands,
    complete_files,
    detect_completion_trigger,
)
from peanuts.config import PeanutsConfig
from peanuts.server import create_app
from peanuts.tui import PeanutsTUI


def test_detect_trigger_and_apply_completion() -> None:
    # Command trigger at start
    t_cmd = detect_completion_trigger("/pro", 4)
    assert t_cmd is not None and t_cmd.kind == "command" and t_cmd.query == "pro"

    # File trigger at start or after whitespace
    t_file = detect_completion_trigger("please inspect @src/tui", 23)
    assert t_file is not None
    assert t_file.kind == "file"
    assert t_file.query == "src/tui"
    assert t_file.start == 15 and t_file.end == 23

    # Email address must NOT trigger @file completion
    assert detect_completion_trigger("contact user@example.com", 24) is None

    # `#01` agent view switch trigger & `@#01` direct agent mention trigger
    t_sw = detect_completion_trigger("#01", 3)
    assert t_sw is not None and t_sw.kind == "agent_switch" and t_sw.query == "01"
    t_mn = detect_completion_trigger("@#02", 4)
    assert t_mn is not None and t_mn.kind == "agent_mention" and t_mn.query == "02"

    sw_item = CompletionItem(value="#01", label="#01", kind="agent_switch")
    assert apply_completion("#0", t_sw, sw_item) == ("#01", 3)
    mn_item = CompletionItem(value="@#02", label="@#02", kind="agent_mention")
    assert apply_completion("@#0", t_mn, mn_item) == ("@#02 ", 5)

    # Applying completion preserves surrounding text cleanly
    item = CompletionItem(value="@src/peanuts/tui.py", label="@src/peanuts/tui.py", kind="file")
    new_text, new_cursor = apply_completion("please inspect @src/tui now", t_file, item)
    assert new_text == "please inspect @src/peanuts/tui.py now"
    assert new_cursor == len("please inspect @src/peanuts/tui.py")


def test_command_and_file_fuzzy_ranking(tmp_path: Path) -> None:
    (tmp_path / "src" / "peanuts").mkdir(parents=True)
    (tmp_path / "src" / "peanuts" / "tui.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "peanuts" / "state_machine.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("", encoding="utf-8")

    cmds = complete_commands("me")
    assert cmds[0].value == "/merge"

    files = complete_files(tmp_path, "tui")
    assert files[0].value == "@src/peanuts/tui.py"
    assert all(".venv" not in f.value for f in files)

    subseq = complete_files(tmp_path, "sm.py")
    assert any(f.value == "@src/peanuts/state_machine.py" for f in subseq)


def test_completion_state_machine_tla_invariants_and_stale_discard() -> None:
    sm = CompletionStateMachine()
    sm.check_invariants()

    t1 = detect_completion_trigger("@src", 4)
    t2 = detect_completion_trigger("@src/tui", 8)
    assert t1 is not None and t2 is not None

    req1 = sm.begin_query(t1)
    req2 = sm.begin_query(t2)

    three_items = [
        CompletionItem(value="@a.py", label="@a.py", kind="file"),
        CompletionItem(value="@b.py", label="@b.py", kind="file"),
        CompletionItem(value="@c.py", label="@c.py", kind="file"),
    ]
    one_item = [CompletionItem(value="@src/tui.py", label="@src/tui.py", kind="file")]

    # Req 2 arrives first -> opens popup
    assert sm.apply_response(req2, t2, three_items) is True
    sm.check_invariants()
    assert sm.popup_open and sm.selected_index == 1

    # Move down twice -> index=3
    sm.move_down()
    sm.move_down()
    assert sm.selected_index == 3
    sm.check_invariants()

    # Stale req 1 arrives late -> MUST be discarded!
    assert sm.apply_response(req1, t1, one_item) is False
    assert len(sm.items) == 3
    sm.check_invariants()

    # Candidate list shrinks from 3 -> 1 -> selected_index clamps safely back to 1
    req3 = sm.begin_query(t2)
    assert sm.apply_response(req3, t2, one_item) is True
    assert sm.selected_index == 1
    sm.check_invariants()

    # In-flight req 4 invalidated by Escape -> late response MUST NOT re-open popup!
    req4 = sm.begin_query(t2)
    sm.dismiss()
    sm.check_invariants()
    assert sm.apply_response(req4, t2, one_item) is False
    assert not sm.popup_open
    sm.check_invariants()


def test_complete_api_endpoint(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    cfg = PeanutsConfig(db_path=tmp_path / "api.db", cwd=tmp_path, root_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r_cmd = client.get("/api/v1/complete", params={"kind": "command", "q": "co"})
        assert r_cmd.status_code == 200
        assert any(i["value"] == "/copy" for i in r_cmd.json()["items"])

        r_file = client.get("/api/v1/complete", params={"kind": "file", "q": "hel"})
        assert r_file.status_code == 200
        assert r_file.json()["items"][0]["value"] == "@hello.py"


async def test_tui_autocomplete_keyboard_flow_prevents_accidental_submit(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("", encoding="utf-8")
    (tmp_path / "alpha_beta.py").write_text("", encoding="utf-8")
    port = _find_free_port("127.0.0.1")
    cfg = PeanutsConfig(
        db_path=tmp_path / "tui_comp.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        host="127.0.0.1",
        port=port,
    )
    fastapi_app = create_app(cfg)
    server = uvicorn.Server(uvicorn.Config(fastapi_app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(url)
        tui = PeanutsTUI(server_url=url)
        async with tui.run_test() as pilot:
            await pilot.pause(0.1)
            prompt = tui.query_one("#prompt_input", Input)
            prompt.focus()

            # Type `@alp` -> popup opens with alpha.py and alpha_beta.py
            await pilot.press("@", "a", "l", "p")
            for _ in range(20):
                await pilot.pause(0.05)
                if tui._completion.popup_open:
                    break
            assert tui._completion.popup_open
            assert len(tui._completion.items) == 2

            # Down selects 2nd item (`@alpha_beta.py`), Enter accepts WITHOUT submitting prompt!
            await pilot.press("down")
            assert tui._completion.selected_index == 2
            await pilot.press("enter")
            await pilot.pause(0.05)

            assert not tui._completion.popup_open
            assert prompt.value == "@alpha_beta.py "
            # Verify NO human message was accidentally submitted to the server!
            msgs = await tui.client.get_agent_messages("root")
            assert msgs == []

            # Now test `/he` + Tab accepts `/help` (and does not steal focus!)
            prompt.value = ""
            await pilot.pause(0.05)
            await pilot.press("/", "h", "e")
            for _ in range(20):
                await pilot.pause(0.05)
                if tui._completion.popup_open:
                    break
            assert tui._completion.popup_open
            await pilot.press("tab")
            await pilot.pause(0.05)
            assert not tui._completion.popup_open
            assert prompt.value == "/help"

            # Now test `/co` + Escape dismisses popup cleanly
            prompt.value = ""
            await pilot.pause(0.05)
            await pilot.press("/", "c", "o")
            for _ in range(20):
                await pilot.pause(0.05)
                if tui._completion.popup_open:
                    break
            assert tui._completion.popup_open
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert not tui._completion.popup_open

            # TLA+ MoveUp wrap + ReceiveFreshEmpty (0 matches closes popup) + SubmitPrompt after Escape
            prompt.value = ""
            await pilot.pause(0.05)
            await pilot.press("@", "a", "l", "p")
            for _ in range(20):
                await pilot.pause(0.05)
                if tui._completion.popup_open:
                    break
            assert tui._completion.popup_open and tui._completion.selected_index == 1
            await pilot.press("up")
            assert tui._completion.selected_index == 2  # wrapped to last candidate

            # Typing non-matching suffix yields 0 candidates -> ReceiveFreshEmpty closes popup cleanly
            await pilot.press("z", "z", "z")
            for _ in range(20):
                await pilot.pause(0.05)
                if not tui._completion.popup_open:
                    break
            assert not tui._completion.popup_open
            tui._completion.check_invariants()

            # Escape dismissal lets Enter submit normally (TLA+ SubmitPrompt)
            prompt.value = ""
            await pilot.pause(0.05)
            await pilot.press("/", "c", "o")
            for _ in range(20):
                await pilot.pause(0.05)
                if tui._completion.popup_open:
                    break
            assert tui._completion.popup_open
            await pilot.press("escape")
            await pilot.pause(0.05)
            assert not tui._completion.popup_open
            prompt.value = "direct: hello after escape"
            await pilot.pause(0.05)
            await pilot.press("enter")
            for _ in range(30):
                await pilot.pause(0.05)
                msgs_after = await tui.client.get_agent_messages("root")
                if any("hello after escape" in m.content for m in msgs_after):
                    break
            msgs_after = await tui.client.get_agent_messages("root")
            assert any("hello after escape" in m.content for m in msgs_after)
    finally:
        server.should_exit = True
        await asyncio.to_thread(thread.join, 3.0)


def test_completion_state_machine_all_tla_actions_and_edge_cases() -> None:
    """Covers ReceiveFreshEmpty, ClearTrigger, MoveUp/MoveDown wrap, index preservation, AcceptCompletion, and AGENT mode."""
    sm = CompletionStateMachine()
    sm.check_invariants()

    # No-op actions when closed
    sm.move_down()
    sm.move_up()
    assert sm.accept() is None
    sm.check_invariants()

    t_file = detect_completion_trigger("@src", 4)
    t_cmd = detect_completion_trigger("/co", 3)
    t_agent = detect_completion_trigger("#01", 3)
    assert t_file is not None and t_cmd is not None and t_agent is not None

    items3 = [
        CompletionItem(value="@a.py", label="@a.py", kind="file"),
        CompletionItem(value="@b.py", label="@b.py", kind="file"),
        CompletionItem(value="@c.py", label="@c.py", kind="file"),
    ]
    items2 = items3[:2]

    # Open with 3 items -> MoveDown wrap (1->2->3->1) and MoveUp wrap (1->3->2)
    r1 = sm.begin_query(t_file)
    assert sm.apply_response(r1, t_file, items3) is True
    assert sm.selected_index == 1
    sm.move_down()
    sm.move_down()
    assert sm.selected_index == 3
    sm.move_down()
    assert sm.selected_index == 1  # wrap down
    sm.move_up()
    assert sm.selected_index == 3  # wrap up
    sm.move_up()
    assert sm.selected_index == 2
    sm.check_invariants()

    # ReceiveFreshResults preserves selected_index when 1 <= prev_idx <= len(new_items)
    r2 = sm.begin_query(t_file)
    assert sm.apply_response(r2, t_file, items2) is True
    assert sm.selected_index == 2
    sm.check_invariants()

    # ReceiveFreshEmpty closes open popup cleanly (prevents empty ghost popup)
    r3 = sm.begin_query(t_file)
    assert sm.apply_response(r3, t_file, []) is True
    assert not sm.popup_open and sm.mode == "NONE" and sm.selected_index == 0
    sm.check_invariants()

    # ClearTrigger invalidates in-flight queries so late response is discarded
    r4 = sm.begin_query(t_cmd)
    sm.clear_trigger()
    sm.check_invariants()
    assert sm.apply_response(r4, t_cmd, [CompletionItem(value="/copy", label="/copy", kind="command")]) is False
    assert not sm.popup_open
    sm.check_invariants()

    # AcceptCompletion returns chosen item, closes popup, and invalidates in-flight queries
    r5 = sm.begin_query(t_agent)
    agent_items = [CompletionItem(value="#01", label="#01", kind="agent_switch")]
    assert sm.apply_response(r5, t_agent, agent_items) is True
    assert sm.mode == "AGENT"
    assert sm.accept(index_1based=99) is None  # out-of-bounds index rejected
    r6_inflight = sm.begin_query(t_agent)
    chosen = sm.accept()
    assert chosen == (t_agent, agent_items[0])
    assert not sm.popup_open
    # Late response for r6_inflight MUST be discarded because AcceptCompletion invalidated up to latest_req_id
    assert sm.apply_response(r6_inflight, t_agent, agent_items) is False
    assert not sm.popup_open
    sm.check_invariants()


def test_completion_state_machine_stateful_tla_exploration() -> None:
    """Randomized deterministic exploration of all PeanutsCompletion.tla actions with per-step invariant checks."""
    import random

    rng = random.Random(20261127)
    sm = CompletionStateMachine()
    in_flight: dict[int, CompletionTrigger] = {}
    triggers = [
        detect_completion_trigger("/he", 3),
        detect_completion_trigger("@src", 4),
        detect_completion_trigger("#01", 3),
        detect_completion_trigger("@#02", 4),
    ]
    valid_triggers = [t for t in triggers if t is not None]
    action_counts: dict[str, int] = {}

    for step in range(400):
        candidates = ["type_query"]
        if sm.popup_open or in_flight:
            candidates.append("clear_trigger")
        if in_flight:
            candidates.extend(["deliver_latest_results", "deliver_results", "deliver_empty"])
        if sm.popup_open:
            candidates.extend(["move_down", "move_up", "accept", "dismiss"])

        choice = rng.choice(candidates)
        action_counts[choice] = action_counts.get(choice, 0) + 1

        if choice == "type_query":
            trig = rng.choice(valid_triggers)
            rid = sm.begin_query(trig)
            in_flight[rid] = trig
        elif choice == "clear_trigger":
            sm.clear_trigger()
        elif choice == "deliver_latest_results":
            rid = max(in_flight)
            trig = in_flight.pop(rid)
            k = rng.randint(1, 4)
            items = [
                CompletionItem(value=f"v{i}-{step}", label=f"v{i}-{step}", kind=trig.kind) for i in range(1, k + 1)
            ]
            applied = sm.apply_response(rid, trig, items)
            expected_fresh = rid == sm.latest_req_id and rid > sm.invalidated_up_to
            assert applied == expected_fresh
        elif choice == "deliver_results":
            rid = rng.choice(list(in_flight))
            trig = in_flight.pop(rid)
            k = rng.randint(1, 4)
            items = [
                CompletionItem(value=f"v{i}-{step}", label=f"v{i}-{step}", kind=trig.kind) for i in range(1, k + 1)
            ]
            applied = sm.apply_response(rid, trig, items)
            expected_fresh = rid == sm.latest_req_id and rid > sm.invalidated_up_to
            assert applied == expected_fresh
        elif choice == "deliver_empty":
            rid = rng.choice(list(in_flight))
            trig = in_flight.pop(rid)
            applied = sm.apply_response(rid, trig, [])
            expected_fresh = rid == sm.latest_req_id and rid > sm.invalidated_up_to
            assert applied == expected_fresh
        elif choice == "move_down":
            sm.move_down()
        elif choice == "move_up":
            sm.move_up()
        elif choice == "accept":
            assert sm.accept() is not None
        elif choice == "dismiss":
            sm.dismiss()

        sm.check_invariants()

    # Drain all remaining in-flight responses (AllInFlightEventuallyDrain)
    for rid in list(in_flight):
        trig = in_flight.pop(rid)
        sm.apply_response(rid, trig, [])
        sm.check_invariants()

    for expected in (
        "type_query",
        "clear_trigger",
        "deliver_results",
        "deliver_empty",
        "move_down",
        "move_up",
        "accept",
        "dismiss",
    ):
        assert action_counts.get(expected, 0) > 0, f"Completion action {expected} was never exercised"
