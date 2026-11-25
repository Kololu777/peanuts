"""Unit tests for Claude CLI runner, Codex CLI runner, and structured action parsing."""

from __future__ import annotations

import json

import pytest

from peanuts.models import AskHumanAction, DelegateAction, IntegrateAction, ReplyAction
from peanuts.protocol import parse_agent_action
from peanuts.runners.base import RunnerError
from peanuts.runners.claude import build_claude_command, extract_claude_progress, parse_claude_json
from peanuts.runners.codex import build_codex_command, extract_codex_progress, parse_codex_jsonl


def test_claude_command_and_json_parsing() -> None:
    assert build_claude_command("claude", "hello", None) == [
        "claude",
        "-p",
        "hello",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    assert build_claude_command("claude", "follow up", "sess-123", permission_mode="auto") == [
        "claude",
        "-p",
        "follow up",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "auto",
        "--resume",
        "sess-123",
    ]

    # Single-object JSON format (backwards compatibility)
    stdout = json.dumps(
        {
            "session_id": "3823925d-28c3-428d-9cc0-ffc90921699e",
            "result": "Fixed the bug.",
            "is_error": False,
        }
    )
    out = parse_claude_json(stdout)
    assert out.text == "Fixed the bug."
    assert out.provider_session_id == "3823925d-28c3-428d-9cc0-ffc90921699e"

    # Streaming JSONL format (`--output-format stream-json --verbose`)
    stream_events = [
        {
            "type": "assistant",
            "session_id": "sess-stream-1",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
                ]
            },
        },
        {
            "type": "user",
            "session_id": "sess-stream-1",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "15 passed", "is_error": False},
                ]
            },
        },
        {
            "type": "result",
            "session_id": "sess-stream-1",
            "result": "All tests pass.",
            "is_error": False,
        },
    ]
    assert extract_claude_progress(stream_events[0]) == ["Tool Bash: pytest -q"]
    assert extract_claude_progress(stream_events[1]) == ["Tool done: 15 passed"]
    stream_out = parse_claude_json("\n".join(json.dumps(e) for e in stream_events))
    assert stream_out.provider_session_id == "sess-stream-1"
    assert stream_out.text == "All tests pass."

    err_stdout = json.dumps({"session_id": "s1", "result": "Auth expired", "is_error": True})
    with pytest.raises(RunnerError, match="Auth expired"):
        parse_claude_json(err_stdout)


def test_codex_command_and_jsonl_parsing() -> None:
    assert build_codex_command("codex", "fix tests", None) == ["codex", "exec", "--json", "fix tests"]
    assert build_codex_command(
        "codex",
        "continue",
        "0199a213-thread",
        sandbox="workspace-write",
        network_access=True,
    ) == [
        "codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "-c",
        "sandbox_workspace_write.network_access=true",
        "--json",
        "resume",
        "0199a213-thread",
        "continue",
    ]

    jsonl = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a53"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "thinking"}}),
            json.dumps({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "All tests pass."}}),
            json.dumps({"type": "turn.completed"}),
        ]
    )
    out = parse_codex_jsonl(jsonl)
    assert out.provider_session_id == "0199a213-81c0-7800-8aa1-bbab2a035a53"
    assert out.text == "All tests pass."
    assert extract_codex_progress(
        {"type": "item.started", "item": {"type": "command_execution", "command": "uv run pytest"}}
    ) == ["Running command: uv run pytest"]

    err_jsonl = json.dumps({"type": "turn.failed", "message": "Sandbox denied"})
    with pytest.raises(RunnerError, match="Sandbox denied"):
        parse_codex_jsonl(err_jsonl)


def test_parse_agent_action_variants() -> None:
    fenced_delegate = (
        "Routing this now.\n"
        "```peanuts-action\n"
        '{"action": "delegate", "message": "splitting", "tasks": [{"theme": "auth", "provider": "codex", "prompt": "check jwt"}]}\n'
        "```"
    )
    act = parse_agent_action(fenced_delegate)
    assert isinstance(act, DelegateAction)
    assert len(act.tasks) == 1
    assert act.tasks[0].theme == "auth"
    assert act.tasks[0].provider == "codex"

    raw_ask = '{"action": "ask_human", "question": "Proceed with drop table?"}'
    act2 = parse_agent_action(raw_ask)
    assert isinstance(act2, AskHumanAction)
    assert act2.question == "Proceed with drop table?"

    plain = "Just a normal markdown reply from Claude or Codex."
    act3 = parse_agent_action(plain)
    assert isinstance(act3, ReplyAction)
    assert act3.message == plain

    raw_integrate = (
        "```peanuts-action\n"
        '{"action": "integrate", "mode": "instruction", "instruction": "git add -A && git commit -m init", "message": "committing"}\n'
        "```"
    )
    act4 = parse_agent_action(raw_integrate)
    assert isinstance(act4, IntegrateAction)
    assert act4.mode == "instruction"
    assert "git commit" in act4.instruction
