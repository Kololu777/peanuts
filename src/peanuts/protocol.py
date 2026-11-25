"""Prompt construction and structured action parsing for CLI agents (Claude / Codex / Mock)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import TypeAdapter, ValidationError

from peanuts.models import AgentAction, AgentRecord, ProviderName, ReplyAction

_ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)
_FENCED_ACTION_RE = re.compile(r"```peanuts-action\s*(\{.*?\})\s*```", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def build_agent_prompt(
    *,
    agent: AgentRecord,
    user_input: str,
    existing_children: Sequence[AgentRecord],
    available_providers: Sequence[ProviderName],
    max_depth: int,
    remaining_questions: int,
    integrator_available: bool = False,
    base_branch: str | None = None,
) -> str:
    """Build a prompt instructing Claude/Codex how to reply, delegate, integrate, or ask the human."""
    can_delegate = agent.depth < max_depth
    example_provider: ProviderName = available_providers[0] if available_providers else "mock"
    integrator_note = (
        f"- You are READ-ONLY on parent branch `{base_branch or 'HEAD'}`: NEVER try to run write git commands "
        "(`git add`, `git commit`, `git merge`) yourself. Instead, emit an `integrate` action so the dedicated "
        "Single-Writer Integrator agent executes it safely.\n"
        if integrator_available
        else ""
    )
    role_desc = (
        "You are the ROOT mediator agent (Peanuts Parent, depth=0), modeled after OpenAI dots.\n"
        "- Your primary job is to route distinct user themes/questions/tasks to dedicated child agents "
        "via the `delegate` action so each theme has its own isolated multi-turn context.\n"
        "- When the human sends a new topic/task (or multiple topics), ALWAYS prefer emitting a `delegate` "
        "peanuts-action block with a concise theme slug (e.g. 'react-hooks', 'auth-bug', 'travel-osaka'). "
        "Reuse an existing child's `agent_id` if the human follows up on that same theme.\n"
        f"{integrator_note}"
        "- When you receive a 'Child agent update' or 'Integrator ...' notification, summarize/report the result "
        "clearly to the human using plain text or `reply`.\n"
        "- Only answer directly without delegating for trivial greetings or meta-questions about Peanuts itself."
        if agent.depth == 0
        else f"You are a CHILD specialist agent (id={agent.id}, theme={agent.theme}, depth={agent.depth}/{max_depth}). "
        "Focus on your delegated theme and return a concise, actionable result to your parent agent."
    )

    children_summary = "None yet."
    if existing_children:
        children_summary = "\n".join(
            f"  - id={c.id}, theme={c.theme}, provider={c.provider}, branch={c.branch_name or '-'}, status={c.status.value}"
            for c in existing_children
        )

    actions_doc = [
        "1. Final reply to caller (default if you respond in plain text):\n"
        '```peanuts-action\n{"action": "reply", "message": "your response"}\n```'
    ]
    if can_delegate:
        providers_str = ", ".join(available_providers)
        actions_doc.append(
            "2. Delegate one or more tasks in parallel to child agents "
            f"(ONLY use providers from [{providers_str}]; include agent_id to continue an existing child):\n"
            "```peanuts-action\n"
            '{"action": "delegate", "message": "brief status note for human", '
            f'"tasks": [{{"theme": "topic-slug", "provider": "{example_provider}", "prompt": "..."}}]}}\n'
            "```"
        )
    if remaining_questions > 0:
        actions_doc.append(
            f"3. Ask the human a clarification question (remaining budget: {remaining_questions}):\n"
            '```peanuts-action\n{"action": "ask_human", "question": "your question"}\n```'
        )
    if agent.depth == 0 and integrator_available:
        actions_doc.append(
            f"4. Dispatch git operations on parent branch `{base_branch or 'HEAD'}` to the Single-Writer Integrator:\n"
            '   - Merge a child branch: {"action": "integrate", "mode": "merge", "agent_id": "<child_id>", "message": "..."}\n'
            '   - Resume paused merges: {"action": "integrate", "mode": "resume", "message": "..."}\n'
            "   - Run git housekeeping (e.g. git add/commit on base branch): "
            '{"action": "integrate", "mode": "instruction", "instruction": "detailed git task", "message": "..."}\n'
            "```peanuts-action\n"
            '{"action": "integrate", "mode": "instruction", "instruction": "Stage and commit initial project files", "message": "Asking Integrator to commit on base branch."}\n'
            "```"
        )

    return (
        f"[PEANUTS SYSTEM CONTEXT]\n"
        f"{role_desc}\n"
        f"Existing direct child agents:\n{children_summary}\n\n"
        f"When you want Peanuts to orchestrate delegation or human clarification, emit ONE "
        f"```peanuts-action ... ``` JSON block. Otherwise plain text is treated as a final reply.\n"
        + "\n".join(actions_doc)
        + f"\n\n[INPUT]\n{user_input}"
    )


def parse_agent_action(raw_output: str) -> AgentAction:
    """Parse a structured AgentAction from CLI output, falling back gracefully to ReplyAction."""
    text = raw_output.strip()
    if not text:
        return ReplyAction(message="")

    candidates: list[str] = []
    for regex in (_FENCED_ACTION_RE, _FENCED_JSON_RE):
        matches = regex.findall(text)
        if matches:
            candidates.append(matches[-1].strip())

    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action" in data:
            try:
                return _ACTION_ADAPTER.validate_python(data)
            except ValidationError:
                continue

    return ReplyAction(message=text)
