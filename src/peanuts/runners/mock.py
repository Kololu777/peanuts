"""Deterministic MockRunner for offline exploration, demos, and automated tests."""

from __future__ import annotations

import asyncio
import json

from peanuts.models import ProviderName
from peanuts.runners.base import ProgressCallback, RunnerError, TurnInput, TurnOutput


def _extract_input_section(prompt: str) -> str:
    marker = "[INPUT]\n"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    return prompt.strip()


def _wrap_action(payload: dict[str, object]) -> str:
    return f"```peanuts-action\n{json.dumps(payload, ensure_ascii=False)}\n```"


class MockRunner:
    """Offline mock runner demonstrating multi-theme delegation, recursion, and human Q&A."""

    provider: ProviderName = "mock"

    def __init__(self, delay_seconds: float = 0.02) -> None:
        self.delay_seconds = delay_seconds

    async def run_turn(
        self,
        turn: TurnInput,
        on_progress: ProgressCallback | None = None,
    ) -> TurnOutput:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        agent = turn.agent
        session_id = agent.provider_session_id or f"mock-sess-{agent.id}"
        user_input = _extract_input_section(turn.prompt)
        lower = user_input.lower()

        if on_progress is not None and agent.depth > 0:
            await on_progress(f"Working on '{agent.theme}': {user_input[:80]}")

        if "slow_child" in lower and agent.depth > 0:
            await asyncio.sleep(0.25)

        # Failure simulation for testing error recovery
        if "force_fail" in lower and agent.depth > 0:
            raise RunnerError(f"Simulated CLI error on child agent {agent.id}")

        # Synthesize child results (either async root update or coalesced grandchild batch)
        if user_input.startswith("All delegated child tasks have settled.") or user_input.startswith(
            "Child agent update"
        ):
            summary_lines = [line for line in user_input.splitlines() if line.strip()]
            joined = "\n".join(summary_lines)
            return TurnOutput(
                text=_wrap_action(
                    {
                        "action": "reply",
                        "message": f"[{agent.theme} report]\n{joined}",
                    }
                ),
                provider_session_id=session_id,
            )

        # Resume after human answered a clarification question
        if user_input.startswith("Human answered your question"):
            return TurnOutput(
                text=_wrap_action(
                    {
                        "action": "reply",
                        "message": f"[{agent.theme}] Completed using human answer: {user_input}",
                    }
                ),
                provider_session_id=session_id,
            )

        # Root mediator behaviors
        if agent.depth == 0:
            if lower.startswith("direct:"):
                return TurnOutput(
                    text=_wrap_action({"action": "reply", "message": f"[root direct] {user_input[7:].strip()}"}),
                    provider_session_id=session_id,
                )
            if "ask_root" in lower:
                return TurnOutput(
                    text=_wrap_action(
                        {
                            "action": "ask_human",
                            "question": f"Root needs clarification before proceeding on: {user_input}",
                        }
                    ),
                    provider_session_id=session_id,
                )
            if "recursive" in lower or "孫" in user_input:
                return TurnOutput(
                    text=_wrap_action(
                        {
                            "action": "delegate",
                            "message": "Spawning child agent that will recursively delegate to a grandchild.",
                            "tasks": [
                                {
                                    "theme": "parent-workstream",
                                    "provider": "mock",
                                    "prompt": f"spawn_grandchild: {user_input}",
                                }
                            ],
                        }
                    ),
                    provider_session_id=session_id,
                )
            # Multi-theme parallel split via ';'
            if ";" in user_input:
                parts = [p.strip() for p in user_input.split(";") if p.strip()]
                tasks: list[dict[str, str]] = []
                for idx, part in enumerate(parts, start=1):
                    if ":" in part:
                        theme_raw, body = part.split(":", 1)
                        theme = theme_raw.strip() or f"theme-{idx}"
                        prompt_body = body.strip() or part
                    else:
                        theme = f"theme-{idx}"
                        prompt_body = part
                    tasks.append({"theme": theme, "provider": "mock", "prompt": prompt_body})
                return TurnOutput(
                    text=_wrap_action(
                        {
                            "action": "delegate",
                            "message": f"Routing {len(tasks)} themes in parallel.",
                            "tasks": tasks,
                        }
                    ),
                    provider_session_id=session_id,
                )
            # Default root behavior: route to a focused theme child
            theme = "general"
            if ":" in user_input and len(user_input.split(":", 1)[0].strip()) <= 24:
                theme = user_input.split(":", 1)[0].strip()
            return TurnOutput(
                text=_wrap_action(
                    {
                        "action": "delegate",
                        "message": f"Delegating theme '{theme}' to a child agent.",
                        "tasks": [{"theme": theme, "provider": "mock", "prompt": user_input}],
                    }
                ),
                provider_session_id=session_id,
            )

        # Child / grandchild behaviors
        if "spawn_grandchild" in lower and agent.depth == 1:
            return TurnOutput(
                text=_wrap_action(
                    {
                        "action": "delegate",
                        "message": "Child delegating verification subtask to grandchild.",
                        "tasks": [
                            {
                                "theme": f"{agent.theme}-sub",
                                "provider": "mock",
                                "prompt": f"Execute grandchild task for: {user_input}",
                            }
                        ],
                    }
                ),
                provider_session_id=session_id,
            )

        if "ask_human" in lower or "確認" in user_input:
            return TurnOutput(
                text=_wrap_action(
                    {
                        "action": "ask_human",
                        "question": f"[{agent.theme}] Should I proceed with '{user_input}'?",
                    }
                ),
                provider_session_id=session_id,
            )

        return TurnOutput(
            text=_wrap_action(
                {
                    "action": "reply",
                    "message": f"[{agent.theme} / depth={agent.depth} / sess={session_id}] Done: {user_input}",
                }
            ),
            provider_session_id=session_id,
        )
