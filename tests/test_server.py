"""Tests for FastAPI REST endpoints, WebSocket event replay, and OpenAPI schema export."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from peanuts.config import PeanutsConfig
from peanuts.server import create_app


def test_rest_api_and_openapi_for_react(tmp_path: Path) -> None:
    cfg = PeanutsConfig(
        db_path=tmp_path / "api.db",
        cwd=tmp_path,
        root_provider="mock",
        default_child_provider="mock",
        max_depth=2,
    )
    app = create_app(cfg)

    with TestClient(app) as client:
        # Health & OpenAPI schema
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        hdata = health.json()
        assert hdata["status"] == "ok"
        assert hdata["root_agent_id"] == "root"

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        paths = openapi.json()["paths"]
        assert "/api/v1/messages" in paths
        assert "/api/v1/state" in paths
        assert "/api/v1/questions/{question_id}/answer" in paths

        # Send a message that triggers child ask_human
        send_resp = client.post("/api/v1/messages", json={"content": "db: 確認 実行してよいですか"})
        assert send_resp.status_code == 200

        # Wait until background turns settle using test orchestrator handle
        import asyncio

        orch = app.state.orchestrator
        # Poll state via REST until open_questions appears
        state_resp = client.get("/api/v1/state")
        for _ in range(50):
            state_resp = client.get("/api/v1/state")
            if state_resp.json()["open_questions"]:
                break
            import time

            time.sleep(0.03)

        state = state_resp.json()
        assert len(state["open_questions"]) == 1
        qid = state["open_questions"][0]["id"]

        # Answer via REST
        ans_resp = client.post(f"/api/v1/questions/{qid}/answer", json={"answer": "OK!"})
        assert ans_resp.status_code == 200
        assert ans_resp.json()["status"] == "ANSWERED"

        # Wait for root to return IDLE
        for _ in range(50):
            s = client.get("/api/v1/state").json()
            root_agent = next(a for a in s["agents"] if a["id"] == "root")
            if root_agent["status"] == "IDLE":
                break
            import time

            time.sleep(0.03)

        final_state = client.get("/api/v1/state").json()
        assert final_state["open_questions"] == []
        assert any(a["theme"] == "db" for a in final_state["agents"])
        assert orch is not None
        assert asyncio is not None

        # Verify WebSocket replays persisted events cleanly
        with client.websocket_connect("/api/v1/ws?after_seq=0") as ws:
            first_event = ws.receive_json()
            assert first_event["seq"] >= 1
            assert "type" in first_event
            assert "payload" in first_event


async def test_textual_tui_end_to_end_against_live_fastapi(tmp_path: Path) -> None:
    """Verify the minimal Textual TUI talks cleanly over HTTP+WS to the live FastAPI server."""
    import asyncio
    import threading

    import uvicorn

    from peanuts.cli import _find_free_port, _wait_for_health
    from peanuts.tui import PeanutsTUI

    port = _find_free_port("127.0.0.1")
    cfg = PeanutsConfig(
        db_path=tmp_path / "tui_e2e.db",
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
            await tui._handle_command_or_message("alpha: hello; beta: world")
            for _ in range(30):
                await pilot.pause(0.05)
                if tui._state is not None and len(tui._state.agents) >= 3:
                    break
            assert tui._state is not None
            themes = {a.theme for a in tui._state.agents}
            assert {"root", "alpha", "beta"}.issubset(themes)

            # Wait until alpha finishes its mock turn, then select it in the TUI
            for _ in range(30):
                await pilot.pause(0.05)
                if tui._state is not None:
                    alpha_cur = next((a for a in tui._state.agents if a.theme == "alpha"), None)
                    if alpha_cur is not None and alpha_cur.status.value == "IDLE":
                        break
            assert tui._state is not None
            alpha = next(a for a in tui._state.agents if a.theme == "alpha")
            alpha_short = tui._short_id_by_agent[alpha.id]

            # `@#01 follow up` -> Human -> Root (displayed) -> deterministic Root -> alpha delegation -> Root report!
            await tui._handle_command_or_message(f"@#{alpha_short} direct follow up")
            for _ in range(40):
                await pilot.pause(0.05)
                root_msgs_now = await tui.client.get_agent_messages("root")
                if any("direct follow up" in m.content and m.role.value == "agent" for m in root_msgs_now):
                    break
            root_msgs_after = await tui.client.get_agent_messages("root")
            assert any(f"@#{alpha_short} direct follow up" in m.content for m in root_msgs_after)
            assert any("direct follow up" in m.content for m in await tui.client.get_agent_messages(alpha.id))

            # Switch to child via `#01` (2-digit short ID)
            await tui._handle_command_or_message(f"#{alpha_short}")
            assert tui._selected_agent_id == alpha.id
            alpha_msgs = await tui.client.get_agent_messages(alpha.id)
            assert any(m.role.value == "progress" for m in alpha_msgs)
            assert len(tui._seen_message_ids) == len(alpha_msgs)
    finally:
        server.should_exit = True
        await asyncio.to_thread(thread.join, 3.0)
