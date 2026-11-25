"""FastAPI REST + WebSocket server (shared backend for Textual TUI and future Web+React UI)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from peanuts import __version__
from peanuts.config import PeanutsConfig
from peanuts.git_ops import GitError
from peanuts.models import (
    AgentRecord,
    AnswerQuestionRequest,
    DelegationRecord,
    EventRecord,
    HealthResponse,
    MergeJobRecord,
    MessageRecord,
    ProviderName,
    QuestionRecord,
    SendMessageRequest,
    SendMessageResponse,
    SystemStateResponse,
    UpdateProviderRequest,
)
from peanuts.orchestrator import Orchestrator
from peanuts.runners import AgentRunner
from peanuts.state_machine import StateTransitionError


def create_app(
    config: PeanutsConfig | None = None,
    *,
    runners: Mapping[ProviderName, AgentRunner] | None = None,
) -> FastAPI:
    """Create a FastAPI application exposing REST + WebSocket endpoints for Peanuts."""
    cfg = config or PeanutsConfig()
    orchestrator = Orchestrator(cfg, runners=runners)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await orchestrator.shutdown()

    app = FastAPI(
        title="Peanuts Orchestrator API",
        version=__version__,
        description=(
            "Parent-child CLI agent mediator API (Claude Code / Codex / Mock). "
            "Consumed by the Textual TUI and ready for Web + React frontends."
        ),
        lifespan=lifespan,
    )
    app.state.orchestrator = orchestrator
    app.state.config = cfg

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # REST Endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            version=__version__,
            root_agent_id=orchestrator.root_agent_id,
            base_branch=orchestrator.git_ctx.base_branch,
            worktrees_enabled=orchestrator.worktree_mgr is not None,
            max_depth=cfg.max_depth,
            max_questions_per_turn=cfg.max_questions_per_turn,
            providers=cfg.detect_providers(),
        )

    @app.get("/api/v1/state", response_model=SystemStateResponse, tags=["state"])
    async def get_state(message_limit: int = Query(default=100, ge=1, le=500)) -> SystemStateResponse:
        return orchestrator.get_state(message_limit=message_limit)

    @app.get("/api/v1/agents", response_model=list[AgentRecord], tags=["agents"])
    async def list_agents() -> list[AgentRecord]:
        return orchestrator.repo.list_agents()

    @app.get("/api/v1/agents/{agent_id}/messages", response_model=list[MessageRecord], tags=["messages"])
    async def get_agent_messages(
        agent_id: str,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[MessageRecord]:
        if orchestrator.repo.get_agent(agent_id) is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return orchestrator.repo.list_messages(agent_id=agent_id, limit=limit)

    @app.patch("/api/v1/agents/{agent_id}/provider", response_model=AgentRecord, tags=["agents"])
    async def update_agent_provider(agent_id: str, body: UpdateProviderRequest) -> AgentRecord:
        try:
            return await orchestrator.set_agent_provider(agent_id, body.provider)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/agents/{agent_id}/retry", response_model=AgentRecord, tags=["agents"])
    async def retry_agent(agent_id: str) -> AgentRecord:
        try:
            return await orchestrator.retry_agent(agent_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/messages", response_model=SendMessageResponse, tags=["messages"])
    async def send_message(body: SendMessageRequest) -> SendMessageResponse:
        try:
            agent_id, msg = await orchestrator.send_human_message(
                body.content,
                target_agent_id=body.target_agent_id,
                theme=body.theme,
                provider=body.provider,
            )
            return SendMessageResponse(agent_id=agent_id, message=msg)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/delegations", response_model=list[DelegationRecord], tags=["delegations"])
    async def list_delegations() -> list[DelegationRecord]:
        return orchestrator.repo.list_delegations()

    @app.post("/api/v1/delegations/{delegation_id}/cancel", response_model=DelegationRecord, tags=["delegations"])
    async def cancel_delegation(delegation_id: str) -> DelegationRecord:
        try:
            return await orchestrator.cancel_delegation(delegation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/questions", response_model=list[QuestionRecord], tags=["questions"])
    async def list_questions(only_open: bool = Query(default=True)) -> list[QuestionRecord]:
        return orchestrator.repo.list_questions(only_open=only_open)

    @app.post("/api/v1/questions/{question_id}/answer", response_model=QuestionRecord, tags=["questions"])
    async def answer_question(question_id: str, body: AnswerQuestionRequest) -> QuestionRecord:
        try:
            return await orchestrator.answer_question(question_id, body.answer)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StateTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/merges", response_model=list[MergeJobRecord], tags=["merges"])
    async def list_merges() -> list[MergeJobRecord]:
        return orchestrator.repo.list_merge_jobs()

    @app.post("/api/v1/agents/{agent_id}/merge", response_model=MergeJobRecord, tags=["merges"])
    async def trigger_agent_merge(agent_id: str) -> MergeJobRecord:
        try:
            return await orchestrator.enqueue_merge_for_agent(agent_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except GitError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/merges/resume", response_model=list[MergeJobRecord], tags=["merges"])
    async def resume_merges() -> list[MergeJobRecord]:
        return await orchestrator.resume_merge_queue()

    @app.get("/api/v1/events", response_model=list[EventRecord], tags=["events"])
    async def list_events(after_seq: int = Query(default=0, ge=0), limit: int = Query(default=200, ge=1, le=1000)) -> list[EventRecord]:
        return orchestrator.repo.list_events_after(after_seq=after_seq, limit=limit)

    # ------------------------------------------------------------------
    # WebSocket Event Stream (supports gapless replay via after_seq)
    # ------------------------------------------------------------------

    @app.websocket("/api/v1/ws")
    async def events_ws(websocket: WebSocket, after_seq: int = Query(default=0, ge=0)) -> None:
        await websocket.accept()
        last_sent_seq = after_seq
        try:
            async with orchestrator.event_bus.subscribe() as queue:
                # 1. Replay persisted events newer than after_seq
                backlog = orchestrator.repo.list_events_after(after_seq=last_sent_seq, limit=500)
                for ev in backlog:
                    await websocket.send_json(ev.model_dump(mode="json"))
                    last_sent_seq = max(last_sent_seq, ev.seq)

                # 2. Stream live events, deduplicating anything already sent in backlog
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=20.0)
                    except TimeoutError:
                        await websocket.send_json({"type": "ping", "seq": last_sent_seq})
                        continue
                    if ev.seq <= last_sent_seq:
                        continue
                    await websocket.send_json(ev.model_dump(mode="json"))
                    last_sent_seq = ev.seq
        except WebSocketDisconnect:
            return

    return app
