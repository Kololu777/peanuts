"""Thin async HTTP + WebSocket client for Peanuts (used by Textual TUI; mirrors React API usage)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import websockets

from peanuts.models import (
    AgentRecord,
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
)


class PeanutsClient:
    """HTTP + WebSocket client speaking strictly the public `/api/v1` protocol."""

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    def ws_url(self, after_seq: int = 0) -> str:
        if self.base_url.startswith("https://"):
            prefix = "wss://" + self.base_url[len("https://") :]
        elif self.base_url.startswith("http://"):
            prefix = "ws://" + self.base_url[len("http://") :]
        else:
            prefix = f"ws://{self.base_url}"
        return f"{prefix}/api/v1/ws?after_seq={after_seq}"

    async def health(self) -> HealthResponse:
        resp = await self._http.get("/api/v1/health")
        resp.raise_for_status()
        return HealthResponse.model_validate(resp.json())

    async def get_state(self, message_limit: int = 100) -> SystemStateResponse:
        resp = await self._http.get("/api/v1/state", params={"message_limit": message_limit})
        resp.raise_for_status()
        return SystemStateResponse.model_validate(resp.json())

    async def get_agent_messages(self, agent_id: str, limit: int = 100) -> list[MessageRecord]:
        resp = await self._http.get(f"/api/v1/agents/{agent_id}/messages", params={"limit": limit})
        resp.raise_for_status()
        return [MessageRecord.model_validate(item) for item in resp.json()]

    async def send_message(
        self,
        content: str,
        *,
        target_agent_id: str | None = None,
        theme: str | None = None,
        provider: ProviderName | None = None,
    ) -> SendMessageResponse:
        req = SendMessageRequest(
            content=content,
            target_agent_id=target_agent_id,
            theme=theme,
            provider=provider,
        )
        resp = await self._http.post("/api/v1/messages", json=req.model_dump(mode="json"))
        resp.raise_for_status()
        return SendMessageResponse.model_validate(resp.json())

    async def answer_question(self, question_id: str, answer: str) -> QuestionRecord:
        resp = await self._http.post(
            f"/api/v1/questions/{question_id}/answer",
            json={"answer": answer},
        )
        resp.raise_for_status()
        return QuestionRecord.model_validate(resp.json())

    async def set_provider(self, agent_id: str, provider: ProviderName) -> AgentRecord:
        resp = await self._http.patch(
            f"/api/v1/agents/{agent_id}/provider",
            json={"provider": provider},
        )
        resp.raise_for_status()
        return AgentRecord.model_validate(resp.json())

    async def retry_agent(self, agent_id: str) -> AgentRecord:
        resp = await self._http.post(f"/api/v1/agents/{agent_id}/retry")
        resp.raise_for_status()
        return AgentRecord.model_validate(resp.json())

    async def cancel_delegation(self, delegation_id: str) -> DelegationRecord:
        resp = await self._http.post(f"/api/v1/delegations/{delegation_id}/cancel")
        resp.raise_for_status()
        return DelegationRecord.model_validate(resp.json())

    async def trigger_merge(self, agent_id: str) -> MergeJobRecord:
        resp = await self._http.post(f"/api/v1/agents/{agent_id}/merge")
        resp.raise_for_status()
        return MergeJobRecord.model_validate(resp.json())

    async def resume_merges(self) -> list[MergeJobRecord]:
        resp = await self._http.post("/api/v1/merges/resume")
        resp.raise_for_status()
        return [MergeJobRecord.model_validate(item) for item in resp.json()]

    async def stream_events(self, after_seq: int = 0) -> AsyncIterator[EventRecord]:
        async with websockets.connect(self.ws_url(after_seq=after_seq)) as ws:
            async for raw in ws:
                data = json.loads(raw)
                if data.get("type") == "ping":
                    continue
                yield EventRecord.model_validate(data)
