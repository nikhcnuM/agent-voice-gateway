from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Literal
from uuid import uuid4

import httpx

from voice_gateway.config import AgentBusConfig

StreamKind = Literal["commands", "events"]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class BusEnvelope:
    type: str
    source: str = "agent-voice-gateway"
    id: str = field(default_factory=lambda: "evt_" + uuid4().hex)
    timestamp: str = field(default_factory=_timestamp)
    correlation_id: str | None = None
    session_id: str | None = None
    target: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BusEnvelope":
        return cls(
            id=str(value.get("id") or "evt_" + uuid4().hex),
            type=str(value["type"]),
            source=str(value.get("source") or "unknown"),
            timestamp=str(value.get("timestamp") or _timestamp()),
            correlation_id=value.get("correlation_id"),
            session_id=value.get("session_id"),
            target=value.get("target"),
            payload=value.get("payload") if isinstance(value.get("payload"), dict) else {},
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "target": self.target,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class StoredBusEnvelope:
    stream_id: str
    envelope: BusEnvelope


class AgentBusError(RuntimeError):
    pass


class AgentBusClient:
    def __init__(self, config: AgentBusConfig, http_client: httpx.AsyncClient | None = None):
        if not config.base_url.startswith("http://127.0.0.1:"):
            raise ValueError("agent bus client must target http://127.0.0.1")
        self.config = config
        self._client = http_client

    async def publish_event(self, envelope: BusEnvelope) -> None:
        await self._request("POST", "/events", envelope.to_mapping())

    async def consume_commands(self, count: int = 10) -> list[StoredBusEnvelope]:
        payload = await self._request(
            "GET",
            f"/consume/commands?group={self.config.group}&consumer={self.config.consumer}&count={count}&block_ms=1",
        )
        return [
            StoredBusEnvelope(
                stream_id=str(item["stream_id"]),
                envelope=BusEnvelope.from_mapping(item["envelope"]),
            )
            for item in payload.get("messages", [])
        ]

    async def ack_commands(self, stream_ids: list[str]) -> None:
        if not stream_ids:
            return
        await self._request(
            "POST",
            f"/consume/commands/ack?group={self.config.group}",
            {"stream_ids": stream_ids},
        )

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        client = self._client or httpx.AsyncClient(timeout=10)
        close_client = self._client is None
        try:
            response = await client.request(method, f"{self.config.base_url.rstrip('/')}{path}", json=payload)
            response.raise_for_status()
            return json.loads(response.text) if response.text else {}
        except httpx.HTTPError as exc:
            raise AgentBusError(str(exc)) from exc
        finally:
            if close_client:
                await client.aclose()
