from __future__ import annotations

import httpx

from voice_gateway.config import LaunchpadBridgeConfig
from voice_gateway.models import AgentAlert, AgentSession, HermesResult


class LaunchpadBridgeError(RuntimeError):
    pass


class LaunchpadBridgeClient:
    def __init__(
        self,
        config: LaunchpadBridgeConfig,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self._client = http_client

    async def publish(self, session: AgentSession) -> None:
        if not self.config.publish_alerts:
            return

        client = self._client or httpx.AsyncClient(timeout=10)
        close_client = self._client is None
        try:
            url = f"{self.config.base_url.rstrip('/')}/agent/session"
            response = await client.put(url, json=session.model_dump(exclude_none=True))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LaunchpadBridgeError(str(exc)) from exc
        finally:
            if close_client:
                await client.aclose()


def completed_session(result: HermesResult) -> AgentSession:
    return AgentSession(
        session_id=result.response_id or "hermes-response",
        status="completed",
        summary=result.assistant_text,
        alert=AgentAlert(severity="info", color="yellow"),
        options=[],
    )


def awaiting_choice_session(result: HermesResult) -> AgentSession:
    return AgentSession(
        session_id=result.response_id or "hermes-response",
        status="awaiting_choice",
        summary=result.assistant_text,
        alert=AgentAlert(severity="info", color="yellow"),
        options=result.options,
    )


def error_session(session_id: str, message: str) -> AgentSession:
    return AgentSession(
        session_id=session_id,
        status="error",
        summary=message,
        alert=AgentAlert(severity="critical", color="red"),
        options=[],
    )


def empty_transcript_session(session_id: str) -> AgentSession:
    return AgentSession(
        session_id=session_id,
        status="needs_attention",
        summary="No speech detected.",
        alert=AgentAlert(severity="warning", color="amber"),
        options=[],
    )
