from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException, Request, status

from voice_gateway.audio import FfmpegRecorder
from voice_gateway.bus import AgentBusClient, AgentBusError
from voice_gateway.config import GatewayConfig, load_config
from voice_gateway.hermes import HermesClient
from voice_gateway.launchpad import LaunchpadBridgeClient
from voice_gateway.models import HealthResponse, OptionSelectionRequest
from voice_gateway.service import (
    ActiveSessionMissing,
    GatewayService,
    OptionSessionMissing,
    OptionUnavailable,
)
from voice_gateway.transcriber import WhisperCliTranscriber
from voice_gateway.tts import GeminiTtsClient

logger = logging.getLogger(__name__)


def create_app(config: GatewayConfig | None = None, service: GatewayService | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="Voice Gateway", version="0.1.0")
    app.state.config = config
    if service is None:
        tts_client = GeminiTtsClient(config.tts) if config.tts.enabled else None
        bus_client = AgentBusClient(config.agent_bus) if config.agent_bus.enabled else None
        service = GatewayService(
            config=config,
            recorder=FfmpegRecorder(config.audio),
            transcriber=WhisperCliTranscriber(config.whisper),
            hermes=HermesClient(config.hermes),
            launchpad=LaunchpadBridgeClient(config.launchpad_bridge),
            tts=tts_client,
            agent_bus=bus_client,
        )
    app.state.service = service
    app.state.bus_task = None

    @app.on_event("startup")
    async def startup() -> None:
        if config.agent_bus.enabled and service.agent_bus is not None:
            app.state.bus_task = asyncio.create_task(_consume_bus_commands(service))

    @app.on_event("shutdown")
    async def shutdown() -> None:
        task = app.state.bus_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            audio_input=config.audio.input_device,
            whisper_model=config.whisper.model,
            hermes_url=config.hermes.base_url,
            tts=config.tts.provider if config.tts.enabled else "disabled",
        )

    @app.post("/launchpad/options/{session_id}/response")
    async def launchpad_option_response(
        session_id: str,
        payload: OptionSelectionRequest,
        request: Request,
    ):
        try:
            return await gateway(request).select_option(session_id, payload.option_id)
        except OptionSessionMissing as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except OptionUnavailable as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.get("/launchpad/options/{session_id}/response")
    async def launchpad_selected_option(session_id: str, request: Request):
        try:
            return await gateway(request).selected_option(session_id)
        except OptionSessionMissing as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except OptionUnavailable as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return app


def gateway(request: Request) -> GatewayService:
    return request.app.state.service


async def _consume_bus_commands(service: GatewayService) -> None:
    assert service.agent_bus is not None
    backoff_seconds = max(service.config.agent_bus.poll_interval_seconds, 0.2)
    while True:
        try:
            messages = await service.agent_bus.consume_commands(count=10)
        except AgentBusError as exc:
            logger.warning("agent_bus_consume_failed error=%s", exc)
            await asyncio.sleep(backoff_seconds)
            continue

        ack_ids: list[str] = []
        for message in messages:
            try:
                await service.handle_bus_command(message.envelope)
            except ActiveSessionMissing:
                logger.warning("agent_bus_command_discarded stream_id=%s reason=active_session_missing", message.stream_id)
                ack_ids.append(message.stream_id)
            except Exception as exc:  # noqa: BLE001 - keep consumer alive; leave unacked for retry/recovery
                logger.warning("agent_bus_command_failed stream_id=%s error=%s", message.stream_id, exc)
            else:
                ack_ids.append(message.stream_id)
        try:
            await service.agent_bus.ack_commands(ack_ids)
        except AgentBusError as exc:
            logger.warning("agent_bus_ack_failed error=%s", exc)
            await asyncio.sleep(backoff_seconds)
            continue
        await asyncio.sleep(service.config.agent_bus.poll_interval_seconds)


app = create_app()
