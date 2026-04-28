from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status

from voice_gateway.audio import FfmpegRecorder
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


def create_app(config: GatewayConfig | None = None, service: GatewayService | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="Voice Gateway", version="0.1.0")
    app.state.config = config
    app.state.service = service or GatewayService(
        config=config,
        recorder=FfmpegRecorder(config.audio),
        transcriber=WhisperCliTranscriber(config.whisper),
        hermes=HermesClient(config.hermes),
        launchpad=LaunchpadBridgeClient(config.launchpad_bridge),
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            audio_input=config.audio.input_device,
            whisper_model=config.whisper.model,
            hermes_url=config.hermes.base_url,
            tts="disabled",
        )

    @app.post("/ptt/start")
    async def ptt_start(request: Request):
        return await gateway(request).start_ptt()

    @app.post("/ptt/stop")
    async def ptt_stop(request: Request):
        try:
            return await gateway(request).stop_ptt()
        except ActiveSessionMissing as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post("/ptt/cancel")
    async def ptt_cancel(request: Request):
        try:
            return await gateway(request).cancel_ptt()
        except ActiveSessionMissing as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

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


app = create_app()
