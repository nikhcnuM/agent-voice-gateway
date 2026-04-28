from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from voice_gateway.audio import FfmpegRecorder, RecordingError, RecordingHandle, RecordingTimeout
from voice_gateway.audio_analysis import analyze_wav, is_too_quiet
from voice_gateway.bus import AgentBusClient, AgentBusError, BusEnvelope
from voice_gateway.config import GatewayConfig
from voice_gateway.hermes import HermesClient, HermesError
from voice_gateway.launchpad import (
    LaunchpadBridgeError,
    LaunchpadBridgeClient,
    awaiting_choice_session,
    completed_session,
    empty_transcript_session,
    error_session,
)
from voice_gateway.models import AgentOption, OptionSelectionResponse, PttResponse
from voice_gateway.transcriber import TranscriptionError, WhisperCliTranscriber
from voice_gateway.tts import TtsClient

logger = logging.getLogger(__name__)


class ActiveSessionMissing(RuntimeError):
    pass


class OptionSessionMissing(RuntimeError):
    pass


class OptionUnavailable(RuntimeError):
    pass


@dataclass
class OptionState:
    session_id: str
    options: dict[str, AgentOption]
    selected_option_id: str | None = None


@dataclass
class GatewayService:
    config: GatewayConfig
    recorder: FfmpegRecorder
    transcriber: WhisperCliTranscriber
    hermes: HermesClient
    launchpad: LaunchpadBridgeClient
    tts: TtsClient | None = None
    agent_bus: AgentBusClient | None = None
    active_recording: RecordingHandle | None = None
    option_sessions: dict[str, OptionState] = field(default_factory=dict)
    _pending_tts_tasks: list[asyncio.Task] = field(default_factory=list, repr=False)

    async def start_ptt(self) -> PttResponse:
        if self.active_recording is not None:
            return PttResponse(session_id=self.active_recording.session_id, status="recording")

        session_id = new_session_id()
        self.active_recording = self.recorder.start(session_id)
        await self._publish_event(
            "voice.recording.started",
            correlation_id=session_id,
            session_id=session_id,
            payload={"session_id": session_id, "status": "recording"},
        )
        return PttResponse(session_id=session_id, status="recording")

    async def stop_ptt(self) -> PttResponse:
        if self.active_recording is None:
            raise ActiveSessionMissing("No active push-to-talk session")

        recording = self.active_recording
        self.active_recording = None

        try:
            self.recorder.stop(recording)
            await self._publish_event(
                "voice.recording.stopped",
                correlation_id=recording.session_id,
                session_id=recording.session_id,
                payload={"session_id": recording.session_id, "status": "stopped"},
            )
            audio_stats = analyze_wav(recording.audio_path)
            if is_too_quiet(
                audio_stats,
                min_rms=self.config.audio.min_rms,
                min_peak=self.config.audio.min_peak,
            ):
                publish_error = await self._publish_session(
                    empty_transcript_session(recording.session_id)
                )
                await self._publish_event(
                    "voice.transcription.empty",
                    correlation_id=recording.session_id,
                    session_id=recording.session_id,
                    payload={"session_id": recording.session_id, "detail": audio_stats.summary},
                )
                return PttResponse(
                    session_id=recording.session_id,
                    status="empty_transcript",
                    detail=_join_details(audio_stats.summary, publish_error),
                )

            transcript = self.transcriber.transcribe(recording.audio_path).strip()
            if not transcript:
                publish_error = await self._publish_session(
                    empty_transcript_session(recording.session_id)
                )
                await self._publish_event(
                    "voice.transcription.empty",
                    correlation_id=recording.session_id,
                    session_id=recording.session_id,
                    payload={"session_id": recording.session_id},
                )
                return PttResponse(
                    session_id=recording.session_id,
                    status="empty_transcript",
                    detail=publish_error,
                )

            await self._publish_event(
                "voice.transcription.completed",
                correlation_id=recording.session_id,
                session_id=recording.session_id,
                payload={"session_id": recording.session_id, "transcript": transcript},
            )
            return PttResponse(
                session_id=recording.session_id,
                status="transcribed",
                transcript=transcript,
            )
        except RecordingTimeout as exc:
            await self._publish_error(recording.session_id, "Recording timed out")
            return PttResponse(session_id=recording.session_id, status="timeout", detail=str(exc))
        except (RecordingError, TranscriptionError) as exc:
            await self._publish_error(recording.session_id, str(exc))
            return PttResponse(session_id=recording.session_id, status="error", detail=str(exc))
        finally:
            if not self.config.audio.keep_temp_files:
                recording.audio_path.unlink(missing_ok=True)

    async def cancel_ptt(self) -> PttResponse:
        if self.active_recording is None:
            raise ActiveSessionMissing("No active push-to-talk session")

        recording = self.active_recording
        self.active_recording = None
        self.recorder.cancel(recording)
        await self._publish_event(
            "voice.recording.cancelled",
            correlation_id=recording.session_id,
            session_id=recording.session_id,
            payload={"session_id": recording.session_id, "status": "cancelled"},
        )
        return PttResponse(session_id=recording.session_id, status="cancelled")

    async def handle_bus_command(self, envelope: BusEnvelope) -> None:
        if envelope.target not in (None, "agent-voice-gateway"):
            return
        if envelope.type == "voice.ptt.start":
            await self.start_ptt()
        elif envelope.type == "voice.ptt.stop":
            await self.stop_ptt()
        elif envelope.type == "voice.ptt.cancel":
            await self.cancel_ptt()
        elif envelope.type == "voice.tts.speak":
            text = str(envelope.payload.get("text") or "")
            await self.speak_tts(text, correlation_id=envelope.correlation_id, session_id=envelope.session_id)

    async def speak_tts(
        self,
        text: str,
        *,
        correlation_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if self.tts is None or not text:
            return
        await self._publish_event(
            "voice.tts.started",
            correlation_id=correlation_id,
            session_id=session_id,
            payload={"text": text},
        )
        try:
            await self.tts.speak(text)
        except Exception as exc:  # noqa: BLE001 - protocol boundary
            await self._publish_event(
                "voice.tts.failed",
                correlation_id=correlation_id,
                session_id=session_id,
                payload={"detail": str(exc)},
            )
            return
        await self._publish_event(
            "voice.tts.completed",
            correlation_id=correlation_id,
            session_id=session_id,
            payload={"text": text},
        )

    async def select_option(self, session_id: str, option_id: str) -> OptionSelectionResponse:
        state = self.option_sessions.get(session_id)
        if state is None:
            raise OptionSessionMissing(f"Unknown option session: {session_id}")
        option = state.options.get(option_id)
        if option is None:
            raise OptionUnavailable(f"Unknown option: {option_id}")
        if not option.enabled:
            raise OptionUnavailable(f"Option disabled: {option_id}")

        state.selected_option_id = option_id
        return OptionSelectionResponse(session_id=session_id, option_id=option_id, status="selected")

    async def selected_option(self, session_id: str) -> OptionSelectionResponse:
        state = self.option_sessions.get(session_id)
        if state is None:
            raise OptionSessionMissing(f"Unknown option session: {session_id}")
        if state.selected_option_id is None:
            raise OptionUnavailable(f"No option selected for session: {session_id}")
        return OptionSelectionResponse(
            session_id=session_id,
            option_id=state.selected_option_id,
            status="selected",
        )

    async def _publish_session(self, session) -> str | None:
        try:
            await self.launchpad.publish(session)
        except LaunchpadBridgeError as exc:
            return f"Launchpad publish failed: {exc}"
        return None

    async def _publish_error(self, session_id: str, message: str) -> None:
        await self._publish_session(error_session(session_id, message))

    async def _publish_event(
        self,
        event_type: str,
        *,
        correlation_id: str | None = None,
        session_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        if self.agent_bus is None:
            return
        try:
            await self.agent_bus.publish_event(
                BusEnvelope(
                    type=event_type,
                    correlation_id=correlation_id,
                    session_id=session_id,
                    payload=dict(payload or {}),
                )
            )
        except AgentBusError as exc:
            logger.warning("Agent bus publish failed: %s", exc)

    def _dispatch_tts(self, text: str | None) -> None:
        if self.tts is None or not text:
            return
        # Housekeeping: drop completed tasks so the list does not grow unbounded.
        self._pending_tts_tasks = [t for t in self._pending_tts_tasks if not t.done()]
        try:
            task = asyncio.create_task(self.tts.speak(text))
        except RuntimeError as exc:  # no running loop (defensive)
            logger.warning("Could not schedule TTS task: %s", exc)
            return
        self._pending_tts_tasks.append(task)


def new_session_id() -> str:
    return "ptt-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _join_details(*details: str | None) -> str | None:
    clean = [detail for detail in details if detail]
    return "; ".join(clean) if clean else None
