from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from voice_gateway.audio import FfmpegRecorder, RecordingError, RecordingHandle, RecordingTimeout
from voice_gateway.audio_analysis import analyze_wav, is_too_quiet
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
    active_recording: RecordingHandle | None = None
    option_sessions: dict[str, OptionState] = field(default_factory=dict)

    async def start_ptt(self) -> PttResponse:
        if self.active_recording is not None:
            return PttResponse(session_id=self.active_recording.session_id, status="recording")

        session_id = new_session_id()
        self.active_recording = self.recorder.start(session_id)
        return PttResponse(session_id=session_id, status="recording")

    async def stop_ptt(self) -> PttResponse:
        if self.active_recording is None:
            raise ActiveSessionMissing("No active push-to-talk session")

        recording = self.active_recording
        self.active_recording = None

        try:
            self.recorder.stop(recording)
            audio_stats = analyze_wav(recording.audio_path)
            if is_too_quiet(
                audio_stats,
                min_rms=self.config.audio.min_rms,
                min_peak=self.config.audio.min_peak,
            ):
                publish_error = await self._publish_session(
                    empty_transcript_session(recording.session_id)
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
                return PttResponse(
                    session_id=recording.session_id,
                    status="empty_transcript",
                    detail=publish_error,
                )

            try:
                result = await self.hermes.ask(transcript)
            except HermesError as exc:
                await self._publish_error(recording.session_id, str(exc))
                return PttResponse(
                    session_id=recording.session_id,
                    status="hermes_error",
                    transcript=transcript,
                    detail=str(exc),
                )

            result_options = result.options
            if not result_options and self.config.launchpad_bridge.completed_ack_option:
                result_options = [
                    AgentOption(
                        option_id="ack",
                        label="OK",
                        color="green",
                        semantic="confirm",
                        position="grid:0,0",
                    )
                ]

            if result_options:
                result.options = result_options
                agent_session = awaiting_choice_session(result)
                self.option_sessions[agent_session.session_id] = OptionState(
                    session_id=agent_session.session_id,
                    options={option.option_id: option for option in result_options},
                )
            else:
                agent_session = completed_session(result)

            publish_error = await self._publish_session(agent_session)
            return PttResponse(
                session_id=recording.session_id,
                status=agent_session.status,
                transcript=transcript,
                hermes_response_id=result.response_id,
                assistant_text=result.assistant_text,
                detail=publish_error,
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
        return PttResponse(session_id=recording.session_id, status="cancelled")

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


def new_session_id() -> str:
    return "ptt-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _join_details(*details: str | None) -> str | None:
    clean = [detail for detail in details if detail]
    return "; ".join(clean) if clean else None
