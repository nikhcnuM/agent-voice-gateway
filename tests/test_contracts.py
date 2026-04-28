"""Producer- and consumer-side contract tests for agent-voice-gateway.

These tests pin the wire shape of every event the gateway publishes and every
command it consumes against the canonical fixtures and schemas in
``../agent-bus/contracts``. A change that breaks the contract (e.g. dropping
``transcript`` from ``voice.transcription.completed``) fails here, before any
manual run.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_gateway.bus import BusEnvelope
from voice_gateway.config import GatewayConfig
from voice_gateway.models import HermesResult
from voice_gateway.service import GatewayService

from tests.contract_check import (
    ContractError,
    known_types,
    load_fixture,
    load_schema,
    validate_envelope,
)


PRODUCED_EVENTS = (
    "voice.recording.started",
    "voice.recording.stopped",
    "voice.recording.cancelled",
    "voice.transcription.completed",
    "voice.transcription.empty",
    "voice.tts.started",
    "voice.tts.completed",
    "voice.tts.failed",
)

CONSUMED_COMMANDS = (
    "voice.ptt.start",
    "voice.ptt.stop",
    "voice.ptt.cancel",
    "voice.tts.speak",
)


class FakeRecorder:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.starts = 0
        self.stops = 0
        self.cancels = 0

    def start(self, session_id: str):
        self.starts += 1
        audio_path = self.tmp_path / f"{session_id}.wav"
        _write_wav(audio_path, amplitude=8000)
        return SimpleNamespace(session_id=session_id, audio_path=audio_path)

    def stop(self, handle) -> None:
        self.stops += 1

    def cancel(self, handle) -> None:
        self.cancels += 1
        handle.audio_path.unlink(missing_ok=True)


class SilentRecorder(FakeRecorder):
    def start(self, session_id: str):
        self.starts += 1
        audio_path = self.tmp_path / f"{session_id}.wav"
        _write_wav(audio_path, amplitude=0)
        return SimpleNamespace(session_id=session_id, audio_path=audio_path)


class FakeTranscriber:
    def __init__(self, transcript: str = "Run the tests"):
        self.transcript = transcript

    def transcribe(self, audio_path: Path) -> str:
        return self.transcript


class FakeHermes:
    async def ask(self, transcript: str) -> HermesResult:
        return HermesResult(response_id="resp_1", assistant_text="ok")


class FakeLaunchpad:
    async def publish(self, session) -> None:
        return None


class FakeTts:
    def __init__(self, error: Exception | None = None):
        self.calls: list[str] = []
        self.error = error

    async def speak(self, text: str) -> None:
        self.calls.append(text)
        if self.error is not None:
            raise self.error


class CapturingBus:
    def __init__(self) -> None:
        self.events: list[BusEnvelope] = []

    async def publish_event(self, envelope: BusEnvelope) -> None:
        self.events.append(envelope)


def _write_wav(path: Path, amplitude: int, sample_rate: int = 16000) -> None:
    samples = [amplitude if index % 2 == 0 else -amplitude for index in range(sample_rate // 4)]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _build_service(tmp_path: Path, *, transcript: str = "Run the tests", silent: bool = False, tts: FakeTts | None = None) -> tuple[GatewayService, CapturingBus]:
    config = GatewayConfig()
    if tts is not None:
        config.tts.enabled = True
    recorder: FakeRecorder = SilentRecorder(tmp_path) if silent else FakeRecorder(tmp_path)
    bus = CapturingBus()
    service = GatewayService(
        config=config,
        recorder=recorder,
        transcriber=FakeTranscriber(transcript),
        hermes=FakeHermes(),
        launchpad=FakeLaunchpad(),
        tts=tts,
        agent_bus=bus,
    )
    return service, bus


def _envelope_to_dict(envelope: BusEnvelope) -> dict:
    return envelope.to_mapping()


# ---------- Registry sanity ----------

def test_registry_lists_every_type_the_gateway_touches() -> None:
    types = known_types()
    for type_name in (*PRODUCED_EVENTS, *CONSUMED_COMMANDS):
        assert type_name in types, f"missing canonical entry for {type_name}"


# ---------- Producer-side ----------

@pytest.mark.anyio
async def test_recording_lifecycle_emits_envelopes_that_match_fixtures(tmp_path: Path) -> None:
    service, bus = _build_service(tmp_path)

    started = await service.start_ptt()
    await service.stop_ptt()

    types = [event.type for event in bus.events]
    assert types == [
        "voice.recording.started",
        "voice.recording.stopped",
        "voice.transcription.completed",
    ]

    started_event = next(e for e in bus.events if e.type == "voice.recording.started")
    validate_envelope(_envelope_to_dict(started_event))
    assert started_event.payload["session_id"] == started.session_id
    assert started_event.payload["status"] == "recording"

    stopped_event = next(e for e in bus.events if e.type == "voice.recording.stopped")
    validate_envelope(_envelope_to_dict(stopped_event))
    assert stopped_event.payload["status"] == "stopped"

    transcript_event = next(e for e in bus.events if e.type == "voice.transcription.completed")
    validate_envelope(_envelope_to_dict(transcript_event))
    assert isinstance(transcript_event.payload["transcript"], str)
    assert transcript_event.payload["transcript"] == "Run the tests"


@pytest.mark.anyio
async def test_cancel_emits_canonical_recording_cancelled(tmp_path: Path) -> None:
    service, bus = _build_service(tmp_path)

    await service.start_ptt()
    await service.cancel_ptt()

    cancelled = next(e for e in bus.events if e.type == "voice.recording.cancelled")
    validate_envelope(_envelope_to_dict(cancelled))
    assert cancelled.payload["status"] == "cancelled"


@pytest.mark.anyio
async def test_silent_audio_emits_canonical_transcription_empty(tmp_path: Path) -> None:
    service, bus = _build_service(tmp_path, silent=True)

    await service.start_ptt()
    await service.stop_ptt()

    empty = next(e for e in bus.events if e.type == "voice.transcription.empty")
    validate_envelope(_envelope_to_dict(empty))
    assert "session_id" in empty.payload


@pytest.mark.anyio
async def test_tts_speak_emits_started_and_completed(tmp_path: Path) -> None:
    tts = FakeTts()
    service, bus = _build_service(tmp_path, tts=tts)

    await service.speak_tts("Build lista.", correlation_id="ptt-1", session_id="ptt-1")

    types = [event.type for event in bus.events]
    assert types == ["voice.tts.started", "voice.tts.completed"]
    for envelope in bus.events:
        validate_envelope(_envelope_to_dict(envelope))
        assert envelope.payload["text"] == "Build lista."


@pytest.mark.anyio
async def test_tts_failure_emits_canonical_voice_tts_failed(tmp_path: Path) -> None:
    tts = FakeTts(error=RuntimeError("boom"))
    service, bus = _build_service(tmp_path, tts=tts)

    await service.speak_tts("hola", correlation_id="ptt-1", session_id="ptt-1")

    failed = next(e for e in bus.events if e.type == "voice.tts.failed")
    validate_envelope(_envelope_to_dict(failed))
    assert failed.payload["detail"]


def test_voice_transcription_completed_requires_transcript_string() -> None:
    fixture = load_fixture("voice.transcription.completed")
    fixture["payload"].pop("transcript")
    with pytest.raises(ContractError) as info:
        validate_envelope(fixture)
    assert info.value.field == "payload.transcript"


def test_voice_transcription_completed_rejects_non_string_transcript() -> None:
    fixture = load_fixture("voice.transcription.completed")
    fixture["payload"]["transcript"] = 123
    with pytest.raises(ContractError) as info:
        validate_envelope(fixture)
    assert info.value.field == "payload.transcript"


# ---------- Consumer-side ----------

@pytest.mark.anyio
@pytest.mark.parametrize("type_name", CONSUMED_COMMANDS)
async def test_consumed_command_fixtures_match_envelope(type_name: str) -> None:
    fixture = load_fixture(type_name)
    validate_envelope(fixture)
    assert fixture["target"] == "agent-voice-gateway"


@pytest.mark.anyio
async def test_voice_ptt_start_fixture_drives_recording(tmp_path: Path) -> None:
    service, _ = _build_service(tmp_path)
    fixture = load_fixture("voice.ptt.start")
    envelope = BusEnvelope.from_mapping(fixture)

    await service.handle_bus_command(envelope)

    assert service.active_recording is not None


@pytest.mark.anyio
async def test_voice_ptt_stop_fixture_drives_transcription(tmp_path: Path) -> None:
    service, bus = _build_service(tmp_path)
    await service.handle_bus_command(BusEnvelope.from_mapping(load_fixture("voice.ptt.start")))
    await service.handle_bus_command(BusEnvelope.from_mapping(load_fixture("voice.ptt.stop")))

    assert service.active_recording is None
    assert "voice.transcription.completed" in [event.type for event in bus.events]


@pytest.mark.anyio
async def test_voice_ptt_cancel_fixture_drives_cancel(tmp_path: Path) -> None:
    service, bus = _build_service(tmp_path)
    await service.handle_bus_command(BusEnvelope.from_mapping(load_fixture("voice.ptt.start")))
    await service.handle_bus_command(BusEnvelope.from_mapping(load_fixture("voice.ptt.cancel")))

    cancelled = [e for e in bus.events if e.type == "voice.recording.cancelled"]
    assert cancelled, "cancel fixture should produce voice.recording.cancelled"


@pytest.mark.anyio
async def test_voice_tts_speak_fixture_drives_tts(tmp_path: Path) -> None:
    tts = FakeTts()
    service, _ = _build_service(tmp_path, tts=tts)
    fixture = load_fixture("voice.tts.speak")

    await service.handle_bus_command(BusEnvelope.from_mapping(fixture))

    assert tts.calls == [fixture["payload"]["text"]]


@pytest.mark.anyio
@pytest.mark.parametrize("type_name", CONSUMED_COMMANDS)
async def test_command_for_other_target_is_ignored(tmp_path: Path, type_name: str) -> None:
    tts = FakeTts() if type_name == "voice.tts.speak" else None
    service, bus = _build_service(tmp_path, tts=tts)
    fixture = load_fixture(type_name)
    fixture["target"] = "some-other-service"
    envelope = BusEnvelope.from_mapping(fixture)

    await service.handle_bus_command(envelope)

    assert service.active_recording is None
    assert bus.events == []
    if tts is not None:
        assert tts.calls == []


# ---------- Schema invariants ----------

@pytest.mark.parametrize("type_name", PRODUCED_EVENTS + CONSUMED_COMMANDS)
def test_canonical_fixture_validates_against_its_schema(type_name: str) -> None:
    validate_envelope(load_fixture(type_name))


def test_voice_tts_speak_schema_requires_non_empty_text() -> None:
    schema = load_schema("voice.tts.speak")
    assert "text" in schema["required"]
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["text"].get("minLength") == 1
