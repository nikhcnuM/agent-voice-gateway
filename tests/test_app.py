from __future__ import annotations

import asyncio
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from voice_gateway.app import _consume_bus_commands, create_app
from voice_gateway.bus import AgentBusError
from voice_gateway.config import GatewayConfig
from voice_gateway.hermes import HermesError
from voice_gateway.launchpad import LaunchpadBridgeError
from voice_gateway.models import AgentOption, HermesResult
from voice_gateway.service import ActiveSessionMissing, GatewayService


class FakeRecorder:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.starts = 0
        self.stops = 0
        self.cancels = 0

    def start(self, session_id: str):
        self.starts += 1
        audio_path = self.tmp_path / f"{session_id}.wav"
        write_test_wav(audio_path, amplitude=8000)
        return SimpleNamespace(session_id=session_id, audio_path=audio_path)

    def stop(self, handle) -> None:
        self.stops += 1

    def cancel(self, handle) -> None:
        self.cancels += 1
        handle.audio_path.unlink(missing_ok=True)


class FakeTranscriber:
    def __init__(self, transcript: str = "Run the tests"):
        self.transcript = transcript

    def transcribe(self, audio_path: Path) -> str:
        return self.transcript


class FakeHermes:
    def __init__(self, result: HermesResult | None = None):
        self.result = result or HermesResult(response_id="resp_123", assistant_text="Tests passed.")
        self.transcripts: list[str] = []

    async def ask(self, transcript: str) -> HermesResult:
        self.transcripts.append(transcript)
        return self.result


class FailingHermes(FakeHermes):
    async def ask(self, transcript: str) -> HermesResult:
        self.transcripts.append(transcript)
        raise HermesError("Hermes unavailable")


class FakeLaunchpad:
    def __init__(self):
        self.sessions = []

    async def publish(self, session) -> None:
        self.sessions.append(session)


class FailingLaunchpad(FakeLaunchpad):
    async def publish(self, session) -> None:
        self.sessions.append(session)
        raise LaunchpadBridgeError("bridge unavailable")


class FakeTts:
    def __init__(self, error: Exception | None = None):
        self.calls: list[str] = []
        self.error = error

    async def speak(self, text: str) -> None:
        self.calls.append(text)
        if self.error is not None:
            raise self.error


class FakeBus:
    def __init__(self):
        self.events = []
        self.commands = []
        self.acks = []

    async def publish_event(self, envelope) -> None:
        self.events.append(envelope)

    async def consume_commands(self, count: int = 10):
        commands, self.commands = self.commands[:count], self.commands[count:]
        return commands

    async def ack_commands(self, stream_ids: list[str]) -> None:
        self.acks.append(stream_ids)


class FlakyCommandBus(FakeBus):
    def __init__(self):
        super().__init__()
        self.fail_next_consume = True

    async def consume_commands(self, count: int = 10):
        if self.fail_next_consume:
            self.fail_next_consume = False
            raise AgentBusError("bus temporarily unavailable")
        return await super().consume_commands(count=count)


def build_client(tmp_path: Path, transcript: str = "Run the tests", result: HermesResult | None = None, tts: FakeTts | None = None):
    config = GatewayConfig()
    if tts is not None:
        config.tts.enabled = True
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber(transcript)
    hermes = FakeHermes(result)
    launchpad = FakeLaunchpad()
    bus = FakeBus()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad, tts=tts, agent_bus=bus)
    return TestClient(create_app(config=config, service=service)), service, recorder, hermes, launchpad, bus


def write_test_wav(path: Path, amplitude: int, sample_rate: int = 16000) -> None:
    samples = [amplitude if index % 2 == 0 else -amplitude for index in range(sample_rate // 4)]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def test_health_reports_configured_dependencies(tmp_path: Path):
    client, *_ = build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["hermes_url"] == "http://127.0.0.1:8642/v1"
    assert response.json()["tts"] == "disabled"


def test_ptt_http_routes_are_removed(tmp_path: Path):
    client, *_ = build_client(tmp_path)

    assert client.post("/ptt/start").status_code == 404
    assert client.post("/ptt/stop").status_code == 404
    assert client.post("/ptt/cancel").status_code == 404


@pytest.mark.anyio
async def test_start_is_idempotent_while_recording(tmp_path: Path):
    _, service, recorder, _, _, _ = build_client(tmp_path)

    first = await service.start_ptt()
    second = await service.start_ptt()

    assert first.session_id == second.session_id
    assert recorder.starts == 1


@pytest.mark.anyio
async def test_stop_without_start_raises_controlled_error(tmp_path: Path):
    _, service, *_ = build_client(tmp_path)

    with pytest.raises(ActiveSessionMissing):
        await service.stop_ptt()


@pytest.mark.anyio
async def test_ptt_stop_transcribes_and_publishes_bus_event(tmp_path: Path):
    _, service, recorder, hermes, launchpad, bus = build_client(tmp_path)

    start = await service.start_ptt()
    response = await service.stop_ptt()

    assert response.session_id == start.session_id
    assert response.status == "transcribed"
    assert response.transcript == "Run the tests"
    assert recorder.stops == 1
    assert hermes.transcripts == []
    assert launchpad.sessions == []
    assert [event.type for event in bus.events] == [
        "voice.recording.started",
        "voice.recording.stopped",
        "voice.transcription.completed",
    ]


@pytest.mark.anyio
async def test_transcription_response_does_not_depend_on_launchpad_bridge(tmp_path: Path):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("resume la build")
    hermes = FakeHermes(HermesResult(response_id="resp_123", assistant_text="Build lista."))
    launchpad = FailingLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)

    await service.start_ptt()
    response = await service.stop_ptt()

    assert response.status == "transcribed"
    assert response.transcript == "resume la build"
    assert response.assistant_text is None
    assert response.detail is None


def test_bus_tts_command_invokes_tts_and_publishes_status(tmp_path: Path):
    tts = FakeTts()
    client, service, *_rest = build_client(tmp_path, tts=tts)
    bus = _rest[-1]

    response = client.get("/health")

    assert response.status_code == 200
    import anyio

    anyio.run(service.handle_bus_command, SimpleNamespace(type="voice.tts.speak", target="agent-voice-gateway", payload={"text": "Build lista."}, correlation_id="c1", session_id="s1"))
    assert tts.calls == ["Build lista."]
    assert [event.type for event in bus.events] == ["voice.tts.started", "voice.tts.completed"]


@pytest.mark.anyio
async def test_bus_ptt_commands_start_stop_and_cancel(tmp_path: Path):
    _, service, recorder, _, _, bus = build_client(tmp_path)

    await service.handle_bus_command(
        SimpleNamespace(type="voice.ptt.start", target="agent-voice-gateway", payload={}, correlation_id="c1", session_id=None)
    )
    assert service.active_recording is not None
    assert recorder.starts == 1

    await service.handle_bus_command(
        SimpleNamespace(type="voice.ptt.stop", target="agent-voice-gateway", payload={}, correlation_id="c1", session_id=None)
    )
    assert service.active_recording is None
    assert recorder.stops == 1
    assert "voice.transcription.completed" in [event.type for event in bus.events]

    await service.handle_bus_command(
        SimpleNamespace(type="voice.ptt.start", target="agent-voice-gateway", payload={}, correlation_id="c2", session_id=None)
    )
    await service.handle_bus_command(
        SimpleNamespace(type="voice.ptt.cancel", target="agent-voice-gateway", payload={}, correlation_id="c2", session_id=None)
    )
    assert recorder.cancels == 1


@pytest.mark.anyio
async def test_bus_command_for_other_target_is_ignored(tmp_path: Path):
    _, service, recorder, _, _, _ = build_client(tmp_path)

    await service.handle_bus_command(
        SimpleNamespace(type="voice.ptt.start", target="other-service", payload={}, correlation_id="c1", session_id=None)
    )

    assert recorder.starts == 0


@pytest.mark.anyio
async def test_bus_consumer_survives_temporary_consume_error(tmp_path: Path):
    _, service, _, _, _, _ = build_client(tmp_path)
    bus = FlakyCommandBus()
    bus.commands.append(
        SimpleNamespace(
            stream_id="1-0",
            envelope=SimpleNamespace(type="voice.ptt.start", target="other-service", payload={}, correlation_id="c1", session_id=None),
        )
    )
    service.agent_bus = bus
    service.config.agent_bus.poll_interval_seconds = 0.01

    task = asyncio.create_task(_consume_bus_commands(service))
    try:
        for _ in range(50):
            if bus.acks:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert bus.fail_next_consume is False
    assert bus.acks == [["1-0"]]


@pytest.mark.anyio
async def test_empty_transcript_does_not_call_hermes(tmp_path: Path):
    _, service, _, hermes, launchpad, _ = build_client(tmp_path, transcript="")

    await service.start_ptt()
    response = await service.stop_ptt()

    assert response.status == "empty_transcript"
    assert hermes.transcripts == []
    assert launchpad.sessions[-1].status == "needs_attention"


@pytest.mark.anyio
async def test_quiet_audio_does_not_call_whisper_or_hermes(tmp_path: Path):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("Gracias por ver el video.")
    hermes = FakeHermes()
    launchpad = FakeLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)

    start = await service.start_ptt()
    write_test_wav(tmp_path / f"{start.session_id}.wav", amplitude=1)
    response = await service.stop_ptt()

    assert response.status == "empty_transcript"
    assert response.detail is not None
    assert "rms=" in response.detail
    assert hermes.transcripts == []


@pytest.mark.anyio
async def test_hermes_is_not_called_by_voice_gateway(tmp_path: Path):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("enciende las luces")
    hermes = FailingHermes()
    launchpad = FakeLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)

    await service.start_ptt()
    response = await service.stop_ptt()

    assert response.status == "transcribed"
    assert response.transcript == "enciende las luces"
    assert response.detail is None
    assert launchpad.sessions == []


@pytest.mark.anyio
async def test_cancel_discards_active_recording(tmp_path: Path):
    _, service, recorder, _, _, _ = build_client(tmp_path)

    start = await service.start_ptt()
    response = await service.cancel_ptt()

    assert response.session_id == start.session_id
    assert response.status == "cancelled"
    assert recorder.cancels == 1


def test_options_callback_accepts_valid_selection(tmp_path: Path):
    result = HermesResult(
        response_id="resp_choices",
        assistant_text="Choose next action",
        options=[
            AgentOption(option_id="run_tests", label="Tests", color="green", semantic="confirm"),
            AgentOption(option_id="cancel", label="Cancel", color="red", semantic="cancel"),
        ],
    )
    client, service, _, _, launchpad, _ = build_client(tmp_path, result=result)
    service.option_sessions["resp_choices"] = SimpleNamespace(
        session_id="resp_choices",
        options={option.option_id: option for option in result.options},
        selected_option_id=None,
    )

    response = client.post(
        "/launchpad/options/resp_choices/response",
        json={"option_id": "run_tests"},
    )

    assert launchpad.sessions == []
    assert response.status_code == 200
    assert response.json() == {
        "session_id": "resp_choices",
        "option_id": "run_tests",
        "status": "selected",
    }
    assert service.option_sessions["resp_choices"].selected_option_id == "run_tests"


def test_options_selection_read_endpoint_reports_pending_selection(tmp_path: Path):
    result = HermesResult(
        response_id="resp_choices",
        assistant_text="Choose next action",
        options=[AgentOption(option_id="run_tests", label="Tests")],
    )
    client, service, *_ = build_client(tmp_path, result=result)
    service.option_sessions["resp_choices"] = SimpleNamespace(
        session_id="resp_choices",
        options={option.option_id: option for option in result.options},
        selected_option_id=None,
    )
    response = client.get("/launchpad/options/resp_choices/response")

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("path", "expected_status"),
    [
        ("/launchpad/options/missing/response", 404),
        ("/launchpad/options/resp_choices/response", 409),
    ],
)
def test_options_callback_rejects_unknown_session_or_option(
    tmp_path: Path,
    path: str,
    expected_status: int,
):
    result = HermesResult(
        response_id="resp_choices",
        assistant_text="Choose next action",
        options=[AgentOption(option_id="run_tests", label="Tests")],
    )
    client, service, *_ = build_client(tmp_path, result=result)
    if "resp_choices" in path:
        service.option_sessions["resp_choices"] = SimpleNamespace(
            session_id="resp_choices",
            options={option.option_id: option for option in result.options},
            selected_option_id=None,
        )

    response = client.post(path, json={"option_id": "missing"})

    assert response.status_code == expected_status


def test_health_reports_tts_provider_when_enabled(tmp_path: Path):
    tts = FakeTts()
    client, *_ = build_client(tmp_path, tts=tts)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["tts"] == "gemini"


def _build_service(tmp_path: Path, tts: FakeTts | None, result: HermesResult | None = None) -> GatewayService:
    config = GatewayConfig()
    if tts is not None:
        config.tts.enabled = True
    return GatewayService(
        config=config,
        recorder=FakeRecorder(tmp_path),
        transcriber=FakeTranscriber("Run the tests"),
        hermes=FakeHermes(result),
        launchpad=FakeLaunchpad(),
        tts=tts,
    )


@pytest.mark.anyio
async def test_tts_speak_invoked_with_assistant_text(tmp_path: Path):
    tts = FakeTts()
    service = _build_service(tmp_path, tts=tts)

    await service.speak_tts("Tests passed.")

    assert tts.calls == ["Tests passed."]


@pytest.mark.anyio
async def test_tts_not_invoked_when_disabled(tmp_path: Path):
    service = _build_service(tmp_path, tts=None)

    await service.start_ptt()
    await service.stop_ptt()

    assert service.tts is None
    assert service._pending_tts_tasks == []


@pytest.mark.anyio
async def test_tts_failure_does_not_break_response(tmp_path: Path):
    tts = FakeTts(error=RuntimeError("boom"))
    service = _build_service(tmp_path, tts=tts)

    await service.speak_tts("Tests passed.")
    assert tts.calls == ["Tests passed."]


@pytest.mark.anyio
async def test_tts_not_invoked_on_empty_assistant_text(tmp_path: Path):
    tts = FakeTts()
    service = _build_service(
        tmp_path,
        tts=tts,
        result=HermesResult(response_id="resp_x", assistant_text=""),
    )

    await service.start_ptt()
    # Empty assistant_text now makes Hermes raise inside the service path —
    # mimic the real assistant by skipping that branch and calling _dispatch_tts
    # directly.
    service._dispatch_tts("")
    assert service._pending_tts_tasks == []
    assert tts.calls == []


class CrashingCommandBus(FakeBus):
    """Bus whose handler always raises an unexpected error on the first command."""

    def __init__(self, command):
        super().__init__()
        self.commands.append(command)


@pytest.mark.anyio
async def test_bus_consumer_does_not_ack_on_handler_crash(tmp_path: Path):
    """A handler crash must leave the message unacked so pending-recovery can retry."""
    _, service, recorder, _, _, _ = build_client(tmp_path)
    bus = CrashingCommandBus(
        SimpleNamespace(
            stream_id="crash-0",
            envelope=SimpleNamespace(
                type="voice.ptt.start",
                target="agent-voice-gateway",
                payload={},
                correlation_id="c1",
                session_id="ptt-1",
            ),
        )
    )
    service.agent_bus = bus
    # Make the handler crash by removing the recorder so start_ptt blows up
    service.recorder = None  # type: ignore[assignment]
    service.config.agent_bus.poll_interval_seconds = 0.01

    task = asyncio.create_task(_consume_bus_commands(service))
    try:
        for _ in range(30):
            await asyncio.sleep(0.01)
            if bus.acks:
                break
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Handler raised, so the message must NOT be in any ack batch
    acked = [sid for batch in bus.acks for sid in batch]
    assert "crash-0" not in acked
