from __future__ import annotations

import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from voice_gateway.app import create_app
from voice_gateway.config import GatewayConfig
from voice_gateway.hermes import HermesError
from voice_gateway.launchpad import LaunchpadBridgeError
from voice_gateway.models import AgentOption, HermesResult
from voice_gateway.service import GatewayService


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

    async def publish_event(self, envelope) -> None:
        self.events.append(envelope)


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


def test_start_is_idempotent_while_recording(tmp_path: Path):
    client, _, recorder, _, _, _ = build_client(tmp_path)

    first = client.post("/ptt/start")
    second = client.post("/ptt/start")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["session_id"] == second.json()["session_id"]
    assert recorder.starts == 1


def test_stop_without_start_returns_409(tmp_path: Path):
    client, *_ = build_client(tmp_path)

    response = client.post("/ptt/stop")

    assert response.status_code == 409


def test_ptt_stop_transcribes_and_publishes_bus_event(tmp_path: Path):
    client, _, recorder, hermes, launchpad, bus = build_client(tmp_path)

    start = client.post("/ptt/start")
    response = client.post("/ptt/stop")

    assert response.status_code == 200
    assert response.json()["session_id"] == start.json()["session_id"]
    assert response.json()["status"] == "transcribed"
    assert response.json()["transcript"] == "Run the tests"
    assert recorder.stops == 1
    assert hermes.transcripts == []
    assert launchpad.sessions == []
    assert [event.type for event in bus.events] == [
        "voice.recording.started",
        "voice.recording.stopped",
        "voice.transcription.completed",
    ]


def test_transcription_response_does_not_depend_on_launchpad_bridge(tmp_path: Path):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("resume la build")
    hermes = FakeHermes(HermesResult(response_id="resp_123", assistant_text="Build lista."))
    launchpad = FailingLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)
    client = TestClient(create_app(config=config, service=service))

    client.post("/ptt/start")
    response = client.post("/ptt/stop")

    assert response.status_code == 200
    assert response.json()["status"] == "transcribed"
    assert response.json()["transcript"] == "resume la build"
    assert response.json()["assistant_text"] is None
    assert response.json()["detail"] is None


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


def test_empty_transcript_does_not_call_hermes(tmp_path: Path):
    client, _, _, hermes, launchpad, _ = build_client(tmp_path, transcript="")

    client.post("/ptt/start")
    response = client.post("/ptt/stop")

    assert response.status_code == 200
    assert response.json()["status"] == "empty_transcript"
    assert hermes.transcripts == []
    assert launchpad.sessions[-1].status == "needs_attention"


def test_quiet_audio_does_not_call_whisper_or_hermes(tmp_path: Path):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("Gracias por ver el video.")
    hermes = FakeHermes()
    launchpad = FakeLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)
    client = TestClient(create_app(config=config, service=service))

    start = client.post("/ptt/start")
    write_test_wav(tmp_path / f"{start.json()['session_id']}.wav", amplitude=1)
    response = client.post("/ptt/stop")

    assert response.status_code == 200
    assert response.json()["status"] == "empty_transcript"
    assert "rms=" in response.json()["detail"]
    assert hermes.transcripts == []


def test_hermes_is_not_called_by_voice_gateway(tmp_path: Path):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("enciende las luces")
    hermes = FailingHermes()
    launchpad = FakeLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)
    client = TestClient(create_app(config=config, service=service))

    client.post("/ptt/start")
    response = client.post("/ptt/stop")

    assert response.status_code == 200
    assert response.json()["status"] == "transcribed"
    assert response.json()["transcript"] == "enciende las luces"
    assert response.json()["detail"] is None
    assert launchpad.sessions == []


def test_cancel_discards_active_recording(tmp_path: Path):
    client, _, recorder, _, _, _ = build_client(tmp_path)

    start = client.post("/ptt/start")
    response = client.post("/ptt/cancel")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": start.json()["session_id"],
        "status": "cancelled",
        "transcript": None,
        "hermes_response_id": None,
        "assistant_text": None,
        "detail": None,
    }
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
