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


def build_client(tmp_path: Path, transcript: str = "Run the tests", result: HermesResult | None = None):
    config = GatewayConfig()
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber(transcript)
    hermes = FakeHermes(result)
    launchpad = FakeLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)
    return TestClient(create_app(config=config, service=service)), service, recorder, hermes, launchpad


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
    client, _, recorder, _, _ = build_client(tmp_path)

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


def test_ptt_stop_transcribes_calls_hermes_and_publishes_completed_session(tmp_path: Path):
    client, _, recorder, hermes, launchpad = build_client(tmp_path)

    start = client.post("/ptt/start")
    response = client.post("/ptt/stop")

    assert response.status_code == 200
    assert response.json()["session_id"] == start.json()["session_id"]
    assert response.json()["status"] == "completed"
    assert response.json()["transcript"] == "Run the tests"
    assert response.json()["hermes_response_id"] == "resp_123"
    assert recorder.stops == 1
    assert hermes.transcripts == ["Run the tests"]
    assert launchpad.sessions[-1].status == "completed"
    assert launchpad.sessions[-1].summary == "Tests passed."


def test_launchpad_publish_error_keeps_json_response_with_transcript(tmp_path: Path):
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
    assert response.json()["status"] == "completed"
    assert response.json()["transcript"] == "resume la build"
    assert response.json()["assistant_text"] == "Build lista."
    assert "Launchpad publish failed" in response.json()["detail"]


def test_completed_ack_option_publishes_awaiting_choice_for_physical_ack(tmp_path: Path):
    config = GatewayConfig()
    config.launchpad_bridge.completed_ack_option = True
    recorder = FakeRecorder(tmp_path)
    transcriber = FakeTranscriber("resume la build")
    hermes = FakeHermes(HermesResult(response_id="resp_123", assistant_text="Build lista."))
    launchpad = FakeLaunchpad()
    service = GatewayService(config, recorder, transcriber, hermes, launchpad)
    client = TestClient(create_app(config=config, service=service))

    client.post("/ptt/start")
    response = client.post("/ptt/stop")
    callback = client.post("/launchpad/options/resp_123/response", json={"option_id": "ack"})
    selected = client.get("/launchpad/options/resp_123/response")

    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_choice"
    assert launchpad.sessions[-1].status == "awaiting_choice"
    assert launchpad.sessions[-1].options[0].option_id == "ack"
    assert callback.status_code == 200
    assert callback.json()["status"] == "selected"
    assert selected.status_code == 200
    assert selected.json()["option_id"] == "ack"


def test_empty_transcript_does_not_call_hermes(tmp_path: Path):
    client, _, _, hermes, launchpad = build_client(tmp_path, transcript="")

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


def test_hermes_error_preserves_transcript_in_response(tmp_path: Path):
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
    assert response.json()["status"] == "hermes_error"
    assert response.json()["transcript"] == "enciende las luces"
    assert response.json()["detail"] == "Hermes unavailable"
    assert launchpad.sessions[-1].status == "error"


def test_cancel_discards_active_recording(tmp_path: Path):
    client, _, recorder, _, _ = build_client(tmp_path)

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
    client, service, _, _, launchpad = build_client(tmp_path, result=result)

    client.post("/ptt/start")
    stop = client.post("/ptt/stop")
    response = client.post(
        "/launchpad/options/resp_choices/response",
        json={"option_id": "run_tests"},
    )

    assert stop.json()["status"] == "awaiting_choice"
    assert launchpad.sessions[-1].status == "awaiting_choice"
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
    client, *_ = build_client(tmp_path, result=result)

    client.post("/ptt/start")
    client.post("/ptt/stop")
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
    client, *_ = build_client(tmp_path, result=result)
    client.post("/ptt/start")
    client.post("/ptt/stop")

    response = client.post(path, json={"option_id": "missing"})

    assert response.status_code == expected_status
