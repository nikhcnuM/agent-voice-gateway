from __future__ import annotations

import httpx
import pytest

from voice_gateway.config import GatewayConfig
from voice_gateway.hermes import HermesClient, PROMPT_TEMPLATE, extract_output_text
from voice_gateway.launchpad import LaunchpadBridgeClient, completed_session
from voice_gateway.models import HermesResult
from voice_gateway.transcriber import WhisperCliTranscriber, parse_whisper_output


def test_parse_whisper_output_strips_timestamps_and_metadata():
    output = """
whisper_init_from_file_with_params_no_state: loading model
[00:00:00.000 --> 00:00:01.000] Run the tests
[00:00:01.000 --> 00:00:02.000] and summarize results.
"""

    assert parse_whisper_output(output) == "Run the tests and summarize results."


def test_whisper_transcriber_includes_configured_extra_args(monkeypatch, tmp_path):
    seen = {}
    config = GatewayConfig()
    config.whisper.binary = "whisper-cli"
    config.whisper.model = "model.bin"
    config.whisper.extra_args = ["-ng"]

    def fake_run(command, capture_output, text, check):
        seen["command"] = command
        return __import__("subprocess").CompletedProcess(command, 0, stdout="Hola", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    transcript = WhisperCliTranscriber(config.whisper).transcribe(tmp_path / "audio.wav")

    assert transcript == "Hola"
    assert seen["command"] == [
        "whisper-cli",
        "-m",
        "model.bin",
        "-f",
        str(tmp_path / "audio.wav"),
        "-l",
        "es",
        "-t",
        "4",
        "-ng",
    ]


def test_extract_output_text_finds_nested_responses_content():
    data = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": "First line."},
                    {"type": "other", "text": "ignored"},
                ]
            },
            {"content": [{"type": "output_text", "text": "Second line."}]},
        ]
    }

    assert extract_output_text(data) == "First line.\nSecond line."


@pytest.mark.anyio
async def test_hermes_client_posts_expected_payload(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["payload"] = dict(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "output": [{"content": [{"type": "output_text", "text": "Done."}]}],
            },
        )

    monkeypatch.setenv("HERMES_API_KEY", "secret")
    config = GatewayConfig()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await HermesClient(config.hermes, http_client).ask("Run the tests")

    assert result.response_id == "resp_123"
    assert result.assistant_text == "Done."
    assert seen["url"] == "http://127.0.0.1:8642/v1/responses"
    assert seen["auth"] == "Bearer secret"
    assert seen["payload"] == {
        "model": "hermes-agent",
        "input": PROMPT_TEMPLATE.format(transcript="Run the tests"),
        "conversation": "launchpad-voice",
        "store": True,
    }


@pytest.mark.anyio
async def test_launchpad_bridge_publishes_agent_session():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["payload"] = dict(__import__("json").loads(request.content))
        return httpx.Response(204)

    config = GatewayConfig()
    session = completed_session(HermesResult(response_id="resp_123", assistant_text="Done."))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        await LaunchpadBridgeClient(config.launchpad_bridge, http_client).publish(session)

    assert seen["method"] == "PUT"
    assert seen["url"] == "http://127.0.0.1:8765/agent/session"
    assert seen["payload"]["session_id"] == "resp_123"
    assert seen["payload"]["status"] == "completed"
    assert seen["payload"]["summary"] == "Done."
