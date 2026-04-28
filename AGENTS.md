# AGENTS.md — agent-voice-gateway

Local FastAPI audio gateway for the Agent Assistant system.

This repo is one child repo inside the multi-repo workspace at
`/Users/carlosledesma/projects/agent-assistant`. Read the workspace-level
`../AGENTS.md` first when working across repos.

## Current Responsibility

`agent-voice-gateway` owns:

- Push-to-talk recording.
- WAV/audio lifecycle and silence detection.
- Speech-to-text through `whisper.cpp`.
- Text-to-speech through Gemini TTS or future TTS providers.
- Audio/STT/TTS events published to `agent-bus`.
- Bus command consumption for PTT and TTS.

It intentionally does **not** call Hermes in the primary architecture. Hermes is
owned by the live companion app in `../mac-widget-hermes`.

HTTP `/ptt/*` endpoints are intentionally removed. PTT is only driven by
`agent-bus` commands:

```text
agent-bus command voice.ptt.start
  -> start recording
  -> publish voice.recording.started

agent-bus command voice.ptt.stop
  -> stop recording
  -> silence check
  -> whisper transcription
  -> publish voice.transcription.completed or voice.transcription.empty

agent-bus command voice.tts.speak
  -> synthesize/play
  -> publish voice.tts.started/completed/failed
```

## Source Map

All source lives under `src/voice_gateway/`.

- `app.py`
  - FastAPI app, health/options HTTP routes, startup/shutdown bus consumer task.
- `service.py`
  - `GatewayService`, PTT state machine, event publication, bus command handlers,
    TTS command execution.
- `bus.py`
  - Async HTTP client for `agent-bus` command consumption and event publication.
- `audio.py`
  - `FfmpegRecorder`, ffmpeg process lifecycle, timeout handling.
- `audio_analysis.py`
  - RMS/peak silence detection before Whisper.
- `transcriber.py`
  - `WhisperCliTranscriber`, external `whisper-cli` invocation.
- `tts.py`
  - `GeminiTtsClient`, optional Google Gemini speech generation and playback.
- `hermes.py`
  - Legacy/debug Hermes client. Do not use in the main PTT pipeline.
- `launchpad.py`
  - Legacy/debug Launchpad bridge client and `AgentSession` helpers.
- `config.py`
  - Pydantic config. Includes `agent_bus`.

## Config Notes

Reference config: `config.example.yaml`.

Important section:

```yaml
agent_bus:
  enabled: true
  base_url: http://127.0.0.1:8790
  group: agent-voice-gateway
  consumer: voice-gateway
  poll_interval_seconds: 0.2
```

Debug HTTP service default:

```yaml
server:
  host: 127.0.0.1
  port: 8788
```

## Bus Contract

Consumes commands from `agentbus.commands` through the bus HTTP consumer API:

- `voice.ptt.start`
- `voice.ptt.stop`
- `voice.ptt.cancel`
- `voice.tts.speak`

Publishes events:

- `voice.recording.started`
- `voice.recording.stopped`
- `voice.recording.cancelled`
- `voice.transcription.completed`
- `voice.transcription.empty`
- `voice.tts.started`
- `voice.tts.completed`
- `voice.tts.failed`

`voice.transcription.completed` payload should include at least:

```json
{
  "session_id": "ptt-...",
  "transcript": "..."
}
```

## Run And Test

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e '.[tts]'
.venv/bin/voice-gateway --config config.example.yaml
.venv/bin/python -m pytest
```

Full suite status after migration: `28 passed`.

## Pitfalls

- macOS microphone permissions are required for real recording.
- `ffmpeg_input_args` are host-specific; default uses AVFoundation.
- `whisper.binary` and `whisper.model` are literal paths. Wrong paths fail.
- TTS is best-effort. Do not let TTS failure break STT/PTT flow.
- `GatewayService.active_recording` assumes one active PTT session at a time.
- The bus consumer currently polls. A future iteration may replace it with a
  stronger long-running stream consumer.
- Keep Hermes out of the main flow; add Hermes work to `../mac-widget-hermes`.

## Sibling Repos

- `../agent-bus`: durable bus and event contract.
- `../launchpad-system-actions`: physical Launchpad controls and LEDs.
- `../mac-widget-hermes`: Hermes companion and widget snapshot UI.
