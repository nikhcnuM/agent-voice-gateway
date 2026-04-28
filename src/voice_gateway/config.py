from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8788


class AudioConfig(BaseModel):
    input_device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    temp_dir: Path = Path("/tmp/launchpad-voice")
    max_duration_seconds: int = 60
    min_rms: float = 0.003
    min_peak: float = 0.02
    keep_temp_files: bool = False
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_input_args: list[str] = Field(
        default_factory=lambda: ["-f", "avfoundation", "-i", ":default"]
    )


class WhisperConfig(BaseModel):
    binary: str = "whisper-cli"
    model: str = "ggml-base.en.bin"
    language: str | None = "es"
    threads: int | None = 4
    extra_args: list[str] = Field(default_factory=list)


class HermesConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8642/v1"
    model: str = "hermes-agent"
    api_key_env: list[str] = Field(default_factory=lambda: ["HERMES_API_KEY", "API_SERVER_KEY"])
    conversation: str = "launchpad-voice"

    def api_key(self) -> str | None:
        for env_name in self.api_key_env:
            value = os.getenv(env_name)
            if value:
                return value
        return None


class LaunchpadBridgeConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8765"
    publish_alerts: bool = True
    completed_ack_option: bool = False


class TtsConfig(BaseModel):
    enabled: bool = False


class GatewayConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    hermes: HermesConfig = Field(default_factory=HermesConfig)
    launchpad_bridge: LaunchpadBridgeConfig = Field(default_factory=LaunchpadBridgeConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)


def load_config(path: str | Path | None = None) -> GatewayConfig:
    config_path = Path(path or os.getenv("VOICE_GATEWAY_CONFIG", "config.yaml"))
    if not config_path.exists():
        return GatewayConfig()

    data: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    return GatewayConfig.model_validate(data)
