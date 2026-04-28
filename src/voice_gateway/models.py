from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool
    audio_input: str
    whisper_model: str
    hermes_url: str
    tts: str


class PttResponse(BaseModel):
    session_id: str
    status: str
    transcript: str | None = None
    hermes_response_id: str | None = None
    assistant_text: str | None = None
    detail: str | None = None


class OptionSelectionRequest(BaseModel):
    option_id: str


class OptionSelectionResponse(BaseModel):
    session_id: str
    option_id: str
    status: str


class AgentAlert(BaseModel):
    severity: str = "info"
    mode: str = "pulse"
    color: str = "yellow"
    target: str = "global"


class AgentOption(BaseModel):
    option_id: str
    label: str
    color: str = "blue"
    semantic: str = "custom"
    position: str | None = None
    enabled: bool = True


class AgentSession(BaseModel):
    session_id: str
    agent_id: str = "hermes-agent"
    status: str
    summary: str
    alert: AgentAlert
    options: list[AgentOption] = Field(default_factory=list)


class HermesResult(BaseModel):
    response_id: str | None = None
    assistant_text: str
    options: list[AgentOption] = Field(default_factory=list)
