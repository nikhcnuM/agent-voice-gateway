from __future__ import annotations

from typing import Any

import httpx

from voice_gateway.config import HermesConfig
from voice_gateway.models import AgentOption, HermesResult


class HermesError(RuntimeError):
    pass


PROMPT_TEMPLATE = """The following text was transcribed from push-to-talk audio. It may contain ASR errors.
User transcript:
{transcript}"""


class HermesClient:
    def __init__(self, config: HermesConfig, http_client: httpx.AsyncClient | None = None):
        self.config = config
        self._client = http_client

    async def ask(self, transcript: str) -> HermesResult:
        payload = {
            "model": self.config.model,
            "input": PROMPT_TEMPLATE.format(transcript=transcript),
            "conversation": self.config.conversation,
            "store": True,
        }
        headers = {}
        api_key = self.config.api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = await self._post("/responses", payload, headers)
        data = response.json()
        assistant_text = extract_output_text(data)
        if not assistant_text:
            raise HermesError("Hermes response did not contain output_text")
        return HermesResult(
            response_id=data.get("id"),
            assistant_text=assistant_text,
            options=extract_options(data),
        )

    async def _post(self, path: str, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        client = self._client or httpx.AsyncClient(timeout=60)
        close_client = self._client is None
        try:
            url = f"{self.config.base_url.rstrip('/')}{path}"
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise HermesError(str(exc)) from exc
        finally:
            if close_client:
                await client.aclose()


def extract_output_text(data: Any) -> str:
    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "output_text" and isinstance(value.get("text"), str):
                texts.append(value["text"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return "\n".join(texts).strip()


def extract_options(data: dict[str, Any]) -> list[AgentOption]:
    raw_options = data.get("voice_gateway_options")
    if not isinstance(raw_options, list):
        return []

    options: list[AgentOption] = []
    for raw_option in raw_options:
        if isinstance(raw_option, dict):
            options.append(AgentOption.model_validate(raw_option))
    return options
