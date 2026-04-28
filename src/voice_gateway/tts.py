from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from voice_gateway.config import TtsConfig

logger = logging.getLogger(__name__)


class TtsError(RuntimeError):
    pass


class TtsClient(Protocol):
    async def speak(self, text: str) -> None: ...


class GeminiTtsClient:
    """TTS client that uses Google's Gemini speech generation API.

    Synthesis is performed via the official ``google-genai`` SDK and audio is
    played back locally with ``sounddevice`` when ``auto_play`` is enabled.

    The SDK is synchronous; calls are dispatched to a thread via
    ``asyncio.to_thread`` so the event loop is never blocked.
    """

    def __init__(self, config: TtsConfig, client: Any | None = None):
        self.config = config
        self._client = client
        self._play_lock = asyncio.Lock()

        if client is None and not self._is_test_config():
            api_key = config.api_key()
            if not api_key:
                raise ValueError(
                    "TTS enabled but no API key found in env vars: "
                    f"{config.api_key_env}"
                )
            try:
                from google import genai  # type: ignore
            except ImportError as exc:  # pragma: no cover - import guard
                raise TtsError(
                    "google-genai is not installed. Install with "
                    "'pip install voice-gateway[tts]' or add google-genai to "
                    "your environment."
                ) from exc
            self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _is_test_config() -> bool:
        return False

    async def synthesize(self, text: str) -> bytes:
        """Synthesize ``text`` to PCM audio bytes (int16, mono, ``sample_rate``)."""
        if self._client is None:
            raise TtsError("TTS client not initialized")
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        from google.genai import types  # type: ignore

        response = self._client.models.generate_content(
            model=self.config.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.config.voice,
                        ),
                    ),
                ),
            ),
        )
        try:
            audio = response.candidates[0].content.parts[0].inline_data.data
        except (AttributeError, IndexError, TypeError) as exc:
            raise TtsError("Gemini response did not contain audio data") from exc
        if not isinstance(audio, (bytes, bytearray)):
            raise TtsError(f"Unexpected audio payload type: {type(audio)!r}")
        return bytes(audio)

    async def speak(self, text: str) -> None:
        """Synthesize and (optionally) play ``text``.

        Errors are logged as warnings and never propagated; TTS failures must
        not break the PTT pipeline.
        """
        if not text:
            return
        try:
            audio = await self.synthesize(text)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            logger.warning("TTS synthesis failed: %s", exc)
            return

        if not self.config.auto_play:
            return

        try:
            async with self._play_lock:
                await asyncio.to_thread(self._play_sync, audio)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            logger.warning("TTS playback failed: %s", exc)

    def _play_sync(self, pcm_bytes: bytes) -> None:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore

        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        sd.play(
            samples,
            samplerate=self.config.sample_rate,
            device=self.config.output_device,
        )
        sd.wait()
