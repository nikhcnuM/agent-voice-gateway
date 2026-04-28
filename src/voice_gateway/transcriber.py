from __future__ import annotations

import re
import subprocess
from pathlib import Path

from voice_gateway.config import WhisperConfig


class TranscriptionError(RuntimeError):
    pass


TIMESTAMP_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


class WhisperCliTranscriber:
    def __init__(self, config: WhisperConfig):
        self.config = config

    def transcribe(self, audio_path: Path) -> str:
        command = [self.config.binary, "-m", self.config.model, "-f", str(audio_path)]
        if self.config.language:
            command.extend(["-l", self.config.language])
        if self.config.threads:
            command.extend(["-t", str(self.config.threads)])
        command.extend(self.config.extra_args)

        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise TranscriptionError(completed.stderr.strip() or "whisper-cli failed")

        transcript = parse_whisper_output(completed.stdout or completed.stderr)
        if not transcript:
            return ""
        return transcript


def parse_whisper_output(output: str) -> str:
    lines: list[str] = []
    for raw_line in output.splitlines():
        line = TIMESTAMP_RE.sub("", raw_line).strip()
        if not line:
            continue
        if line.startswith("whisper_") or line.startswith("system_info:"):
            continue
        lines.append(line)
    return " ".join(lines).strip()
