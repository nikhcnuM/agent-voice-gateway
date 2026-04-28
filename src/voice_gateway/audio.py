from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from voice_gateway.config import AudioConfig


class RecordingError(RuntimeError):
    pass


class RecordingTimeout(RuntimeError):
    pass


@dataclass
class RecordingHandle:
    session_id: str
    audio_path: Path
    process: subprocess.Popen
    started_at: float

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at


class FfmpegRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config

    def start(self, session_id: str) -> RecordingHandle:
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self.config.temp_dir / f"{session_id}.wav"
        command = [
            self.config.ffmpeg_binary,
            *self.config.ffmpeg_input_args,
            "-t",
            str(self.config.max_duration_seconds),
            "-ar",
            str(self.config.sample_rate),
            "-ac",
            str(self.config.channels),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(audio_path),
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return RecordingHandle(
            session_id=session_id,
            audio_path=audio_path,
            process=process,
            started_at=time.monotonic(),
        )

    def stop(self, handle: RecordingHandle) -> None:
        if handle.elapsed_seconds > self.config.max_duration_seconds:
            self._terminate(handle)
            raise RecordingTimeout("Recording exceeded maximum duration")

        stderr = self._terminate(handle)
        if not handle.audio_path.exists():
            raise RecordingError("ffmpeg did not create a WAV file")
        if handle.audio_path.stat().st_size <= 44:
            raise RecordingError("ffmpeg created an empty WAV file")
        if not _is_successful_ffmpeg_stop(handle.process.returncode, stderr):
            raise RecordingError(_last_stderr_line(stderr) or "ffmpeg recording failed")

    def cancel(self, handle: RecordingHandle) -> None:
        self._terminate(handle)
        handle.audio_path.unlink(missing_ok=True)

    def _terminate(self, handle: RecordingHandle) -> str:
        if handle.process.poll() is None:
            _request_ffmpeg_quit(handle.process)
            try:
                handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.process.terminate()
                try:
                    handle.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    handle.process.kill()
                    handle.process.wait(timeout=5)

        return handle.process.stderr.read() if handle.process.stderr else ""


def _request_ffmpeg_quit(process: subprocess.Popen) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.write("q\n")
        process.stdin.flush()
        process.stdin.close()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _is_successful_ffmpeg_stop(returncode: int | None, stderr: str) -> bool:
    if returncode in (0, -15, -2, None):
        return True
    return returncode == 255 and "Exiting normally, received signal 15" in stderr


def _last_stderr_line(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else ""
