from __future__ import annotations

import time
from pathlib import Path
import struct
import wave

import pytest

from voice_gateway.audio import (
    FfmpegRecorder,
    RecordingError,
    RecordingHandle,
    _is_successful_ffmpeg_stop,
)
from voice_gateway.audio_analysis import analyze_wav, is_too_quiet
from voice_gateway.config import AudioConfig


class FakePipe:
    def __init__(self, contents: str = ""):
        self.contents = contents
        self.written = ""
        self.closed = False

    def write(self, value: str) -> None:
        self.written += value

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def read(self) -> str:
        return self.contents


class FakeProcess:
    def __init__(self, returncode: int | None = 0, stderr: str = ""):
        self.returncode = returncode
        self.stdin = FakePipe()
        self.stderr = FakePipe(stderr)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_ffmpeg_signal_15_exit_is_success_when_stderr_says_normal():
    stderr = "size=231KiB\nExiting normally, received signal 15."

    assert _is_successful_ffmpeg_stop(255, stderr) is True


def test_recorder_stop_accepts_ffmpeg_normal_signal_exit(tmp_path: Path):
    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"0" * 256)
    process = FakeProcess(255, "Exiting normally, received signal 15.")
    handle = RecordingHandle("ptt-test", audio_path, process, time.monotonic())
    recorder = FfmpegRecorder(AudioConfig(temp_dir=tmp_path))

    recorder.stop(handle)


def test_recorder_stop_rejects_empty_wav_even_when_process_exits_cleanly(tmp_path: Path):
    audio_path = tmp_path / "empty.wav"
    audio_path.write_bytes(b"0" * 44)
    process = FakeProcess(0, "")
    handle = RecordingHandle("ptt-test", audio_path, process, time.monotonic())
    recorder = FfmpegRecorder(AudioConfig(temp_dir=tmp_path))

    with pytest.raises(RecordingError, match="empty WAV"):
        recorder.stop(handle)


def test_analyze_wav_detects_quiet_and_loud_audio(tmp_path: Path):
    quiet_path = tmp_path / "quiet.wav"
    loud_path = tmp_path / "loud.wav"
    write_test_wav(quiet_path, amplitude=1)
    write_test_wav(loud_path, amplitude=8000)

    quiet = analyze_wav(quiet_path)
    loud = analyze_wav(loud_path)

    assert is_too_quiet(quiet, min_rms=0.003, min_peak=0.02) is True
    assert is_too_quiet(loud, min_rms=0.003, min_peak=0.02) is False
    assert loud.duration_seconds > 0


def write_test_wav(path: Path, amplitude: int, sample_rate: int = 16000) -> None:
    samples = [amplitude if index % 2 == 0 else -amplitude for index in range(sample_rate // 4)]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
