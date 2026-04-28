from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioStats:
    duration_seconds: float
    rms: float
    peak: float

    @property
    def summary(self) -> str:
        return (
            f"audio duration={self.duration_seconds:.2f}s "
            f"rms={self.rms:.4f} peak={self.peak:.4f}"
        )


def analyze_wav(path: Path) -> AudioStats:
    with wave.open(str(path), "rb") as wav:
        frame_count = wav.getnframes()
        frame_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(frame_count)

    if frame_count == 0 or frame_rate == 0 or not frames:
        return AudioStats(duration_seconds=0.0, rms=0.0, peak=0.0)

    samples = list(_pcm_samples(frames, sample_width))
    if not samples:
        return AudioStats(duration_seconds=frame_count / frame_rate, rms=0.0, peak=0.0)

    max_possible = float((1 << (8 * sample_width - 1)) - 1)
    rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5 / max_possible
    peak = max(abs(sample) for sample in samples) / max_possible
    return AudioStats(duration_seconds=frame_count / frame_rate, rms=rms, peak=peak)


def is_too_quiet(stats: AudioStats, min_rms: float, min_peak: float) -> bool:
    return stats.rms < min_rms and stats.peak < min_peak


def _pcm_samples(frames: bytes, sample_width: int):
    if sample_width == 1:
        for value in frames:
            yield value - 128
        return

    if sample_width == 2:
        for index in range(0, len(frames) - 1, 2):
            yield int.from_bytes(frames[index : index + 2], "little", signed=True)
        return

    if sample_width == 3:
        for index in range(0, len(frames) - 2, 3):
            raw = frames[index : index + 3]
            sign_extend = b"\xff" if raw[2] & 0x80 else b"\x00"
            yield int.from_bytes(raw + sign_extend, "little", signed=True)
        return

    if sample_width == 4:
        for index in range(0, len(frames) - 3, 4):
            yield int.from_bytes(frames[index : index + 4], "little", signed=True)
