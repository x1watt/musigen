"""Mastering pass: trim, fade, normalize, encode."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

FADE_IN_SECONDS = 2.0
FADE_OUT_SECONDS = 3.0
PEAK_CEILING = 0.97


def fit_duration(audio: np.ndarray, sampling_rate: int, duration: float) -> np.ndarray:
    """Trim or pad to exactly `duration` seconds."""
    target = int(round(duration * sampling_rate))
    have = audio.shape[0]
    if have == target:
        return audio
    if have > target:
        return audio[:target]
    pad = np.repeat(audio[-1:], target - have, axis=0)
    return np.concatenate([audio, pad], axis=0)


def apply_fades(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Cosine fade in and out to avoid clicks at the boundaries."""
    out = audio.astype(np.float32, copy=True)
    n = out.shape[0]

    fade_in = min(int(FADE_IN_SECONDS * sampling_rate), n // 4)
    if fade_in > 1:
        ramp = np.sin(np.linspace(0.0, np.pi / 2, fade_in, dtype=np.float32))[:, None]
        out[:fade_in] *= ramp

    fade_out = min(int(FADE_OUT_SECONDS * sampling_rate), n // 4)
    if fade_out > 1:
        ramp = np.cos(np.linspace(0.0, np.pi / 2, fade_out, dtype=np.float32))[:, None]
        out[-fade_out:] *= ramp
    return out


def normalize(audio: np.ndarray, ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Peak normalize to a target ceiling. Skipped on silence."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-6:
        return audio
    return (audio * (ceiling / peak)).astype(np.float32)


def master(audio: np.ndarray, sampling_rate: int, duration: float) -> np.ndarray:
    """Full chain: exact length, fades, then peak normalize."""
    out = fit_duration(audio, sampling_rate, duration)
    out = apply_fades(out, sampling_rate)
    return normalize(out)


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and convert to int16."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray, sampling_rate: int) -> None:
    """Write a 16-bit PCM wav without needing extra dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = audio.shape[1] if audio.ndim == 2 else 1
    data = to_pcm16(audio.reshape(-1) if audio.ndim == 1 else audio)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes(data.tobytes())


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def write_mp3(path: Path, wav_path: Path, bitrate: str = "192k") -> bool:
    """Encode to mp3 via ffmpeg if present. Returns False if unavailable."""
    if not _ffmpeg_available():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True