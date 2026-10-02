"""MusicGen inference wrapper: loads once, renders chunks with continuation."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from . import DEFAULT_MODEL

# How much of the previous chunk to feed back in as an audio prompt, seconds.
# Off by default: conditioning on a previous chunk degrades each generation and
# can compound into silent stretches, while crossfades alone keep the track
# healthy. Enable with --continuity for smoother seams at some quality cost.
CONTEXT_SECONDS = 4.0
# Keep context short enough to stay inside the model's 30s receptive window.
_MAX_CONTEXT_SECONDS = 8.0
# Target chunk length. A 1.5B decoder's KV cache grows with sequence length, so
# shorter chunks leave more VRAM headroom on a shared GPU.
DEFAULT_CHUNK_SECONDS = 15.0
# Never split below this, otherwise an OOM retry loop cannot converge.
_MIN_CHUNK_SECONDS = 5.0
# A chunk quieter than this is treated as a silent generation failure.
DEAD_RMS = 0.02
# ...as is any one-second window inside a chunk quieter than this.
DEAD_WINDOW_RMS = 0.02
# ...as is a chunk far quieter than the running average of the track so far.
DEAD_RATIO = 0.25
# How many times to retry a chunk that came back silent.
RETRY_ATTEMPTS = 3

# MusicGen emits ~50 audio tokens per second of output at 32 kHz.
TOKENS_PER_SECOND = 50

ProgressFn = Callable[[int, int, str], None]


@dataclass
class RenderStats:
    generated_seconds: float
    elapsed: float
    real_time_factor: float
    retried_chunks: int = 0


class MusicEngine:
    """Thin wrapper around transformers MusicGen.

    Weights are loaded straight onto the GPU in fp16 so host RAM stays small.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "float16",
        seed: int | None = None,
    ) -> None:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Install a CUDA torch build.")

        self.model_id = model_id
        self.device = device or "cuda"
        self.dtype = getattr(torch, dtype)
        self.seed = seed
        self._processor_cls = AutoProcessor
        self._model_cls = MusicgenForConditionalGeneration

        # seed=None leaves the RNG unseeded, so each run differs.
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Sharded fp16 weights let transformers mmap each shard and move it
        # straight to the GPU, so the full model is never resident in host RAM.
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.decoder = self.model.decoder
        self.audio_encoder = self.model.audio_encoder
        self._retry_without_context = False

        self.sampling_rate = int(self.model.config.audio_encoder.sampling_rate)
        # The decoder's channel count drives both the continuation prompt shape and
        # the audio returned by generate(); musicgen-medium decodes to mono.
        self.channels = int(getattr(self.decoder.config, "audio_channels", 2) or 2)
        self.encoder_channels = int(self.channels)

    def _sample_rate_check(self) -> None:
        if self.sampling_rate <= 0:
            raise RuntimeError("model reports a non-positive sampling rate")

    def render_chunk(
        self,
        text: str,
        tokens: int,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        continuation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate one chunk. Returns float32 audio shaped (samples, channels)."""
        self._sample_rate_check()

        kwargs: dict[str, object] = {}
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        kwargs.update(inputs)

        context_samples = 0
        expected_samples = 0
        if continuation is not None and continuation.size > 0:
            context = self._to_context(continuation)
            if context is not None:
                # transformers>=5 renamed the audio-prompt kwarg to input_values.
                kwargs["input_values"] = context
                context_samples = int(context.shape[-1])

        kwargs.update(
            do_sample=True,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=int(tokens),
        )
        # MusicGen emits ~50 audio tokens per second at 32 kHz.
        expected_samples = int(round(tokens * self.sampling_rate / 50))

        try:
            out = self._generate(kwargs)
        except torch.OutOfMemoryError:
            # Another app may be holding VRAM, so shed memory and retry this chunk
            # without the continuation prompt (costs a little seam continuity).
            self._free_vram()
            if context_samples == 0:
                raise
            kwargs.pop("input_values", None)
            context_samples = 0
            out = self._generate(kwargs)

        # generate() returns (batch, channels, samples); use (samples, channels).
        audio = out[0].detach().float().cpu().numpy().T
        if audio.ndim == 1:
            audio = audio[:, None]

        # An audio prompt comes back prepended to the new audio, so keep only the
        # trailing window the requested token count should have produced.
        if context_samples and audio.shape[0] > expected_samples:
            audio = audio[-expected_samples:]

        audio = self._resample_if_needed(audio)
        if not np.all(np.isfinite(audio)):
            raise RuntimeError("model produced non-finite audio (NaN/Inf)")
        return audio

    def _to_context(self, audio: np.ndarray) -> torch.Tensor | None:
        """Take the tail of the previous chunk as a (batch, ch, samples) prompt."""
        tail_samples = int(CONTEXT_SECONDS * self.sampling_rate)
        tail = audio[-tail_samples:]
        if tail.shape[0] < self.sampling_rate // 2:
            return None

        # The decoder is mono, so downmix any stereo generation output first.
        if tail.shape[1] != self.channels:
            if self.channels == 1 and tail.shape[1] >= 2:
                tail = tail.mean(axis=1, keepdims=True)
            elif tail.shape[1] == 1 and self.channels >= 2:
                tail = np.repeat(tail, self.channels, axis=1)
            else:
                tail = tail[:, : self.channels]

        limit = int(_MAX_CONTEXT_SECONDS * self.sampling_rate)
        if tail.shape[0] > limit:
            tail = tail[-limit:]

        tensor = torch.from_numpy(np.ascontiguousarray(tail.T)).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.dtype)

    def render_seconds(
        self,
        text: str,
        seconds: float,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        continuation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render `seconds` of audio, halving the request if VRAM runs out."""
        tokens = max(50, int(round(seconds * TOKENS_PER_SECOND)))
        try:
            return self.render_chunk(
                text=text,
                tokens=tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                continuation=continuation,
            )
        except torch.OutOfMemoryError:
            self._free_vram()
            if seconds <= _MIN_CHUNK_SECONDS:
                raise
            half = seconds / 2.0
            first = self.render_seconds(
                text, half, guidance_scale, temperature, top_p, continuation
            )
            second = self.render_seconds(
                text, half, guidance_scale, temperature, top_p, first
            )
            return np.concatenate([first, second], axis=0)

    def _generate(self, kwargs: dict[str, object]):
        with torch.inference_mode():
            return self.model.generate(**kwargs)

    def _free_vram(self) -> None:
        """Drop cached blocks so a retry has the best chance of fitting."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _resample_if_needed(self, audio: np.ndarray) -> np.ndarray:
        """Model output rate matches, so this is a no-op guard for custom models."""
        return audio

    def _rms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))

    def _is_dead(self, audio: np.ndarray, reference: float | None) -> bool:
        """Detect a chunk that contains a silent stretch.

        MusicGen sometimes emits audio that decays into near-silence partway
        through a chunk. Whole-chunk RMS hides that, so this checks the quietest
        one-second window as well as the overall level.
        """
        level = self._rms(audio)
        if level < DEAD_RMS:
            return True

        sr = self.sampling_rate
        if audio.shape[0] >= 2 * sr:
            windows = audio[: (audio.shape[0] // sr) * sr].reshape(-1, sr)
            levels = np.array([self._rms(w) for w in windows])
            # Ignore the first and last second: fades and tails are expected there.
            interior = levels[1:-1]
            if interior.size and float(interior.min()) < DEAD_WINDOW_RMS:
                return True

        if reference is not None and reference > DEAD_RMS and level < reference * DEAD_RATIO:
            return True
        return False

    def _render_with_recovery(
        self,
        text: str,
        seconds: float,
        continuation: np.ndarray | None,
        guidance_scale: float,
        temperature: float,
        top_p: float,
        reference: float | None,
        attempts: int = RETRY_ATTEMPTS,
    ) -> tuple[np.ndarray, bool]:
        """Render one chunk, retrying with a fresh seed if it comes back silent.

        Returns the audio and whether any retry was needed.
        """
        last_error: Exception | None = None
        retried = False
        for attempt in range(attempts):
            try:
                audio = self.render_seconds(
                    text=text,
                    seconds=seconds,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    top_p=top_p,
                    # A silent prompt can seed more silence, so drop it on retries.
                    continuation=continuation if attempt == 0 else None,
                )
            except torch.OutOfMemoryError as exc:
                last_error = exc
                self._free_vram()
                retried = True
                continue

            if not self._is_dead(audio, reference):
                return audio, retried

            last_error = RuntimeError(f"silent chunk (rms={self._rms(audio):.6f})")
            self._free_vram()
            retried = True

        raise RuntimeError(
            f"failed to render a non-silent {seconds:.1f}s chunk after "
            f"{attempts} attempts"
        ) from last_error

    def render_plan(
        self,
        chunks,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        top_p: float = 0.95,
        crossfade: float = 0.5,
        progress: ProgressFn | None = None,
        continuity: bool = False,
    ) -> tuple[np.ndarray, RenderStats]:
        """Render every chunk in order, stitching with equal-power crossfades."""
        total = len(chunks)
        pieces: list[np.ndarray] = []
        started = time.perf_counter()
        total_seconds = 0.0
        retry_note = None

        for chunk in chunks:
            if progress:
                progress(chunk.index, total, chunk.role)

            continuation = None
            if continuity and pieces:
                continuation = _tail_of(pieces, CONTEXT_SECONDS, self.sampling_rate)

            # Reference level from chunks rendered so far, used to spot a chunk
            # that collapses far below the track's running loudness.
            reference = None
            if len(pieces) >= 3:
                levels = [self._rms(p) for p in pieces]
                reference = float(np.median(levels))

            audio, retried = self._render_with_recovery(
                text=chunk.text,
                seconds=chunk.seconds,
                continuation=continuation,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                reference=reference,
            )
            if retried:
                retry_note = (retry_note or 0) + 1
                if progress:
                    progress(chunk.index, total, f"{chunk.role} (retried)")

            pieces.append(audio)
            total_seconds += audio.shape[0] / self.sampling_rate

        merged = stitch(pieces, self.sampling_rate, crossfade)
        elapsed = time.perf_counter() - started
        rtf = (total_seconds / elapsed) if elapsed > 0 else 0.0
        return merged, RenderStats(
            generated_seconds=total_seconds,
            elapsed=elapsed,
            real_time_factor=rtf,
            retried_chunks=retry_note or 0,
        )


def _tail_of(pieces: list[np.ndarray], seconds: float, sampling_rate: int) -> np.ndarray:
    """Concatenate just enough of the tail of all rendered pieces to feed back."""
    tail_samples = int(seconds * sampling_rate)
    merged: np.ndarray | None = None
    for piece in reversed(pieces):
        merged = piece if merged is None else np.concatenate([piece, merged], axis=0)
        if merged.shape[0] >= tail_samples:
            break
    assert merged is not None
    return merged[-tail_samples:]


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, samples: int) -> np.ndarray:
    """Overlap-add two arrays over `samples` frames with equal-power ramps."""
    samples = min(samples, a.shape[0], b.shape[0])
    if samples <= 0:
        return np.concatenate([a, b], axis=0)

    fade_in = np.sin(np.linspace(0.0, np.pi / 2, samples, dtype=np.float32))[:, None]
    fade_out = np.cos(np.linspace(0.0, np.pi / 2, samples, dtype=np.float32))[:, None]
    blended = a[-samples:] * fade_out + b[:samples] * fade_in
    return np.concatenate([a[:-samples], blended, b[samples:]], axis=0)


def stitch(pieces: list[np.ndarray], sampling_rate: int, crossfade: float) -> np.ndarray:
    """Stitch chunks into one continuous buffer using equal-power crossfades."""
    if not pieces:
        raise ValueError("no audio pieces to stitch")
    if len(pieces) == 1:
        return pieces[0]

    channels = max(p.shape[1] for p in pieces)
    prepared = []
    for piece in pieces:
        if piece.shape[1] < channels:
            pad = np.repeat(piece[:, -1:], channels - piece.shape[1], axis=1)
            piece = np.concatenate([piece, pad], axis=1)
        prepared.append(piece)

    overlap = int(round(crossfade * sampling_rate))
    merged = prepared[0]
    for nxt in prepared[1:]:
        merged = equal_power_crossfade(merged, nxt, overlap)
    return merged