"""Turn a style + prompt + target duration into an ordered list of timed chunks.

MusicGen conditions on short natural-language descriptions, so each chunk gets
its own sonically-worded prompt rather than reusing one string six times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# A 1.5B decoder's KV cache grows with sequence length, so short chunks keep
# peak VRAM low enough to share the GPU with other apps.
MAX_CHUNK_SECONDS = 15.0

# weight, role key, sonically worded prompt fragment
_ARC: tuple[tuple[float, str, str], ...] = (
    (0.10, "intro", "sparse atmospheric opening, elements fading in one at a time"),
    (0.12, "build", "gradual build, layers stacking, filter opening, rising energy"),
    (0.20, "drop", "full power main section, all instruments at maximum, driving"),
    (0.15, "groove", "steady hypnotic groove, locked in, repeating main riff"),
    (0.10, "breakdown", "half time breakdown, drums drop out, sparse and emotional"),
    (0.20, "climax", "huge climax, biggest arrangement, everything together, triumphant"),
    (0.13, "outro", "gentle outro, elements dropping away, long decaying tail"),
)

# Arc for continuous-energy material such as running mixes: no breakdown and no
# quiet outro, so the track never loses drive once it is up.
_ARC_CONTINUOUS: tuple[tuple[float, str, str], ...] = (
    (0.06, "intro", "immediate driving pulse, sparse but already moving"),
    (0.12, "build", "gradual build, layers stacking, filter opening, rising energy"),
    (0.26, "drop", "full power main section, all instruments at maximum, driving"),
    (0.20, "groove", "steady hypnotic groove, locked in, repeating main riff"),
    (0.24, "climax", "maximum energy, relentless, driving hard throughout"),
    (0.12, "outro", "energy maintained to the end, driving, no fade out"),
)

ARCS: dict[str, tuple[tuple[float, str, str], ...]] = {
    "default": _ARC,
    "continuous": _ARC_CONTINUOUS,
}


@dataclass
class Section:
    role: str
    seconds: float
    text: str


@dataclass
class Chunk:
    index: int
    seconds: float
    role: str
    text: str
    sections: list[Section] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        # MusicGen emits ~50 audio tokens per second of output.
        return max(50, int(round(self.seconds * 50)))


@dataclass
class Plan:
    chunks: list[Chunk]
    crossfade: float
    duration: float

    @property
    def generated_seconds(self) -> float:
        return sum(c.seconds for c in self.chunks)


def _allocate(duration: float, weights: Sequence[float]) -> list[float]:
    """Split duration across roles, whole seconds, so the total matches exactly."""
    total = sum(weights)
    raw = [duration * w / total for w in weights]
    floors = [math.floor(v) for v in raw]
    short = int(round(duration)) - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:short]:
        floors[i] += 1
    return [float(v) for v in floors]


def _slice(seconds: float, limit: float = MAX_CHUNK_SECONDS) -> list[float]:
    """Split one section length into <=limit pieces, remainder last."""
    if seconds <= limit:
        return [seconds]
    pieces = [limit] * int(seconds // limit)
    rest = seconds - limit * len(pieces)
    if rest > 0.05:
        pieces.append(rest)
    return pieces


def _build_chunks(
    duration: float,
    arc: tuple[tuple[float, str, str], ...],
    subject: str,
    style_text: str,
    mix_text: str,
    max_chunk: float,
) -> list[Chunk]:
    """Lay out sections for `duration` and cut each one into model chunks."""
    weights = [w for w, _, _ in arc]
    lengths = _allocate(duration, weights)
    sections = [
        Section(role=role, seconds=length, text=fragment)
        for length, (_, role, fragment) in zip(lengths, arc)
    ]

    chunks: list[Chunk] = []
    for section in sections:
        for piece in _slice(section.seconds, max_chunk):
            text = ", ".join(p for p in (subject, style_text, section.text, mix_text) if p)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    seconds=piece,
                    role=section.role,
                    text=text,
                    sections=[section],
                )
            )
    return chunks


def build_plan(
    duration: float,
    style_text: str,
    subject: str,
    mix_text: str = "",
    arc_text: str = "",
    crossfade: float = 0.5,
    max_chunk: float = MAX_CHUNK_SECONDS,
    arc: str = "default",
) -> Plan:
    """Build the chunk plan for a full track."""
    if arc not in ARCS:
        raise ValueError(f"unknown arc '{arc}'. available: {', '.join(sorted(ARCS))}")
    arc_spec = ARCS[arc]
    chunks = _build_chunks(duration, arc_spec, subject, style_text, mix_text, max_chunk)

    # Crossfade overlaps shorten the timeline, so budget for the overlap we will
    # actually apply. Iterate until the stitched length covers the target.
    target = duration
    for _ in range(4):
        overlap = crossfade * (len(chunks) - 1)
        budget = target + overlap
        if sum(c.seconds for c in chunks) >= budget - 1e-6:
            return Plan(chunks=chunks, crossfade=crossfade, duration=target)
        chunks = _build_chunks(budget, arc_spec, subject, style_text, mix_text, max_chunk)

    return Plan(chunks=chunks, crossfade=crossfade, duration=target)