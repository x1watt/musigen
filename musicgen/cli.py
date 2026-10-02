"""Command line entry point for local AI music generation."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from . import DEFAULT_DURATION, DEFAULT_MODEL, __version__
from . import render as render_mod
from . import styles as styles_mod
from .planner import MAX_CHUNK_SECONDS, build_plan


def planner_max_chunk() -> float:
    return MAX_CHUNK_SECONDS


OUT_DIR = Path(__file__).resolve().parent / "out"


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:limit].strip("-") or "track")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musicgen",
        description="Generate a 3 minute AI music track locally from a style and prompt.",
    )
    parser.add_argument("prompt", nargs="?", default="",
                        help="what the track should be about")
    parser.add_argument("-s", "--style", default="",
                        help=f"style preset: {', '.join(styles_mod.available())} "
                             "(or free-form style text)")
    parser.add_argument("-d", "--duration", type=float, default=DEFAULT_DURATION,
                        help=f"track length in seconds (default {DEFAULT_DURATION:.0f})")
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="output wav path (default: musicgen/out/<slug>-<stamp>.wav)")
    parser.add_argument("--mp3", action="store_true",
                        help="also encode an mp3 next to the wav (needs ffmpeg)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"HF model id (default {DEFAULT_MODEL})")
    
    parser.add_argument("--guidance", type=float, default=3.0,
                        help="classifier-free guidance scale (default 3.0)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--crossfade", type=float, default=0.5,
                        help="crossfade seconds between chunks (default 0.5)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--chunk-seconds", type=float, default=None,
                        help="max seconds per model pass (default 15, lower if VRAM is tight)")
    parser.add_argument("--arc", default="default", choices=["default", "continuous"],
                        help="'continuous' removes the breakdown and quiet outro for "
                             "running/gym mixes")
    parser.add_argument("--continuity", action="store_true",
                        help="feed each chunk the tail of the previous one for smoother "
                             "seams (slower, and can cause quiet or silent chunks)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; omit for a different result each run")
    parser.add_argument("--list-styles", action="store_true",
                        help="print available style presets and exit")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the chunk plan and exit without loading the model")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"musicgen {__version__}")
    return parser


def resolve_style(name: str) -> tuple[str, str, str]:
    """Return (style_text, mix_text, arc_text) from a preset or free-form text."""
    if not name:
        return "", "", ""
    preset = styles_mod.resolve(name)
    if preset is not None:
        return preset["keywords"], preset["mix"], preset["arc"]
    return name.strip(), "", ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_styles:
        for key in styles_mod.available():
            print(f"{key:12s} {styles_mod.describe(key)}")
        return 0

    prompt = args.prompt.strip()
    if not prompt:
        print("error: a prompt is required. Example: musicgen 'rainy city at night' -s techno",
              file=sys.stderr)
        return 2

    if args.duration <= 0:
        print("error: --duration must be positive", file=sys.stderr)
        return 2

    style_text, mix_text, arc_text = resolve_style(args.style)
    max_chunk = args.chunk_seconds if args.chunk_seconds else planner_max_chunk()
    plan = build_plan(
        duration=args.duration,
        style_text=style_text,
        subject=prompt,
        mix_text=mix_text,
        arc_text=arc_text,
        crossfade=max(0.0, args.crossfade),
        max_chunk=max_chunk,
        arc=args.arc,
    )

    if not args.quiet:
        print(f"prompt : {prompt}")
        print(f"style  : {args.style or '(none)'}")
        print(f"target : {args.duration:.1f}s in {len(plan.chunks)} chunks "
              f"({plan.crossfade:.2f}s crossfades)")

    if args.plan_only:
        for chunk in plan.chunks:
            print(f"  [{chunk.index:02d}] {chunk.seconds:5.1f}s {chunk.role:10s} {chunk.text}")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = slugify(f"{args.style or 'freeform'}-{prompt}")
    out_path = args.out or (OUT_DIR / f"{stem}-{stamp}.wav")
    out_path = out_path.with_suffix(".wav")

    from .engine import MusicEngine

    if not args.quiet:
        print(f"loading : {args.model} on cuda ({args.dtype})")
    engine = MusicEngine(
        model_id=args.model,
        dtype=args.dtype,
        seed=args.seed,
    )

    if not args.quiet:
        print(f"engine  : {engine.sampling_rate} Hz, {engine.channels}ch")

    def progress(index: int, total: int, role: str) -> None:
        if args.quiet:
            return
        print(f"  chunk {index + 1}/{total} [{role}] ...", flush=True)

    audio, stats = engine.render_plan(
        plan.chunks,
        guidance_scale=args.guidance,
        temperature=args.temperature,
        top_p=args.top_p,
        crossfade=plan.crossfade,
        progress=progress,
        continuity=args.continuity,
    )

    final = render_mod.master(audio, engine.sampling_rate, args.duration)
    render_mod.write_wav(out_path, final, engine.sampling_rate)

    if not args.quiet:
        print(f"rendered: {stats.generated_seconds:.1f}s of audio in "
              f"{stats.elapsed:.1f}s ({stats.real_time_factor:.2f}x realtime)")
        if stats.retried_chunks:
            print(f"note    : {stats.retried_chunks} silent chunk(s) detected and re-rendered")
        print(f"wav     : {out_path}")

    if args.mp3:
        mp3_path = out_path.with_suffix(".mp3")
        if render_mod.write_mp3(mp3_path, out_path):
            if not args.quiet:
                print(f"mp3     : {mp3_path}")
        elif not args.quiet:
            print("mp3     : skipped (ffmpeg not found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())