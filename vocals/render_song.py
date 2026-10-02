"""Render a full song with vocals and lyrics using ACE-Step.

Separate from musicgen/ because ACE-Step pins transformers==4.50, which conflicts
with the transformers 5.x that musicgen/ needs. Run this with .venv-acestep:

    .venv-acestep/bin/python -m vocals.render_song \
        --lyrics lyrics/cyberpunk-neonlauf.txt \
        --prompt "cyberpunk techno, 145 bpm, female vocal" \
        -d 240 -o musicgen/out/neonlauf.wav

ACE-Step sings the lyrics itself, so this produces actual vocals rather than the
instrumental-only output MusicGen can make.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "models" / "ace-step-v1-3.5b"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a vocal song with ACE-Step.")
    parser.add_argument("--lyrics", type=Path, required=True,
                        help="text file with [Verse]/[Chorus]/[Bridge] tags")
    parser.add_argument("--prompt", default="cyberpunk techno, driving beat, female vocal",
                        help="style description")
    parser.add_argument("-d", "--duration", type=float, default=240.0,
                        help="audio duration in seconds (default 240)")
    parser.add_argument("-o", "--out", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--steps", type=int, default=60, help="diffusion steps")
    parser.add_argument("--guidance", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true", default=False,
                        help="keep weights in RAM and move per-module; needs lots of RAM")
    parser.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false",
                        help="keep weights resident on the GPU (default, needs ~8GB VRAM)")
    parser.add_argument("--no-tags", dest="use_tags", action="store_false", default=True,
                        help="drop [Verse]/[Chorus] structure tags from the lyrics")
    parser.add_argument("--split", action="store_true",
                        help="split long lyrics into per-section renders and stitch them, "
                             "so each pass stays inside the VRAM budget")
    return parser.parse_args(argv)


def split_sections(lyrics: str) -> list[tuple[str, str]]:
    """Split tagged lyrics into (tag, body) sections, preserving order."""
    sections: list[tuple[str, str]] = []
    current_tag: str | None = None
    current: list[str] = []

    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if current_tag is not None:
                sections.append((current_tag, "\n".join(current).strip()))
            current_tag = line
            current = []
        else:
            current.append(line)

    if current_tag is not None:
        body = "\n".join(current).strip()
        if body:
            sections.append((current_tag, body))
    return sections


def stitch_wavs(paths: list[Path], out_path: Path, sampling_rate: int) -> float:
    """Concatenate rendered sections into one song file."""
    import numpy as np
    import soundfile as sf

    pieces = []
    for path in paths:
        data, sr = sf.read(str(path), always_2d=True)
        if sr != sampling_rate:
            raise RuntimeError(f"{path.name} is {sr} Hz, expected {sampling_rate}")
        pieces.append(data)

    merged = np.concatenate(pieces, axis=0)
    peak = float(np.max(np.abs(merged))) if merged.size else 0.0
    if peak > 0.97:
        merged = merged * (0.97 / peak)
    sf.write(str(out_path), merged, sampling_rate, subtype="PCM_16")
    return merged.shape[0] / sampling_rate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    lyrics = args.lyrics.read_text(encoding="utf-8").strip()
    if not lyrics:
        raise SystemExit("lyrics file is empty")

    out_path = args.out or (ROOT / "musicgen" / "out" / "neonlauf.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from acestep.pipeline_ace_step import ACEStepPipeline

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    print(f"loading ACE-Step from {args.checkpoint}", flush=True)
    started = time.perf_counter()
    pipeline = ACEStepPipeline(
        checkpoint_dir=str(args.checkpoint),
        dtype="bfloat16",
        cpu_offload=args.cpu_offload,
    )
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)

    def render(target_lyrics: str, seconds: float, dest: Path, seed: int) -> bool:
        pipeline(
            prompt=args.prompt,
            lyrics=target_lyrics,
            audio_duration=seconds,
            infer_step=args.steps,
            guidance_scale=args.guidance,
            manual_seeds=[seed],
            use_erg_tag=args.use_tags,
            use_erg_lyric=args.use_tags,
            format="wav",
            save_path=str(dest),
        )
        return dest.exists()

    # Long lyrics do not fit in VRAM as a single pass, so render each tagged
    # section on its own and stitch the results.
    if args.split:
        sections = split_sections(lyrics)
        if not sections:
            raise SystemExit("no [Verse]/[Chorus]/[Bridge] sections found to split on")
        per_section = args.duration / len(sections)
        print(f"{len(sections)} sections at {per_section:.0f}s each", flush=True)

        rendered: list[Path] = []
        for index, (tag, body) in enumerate(sections):
            dest = out_path.parent / f"{out_path.stem}-{index:02d}.wav"
            print(f"[{index + 1}/{len(sections)}] {tag} ({per_section:.0f}s)", flush=True)
            chunk_lyrics = f"{tag}\n{body}" if args.use_tags else body
            if render(chunk_lyrics, per_section, dest, args.seed + index):
                rendered.append(dest)
            else:
                print(f"  section {tag} failed to render, skipping", flush=True)

        if not rendered:
            raise SystemExit("no sections rendered")
        total = stitch_wavs(rendered, out_path, sampling_rate=48000)
        print(f"wrote {out_path} ({total:.1f}s from {len(rendered)} sections)", flush=True)
        return 0

    print(f"rendering {args.duration:.0f}s, {args.steps} steps, seed {args.seed}", flush=True)
    started = time.perf_counter()
    result = pipeline(
        prompt=args.prompt,
        lyrics=lyrics,
        audio_duration=args.duration,
        infer_step=args.steps,
        guidance_scale=args.guidance,
        manual_seeds=[args.seed],
        use_erg_tag=args.use_tags,
        use_erg_lyric=args.use_tags,
        format="wav",
        save_path=str(out_path),
    )
    elapsed = time.perf_counter() - started

    # The pipeline saves to save_path; fall back to its returned array if not.
    produced = out_path.exists()
    if not produced and result is not None:
        import numpy as np
        import soundfile as sf

        array = np.asarray(result, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        sf.write(str(out_path), array, 48000)
        produced = True

    if produced:
        size_mb = out_path.stat().st_size / 1e6
        print(f"wrote {out_path} ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)
    else:
        print("pipeline did not produce an audio file", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())