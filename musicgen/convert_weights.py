"""One-time: convert the fp32 pytorch_model.bin into fp16 safetensors shards.

Why this exists: facebook/musicgen-medium only ships an 8.0GB fp32 .bin file.
On this machine host RAM is nearly full, so torch.load of that file gets
OOM-killed. Loading it with mmap=True keeps RAM flat, and we write fp16 shards
so transformers can later map them straight onto the GPU without ever holding
a full fp32 copy.

Run once:  python -m musicgen.convert_weights
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

DEFAULT_SRC = Path(__file__).resolve().parent.parent / "models" / "musicgen-medium"
DEFAULT_DST = Path(__file__).resolve().parent.parent / "models" / "musicgen-medium-fp16"
SHARD_BYTES = 512 * 1024 * 1024


def _index_file(weight_map: dict[str, str], dtype: str) -> str:
    lines = ["{"]
    lines.append('  "metadata": {"total_size": 0},')
    items = list(weight_map.items())
    lines.append('  "weight_map": {')
    body = [
        f'    "{name}": "{shard}"' for name, shard in items
    ]
    lines.append(",\n".join(body))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def convert(src: Path, dst: Path, limit_gb: float | None = None) -> int:
    bin_path = src / "pytorch_model.bin"
    if not bin_path.exists():
        raise SystemExit(f"source weights not found: {bin_path}")

    dst.mkdir(parents=True, exist_ok=True)

    print(f"mapping {bin_path} ({bin_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    try:
        state = torch.load(bin_path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        print("mmap load unsupported here, falling back to full load", file=sys.stderr)
        state = torch.load(bin_path, map_location="cpu", weights_only=True)

    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    print(f"{len(tensors)} tensors mapped", flush=True)

    shard_index = 0
    shard_name = f"model-{shard_index + 1:05d}.safetensors"
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal shard_tensors, shard_bytes, shard_index, shard_name
        if not shard_tensors:
            return
        save_file(shard_tensors, str(dst / shard_name), metadata={"format": "pt"})
        for key in shard_tensors:
            weight_map[key] = shard_name
        print(f"  wrote {shard_name} ({shard_bytes / 1e6:.0f} MB, {len(shard_tensors)} tensors)",
              flush=True)
        shard_tensors = {}
        shard_bytes = 0
        shard_index += 1
        shard_name = f"model-{shard_index + 1:05d}.safetensors"
        gc.collect()

    for key in sorted(tensors):
        tensor = tensors[key]
        half = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
        half = half.contiguous().clone()
        size = half.numel() * half.element_size()

        if size > SHARD_BYTES:
            flush()
            single = f"model-{shard_index + 1:05d}.safetensors"
            save_file({key: half}, str(dst / single), metadata={"format": "pt"})
            weight_map[key] = single
            print(f"  wrote {single} (single tensor {key}, {size / 1e6:.0f} MB)", flush=True)
            shard_index += 1
            shard_name = f"model-{shard_index + 1:05d}.safetensors"
            del half
            gc.collect()
            continue

        if shard_bytes + size > SHARD_BYTES and shard_tensors:
            flush()
        shard_tensors[key] = half
        shard_bytes += size

    flush()

    (dst / "model.safetensors.index.json").write_text(
        _index_file(weight_map, "float16"), encoding="utf-8"
    )

    for name in (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        candidate = src / name
        if candidate.exists():
            target = dst / name
            if not target.exists():
                target.write_bytes(candidate.read_bytes())

    print(f"done -> {dst}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert musicgen-medium to fp16 shards")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args(argv)
    return convert(args.src, args.dst)


if __name__ == "__main__":
    raise SystemExit(main())