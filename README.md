# music-generator

Generates full-length music tracks locally from a prompt plus a style. Two
engines, both running on this machine's GPU with no cloud calls:

- **MusicGen** (`musicgen/`) - instrumental tracks. Fast, chunked, no vocals.
- **ACE-Step** (`vocals/`) - songs **with sung lyrics**, including German.

## Requirements

- NVIDIA GPU. MusicGen needs ~5.3GB VRAM; ACE-Step needs ~7.5GB.
- Python 3.10+
- Weights already downloaded (see Setup)

## Setup

### MusicGen (instrumental)

The venv already exists. To rebuild from scratch:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights live in `models/`. `facebook/musicgen-medium` only publishes an
8GB fp32 `pytorch_model.bin`, which cannot be loaded on this machine's RAM
budget. Convert it once into fp16 safetensors shards that map straight to the GPU:

```bash
.venv/bin/python -m musicgen.convert_weights
```

### ACE-Step (vocals)

ACE-Step pins `transformers==4.50`, which conflicts with the 5.x that MusicGen
needs, so it lives in its own venv. Its code is installed from the GitHub repo
(the PyPI `ACE-Step` package is a broken stub):

```bash
python3 -m venv .venv-acestep
.venv-acestep/bin/pip install torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu124
.venv-acestep/bin/pip install "diffusers==0.32.2" "transformers==4.50.0" \
    "accelerate==1.6.0" peft einops loguru "py3langid==0.3.0" \
    "librosa==0.11.0" "soundfile==0.13.1" click huggingface_hub \
    sentencepiece protobuf tqdm torchvision spacy hangul-romanize cutlet
git clone --depth 1 https://github.com/ace-step/ACE-Step.git /tmp/opencode/ACE-Step
.venv-acestep/bin/pip install -e /tmp/opencode/ACE-Step --no-deps
```

Then fetch the 7.8GB of weights into `models/ace-step-v1-3.5b/`. The Python
downloader gets OOM-killed on this machine, so stream them with curl:

```bash
B=https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main
D=models/ace-step-v1-3.5b
for f in config.json ace_step_transformer/config.json \
         music_dcae_f8c8/config.json music_vocoder/config.json \
         umt5-base/config.json umt5-base/special_tokens_map.json \
         umt5-base/tokenizer.json umt5-base/tokenizer_config.json; do
  mkdir -p "$D/$(dirname "$f")"; curl -sL -o "$D/$f" "$B/$f"
done
curl -sL -o "$D/umt5-base/model.safetensors"                        "$B/umt5-base/model.safetensors"
curl -sL -o "$D/music_dcae_f8c8/diffusion_pytorch_model.safetensors"  "$B/music_dcae_f8c8/diffusion_pytorch_model.safetensors"
curl -sL -o "$D/music_vocoder/diffusion_pytorch_model.safetensors"   "$B/music_vocoder/diffusion_pytorch_model.safetensors"
curl -sL -o "$D/ace_step_transformer/diffusion_pytorch_model.safetensors" "$B/ace_step_transformer/diffusion_pytorch_model.safetensors"
```

### Environment

```bash
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Usage

### Instrumental (MusicGen)

```bash
.venv/bin/python -m musicgen.cli "rainy city at night" -s techno --arc continuous --mp3
```

| flag | meaning |
| --- | --- |
| `prompt` | what the track is about (required) |
| `-s, --style` | a preset name, or free-form style text |
| `-d, --duration` | target seconds (default 180) |
| `--arc` | `continuous` drops the breakdown and quiet outro (good for running) |
| `--continuity` | feed each chunk the previous tail for smoother seams (off by default) |
| `--chunk-seconds` | seconds per model pass (default 15, lower if VRAM is tight) |
| `--seed` | random seed; omit for a different result each run |
| `--guidance` | classifier-free guidance, default 3.0 |
| `-o, --out`, `--mp3` | output path, mp3 encode |
| `--plan-only`, `--list-styles` | inspect without rendering |

Styles: `techno`, `house`, `dnb`, `ambient`, `lofi`, `trap`, `cinematic`,
`punk`, `bossa`, `synthwave`. Any other `-s` value passes through as free text.

### Vocals (ACE-Step)

```bash
.venv-acestep/bin/python -m vocals.render_song \
    --lyrics lyrics/cyberpunk-neonlauf.txt \
    --prompt "cyberpunk techno, 145 bpm driving beat, energetic female vocal" \
    -d 165 --steps 60 --seed 21 --split \
    -o musicgen/out/neonlauf-vocal.wav
```

| flag | meaning |
| --- | --- |
| `--lyrics` | text file with `[Verse]`/`[Chorus]`/`[Bridge]` tags |
| `--prompt` | style description |
| `-d, --duration` | total seconds (default 240) |
| `--split` | render each tagged section separately and stitch (see VRAM note) |
| `--steps` | diffusion steps (default 60) |
| `--seed`, `--guidance` | seed and guidance scale |
| `--no-cpu-offload` | keep weights on GPU (default; `--cpu-offload` needs lots of RAM) |

## How it works

### MusicGen

MusicGen generates 30s per pass, so a 3 minute track needs many passes.

1. **Plan** - `musicgen/planner.py` allocates seconds per section (intro, build,
   drop, groove, breakdown, climax, outro), gives each section its own sonically
   worded prompt, then cuts those into chunks. `--arc continuous` swaps in an
   arc with no breakdown and no quiet outro.
2. **Generate** - `musicgen/engine.py` renders each chunk and crossfades them
   equal-power. Every chunk is checked for silent stretches and re-rendered if
   it comes back dead.
3. **Master** - `musicgen/render.py` trims or pads to the exact duration,
   applies cosine fades, peak-normalizes, and writes 32kHz 16-bit PCM.

Style presets in `musicgen/presets/styles.json` expand a short name into
instrumentation, production, and arc keywords folded into every prompt.

### ACE-Step

Takes a style prompt plus raw lyrics and generates a complete song with sung
vocals in one diffusion pass. Because a single pass only fits ~45-60s in VRAM,
`--split` renders each tagged section separately and concatenates them.

## Hardware notes

- MusicGen output is **mono** 32kHz: musicgen-medium's decoder is mono
  (`audio_channels=1`). ACE-Step output is stereo 48kHz.
- **Do not use `--continuity`.** Audio-prompt continuation degrades each chunk
  and compounds into multi-second silent stretches; crossfades alone are safer.
- Silent-chunk detection checks one-second windows, not whole-chunk RMS, because
  MusicGen often emits audio that decays to near-silence partway through a chunk.
- ACE-Step needs ~7.5GB VRAM and cannot run a whole song in one pass here; 90s
  OOMs while 45s works. `--split` keeps each pass inside that ceiling.
- Rendering runs slower than realtime (MusicGen ~0.72x), so 3 minutes takes
  roughly 4 minutes of wall clock. ACE-Step is faster per second of audio.
- Host RAM is tight. Keep the 8GB fp32 `pytorch_model.bin` only if you intend to
  re-convert; the fp16 shards are what actually gets loaded.