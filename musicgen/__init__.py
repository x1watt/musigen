"""Local AI music generation with MusicGen, fully offline after model download."""

from pathlib import Path

__version__ = "0.1.0"

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Local fp16 shard set produced by musicgen.convert_weights. Falls back to the
# hub id when it has not been built yet.
_local_model = _MODELS_DIR / "musicgen-medium-fp16"
DEFAULT_MODEL = str(_local_model) if (_local_model / "config.json").exists() \
    else "facebook/musicgen-medium"

DEFAULT_DURATION = 180.0
SAMPLE_RATE = 32000
MAX_CHUNK_SECONDS = 30.0