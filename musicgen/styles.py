"""Style preset loading and prompt assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRESET_PATH = Path(__file__).resolve().parent / "presets" / "styles.json"


class StyleError(KeyError):
    """Raised when an unknown style name is requested."""


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise StyleError(f"style preset file missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StyleError(f"style preset file is not valid JSON: {exc}") from exc


def available(path: Path = PRESET_PATH) -> list[str]:
    return sorted(_load(path)["styles"].keys())


def describe(name: str, path: Path = PRESET_PATH) -> str:
    styles = _load(path)["styles"]
    if name not in styles:
        raise StyleError(f"unknown style '{name}'. available: {', '.join(sorted(styles))}")
    return styles[name]["label"]


def get(name: str, path: Path = PRESET_PATH) -> dict[str, str]:
    styles = _load(path)["styles"]
    if name not in styles:
        raise StyleError(f"unknown style '{name}'. available: {', '.join(sorted(styles))}")
    return styles[name]


def resolve(name: str, path: Path = PRESET_PATH) -> dict[str, str] | None:
    """Return a preset by name, or None if the caller passed free-form text."""
    if not name:
        return None
    styles = _load(path)["styles"]
    return styles.get(name.strip().lower())