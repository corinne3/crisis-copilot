"""Configuration loading (config/default.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(*parts: str) -> Path:
    """Absolute path from the repository root, e.g. project_path('data', 'clean')."""
    return ROOT.joinpath(*parts)
