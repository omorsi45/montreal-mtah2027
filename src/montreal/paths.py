"""Filesystem locations and access to the official upstream scorer (imported, never copied)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

SAMPLE_REPO_URL = "https://github.com/metrica-sports/sample-data"
SAMPLE_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"


def repo_dir() -> Path:
    """Root of this repository (src/montreal/paths.py -> parents[2])."""
    return Path(__file__).resolve().parents[2]


def upstream_dir() -> Path:
    """Upstream challenge repo: env MTAH_REPO_DIR, default <repo>/../more-than-a-hack-2027."""
    env = os.environ.get("MTAH_REPO_DIR")
    if env:
        return Path(env)
    return repo_dir().parent / "more-than-a-hack-2027"


def data_dir() -> Path:
    """Competition data: env MTAH_DATA_DIR, default <upstream>/data."""
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        return Path(env)
    return upstream_dir() / "data"


def sample_dir() -> Path:
    """Public Metrica sample data: env MONTREAL_SAMPLE_DIR, default <repo>/external/metrica-sample-data."""
    env = os.environ.get("MONTREAL_SAMPLE_DIR")
    if env:
        return Path(env)
    return repo_dir() / "external" / "metrica-sample-data"


def cache_dir() -> Path:
    """<repo>/.cache, created on demand."""
    path = repo_dir() / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def upstream() -> tuple[ModuleType, ModuleType]:
    """Return (games, possession_metrics) imported from the upstream repo.

    The upstream root and its evaluation/ folder are put on sys.path once.
    """
    root = upstream_dir()
    for entry in (root, root / "evaluation"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    games = importlib.import_module("games")
    pm = importlib.import_module("possession_metrics")
    return games, pm
