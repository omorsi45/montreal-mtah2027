"""check_reproducible.py: the child process environment (no subprocess is started)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_reproducible.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_reproducible", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_child_env_never_injects_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    mod = _load()
    env = mod.child_env(None)
    assert "PYTHONPATH" not in env


def test_cold_runs_get_their_own_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MONTREAL_CACHE_DIR", "should-be-replaced")
    mod = _load()
    a, b = tmp_path / "c1", tmp_path / "c2"
    assert mod.child_env(a)["MONTREAL_CACHE_DIR"] == str(a)
    assert mod.child_env(b)["MONTREAL_CACHE_DIR"] == str(b)
    assert mod.child_env(None)["MONTREAL_CACHE_DIR"] == "should-be-replaced"  # warm: untouched