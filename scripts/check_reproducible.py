"""Reproducibility check: two fresh runs of `python -m montreal.run` must give identical bytes.

Each run is a separate subprocess (same interpreter as this script) writing to its own temporary
file. The script prints the two SHA-256 digests (and whether they equal the SHA of
submissions/Montreal.csv, if that file exists), deletes the temporary files and exits 1 if the two
digests differ. It prints digests only, never any frame labels.

With `--cold` every run also gets its own empty cache folder (`MONTREAL_CACHE_DIR`), so the raw
tracking is parsed again and nothing cached is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSION = REPO / "submissions" / "Montreal.csv"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child_env(cache_dir: Path | None) -> dict:
    """Environment of one run. Same as the README: the package must be installed (`pip install -e .
    --no-deps`); nothing is injected into PYTHONPATH, so a missing install fails here.
    A `cache_dir` points MONTREAL_CACHE_DIR at it (a cold run); None leaves the cache as it is."""
    env = dict(os.environ)
    if cache_dir is not None:
        env["MONTREAL_CACHE_DIR"] = str(cache_dir)
    return env


def run_once(out: Path, cache_dir: Path | None = None) -> None:
    """One fresh interpreter: python -m montreal.run --out <out> (its stdout is not echoed)."""
    proc = subprocess.run(
        [sys.executable, "-m", "montreal.run", "--out", str(out)],
        cwd=REPO, env=child_env(cache_dir), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"run failed (exit {proc.returncode}): {proc.stdout[-500:]}{proc.stderr[-500:]}")
        raise SystemExit(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cold", action="store_true",
                    help="give each run its own empty cache folder (MONTREAL_CACHE_DIR)")
    args = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="montreal_repro_"))
    try:
        digests = []
        for i in (1, 2):
            out = tmp / f"run{i}.csv"
            run_once(out, tmp / f"cache{i}" if args.cold else None)
            digests.append(sha256_of(out))
            print(f"run {i}{' (cold cache)' if args.cold else ''}: sha256 {digests[-1]}")
        same = digests[0] == digests[1]
        print(f"identical: {same}")
        if SUBMISSION.exists():
            match = sha256_of(SUBMISSION) == digests[0]
            print(f"matches submissions/Montreal.csv: {match}")
        return 0 if same else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())
