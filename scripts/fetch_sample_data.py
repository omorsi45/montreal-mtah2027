"""Fetch the public Metrica sample data at the pinned commit into sample_dir()."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from montreal.paths import SAMPLE_COMMIT, SAMPLE_REPO_URL, sample_dir  # noqa: E402

# Optional local copy used instead of the network when it exists.
LOCAL_MIRROR = Path.home() / "Desktop" / "Projects" / "mtah-external" / "metrica-sample-data"


def git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def head(target: Path) -> str:
    return git("rev-parse", "HEAD", cwd=target)


def main() -> int:
    target = sample_dir()
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source = str(LOCAL_MIRROR) if (LOCAL_MIRROR / ".git").exists() else SAMPLE_REPO_URL
        print(f"cloning {source} -> {target}")
        git("clone", "--quiet", source, str(target))
    if head(target) != SAMPLE_COMMIT:
        git("checkout", "--quiet", SAMPLE_COMMIT, cwd=target)
    sha = head(target)
    if sha != SAMPLE_COMMIT:
        print(f"MISMATCH: {sha} != pinned {SAMPLE_COMMIT}", file=sys.stderr)
        return 1
    print(sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
