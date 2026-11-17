"""CLI: python -m montreal.run [--params models/params.json] [--out submissions/Montreal.csv].

Prints aggregates only (label counts, possession count / mean duration, tag weights, SHA-256 and
one public-window sanity line). Never prints frame labels or rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from montreal import paths
from montreal.metrics import score
from montreal.pipeline import LETTERS, load_params, predict_match_detailed, write_submission

N_FRAMES = 199319
GT_RELATIVE = Path("evaluation") / "gt" / "BAR-ATH_possession_gt_public.csv"


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(paths.repo_dir()))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    repo = paths.repo_dir()
    ap = argparse.ArgumentParser(prog="python -m montreal.run", description=__doc__)
    ap.add_argument("--params", default=str(repo / "models" / "params.json"))
    ap.add_argument("--out", default=str(repo / "submissions" / "Montreal.csv"))
    args = ap.parse_args(argv)

    params_path, out_path = Path(args.params), Path(args.out)
    params = load_params(params_path)
    codes, weights, info = predict_match_detailed(params)
    sha = write_submission(codes, out_path)

    _, pm = paths.upstream()
    frame = pm.read_label_csv(out_path)
    problem = pm.validate_submission(frame, N_FRAMES)
    if problem:
        print(f"validation FAILED: {problem}")
        return 1
    print(f"params: {_rel(params_path)}")
    print(f"wrote: {_rel(out_path)}")
    print("validation: OK")

    counts = {LETTERS[k]: int((codes == k).sum()) for k in range(4)}
    print("label counts: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    mean_s = float(np.mean([(b - a) / 25.0 for a, b, _ in spells])) if spells else 0.0
    print(f"possessions: {len(spells)}  mean duration: {mean_s:.3f} s")
    if weights is None:
        print("tag weights: not used")
    else:
        print("tag weights: " + " ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items())))
    if info["calibration"]:
        print("tag agreement p (support n): " + " ".join(
            f"{k}={s['p']:.3f} ({s['n']})" for k, s in sorted(info["calibration"].items())))
    if info["unknown_team_slots"] > 0:
        print(f"players with an unknown team id (ignored): {info['unknown_team_slots']}")
    print(f"sha256: {sha}")

    gt_path = paths.upstream_dir() / GT_RELATIVE
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    r = score(gt, codes)
    print(f"public-window sanity check only (not used for selection): "
          f"spellF1 {r['spell_f1']:.4f} macroF1 {r['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
