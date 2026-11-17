"""Scoring through the official upstream metric (never reimplemented here)."""

from __future__ import annotations

import numpy as np

from montreal import paths


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Official scores for int8 code arrays (H=0, A=1, D=2, X=3).

    Ruling R1: `spell_f1` is upstream's `spell_f1_mean` (upstream's own `spell_f1` key is a
    per-IoU dict).
    """
    _, pm = paths.upstream()
    r = pm.evaluate(np.asarray(gt), np.asarray(pred))
    return {
        "spell_f1": float(r["spell_f1_mean"]),
        "macro_f1": float(r["macro_f1"]),
        "n_pred_spells": int(r["n_pred_spells"]),
        "n_gt_spells": int(r["n_gt_spells"]),
    }
