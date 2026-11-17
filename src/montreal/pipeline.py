"""End-to-end pipeline: features -> emission (+ tag prior) -> Viterbi -> submission file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from montreal import paths
from montreal.decode import cost_matrix, decode_periods
from montreal.emission import SoftmaxRegression, heuristic_logp
from montreal.features import FEATURE_NAMES, compute_features
from montreal.tagging import (apply_tag_prior, calibrate, calibration_stats, check_orientation,
                              confident_team, goal_dead, load_tags, tag_votes)
from montreal.tracking import Tracking, load_competition

GAME = "BAR-ATH"
LETTERS = "HADX"  # code -> letter, H=0 A=1 D=2 X=3


def load_params(path) -> dict:
    """Read a params.json file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def emission_logp(F: np.ndarray, params: dict) -> np.ndarray:
    """(n, 3) log-probabilities over H/A/D from the heuristic or a stored softmax model."""
    emission = params["emission"]
    if isinstance(emission, str):
        if emission != "heuristic":
            raise ValueError(f"unknown emission {emission!r}")
        return heuristic_logp(F, FEATURE_NAMES)
    model = SoftmaxRegression.from_dict(emission)
    if model.feature_names is None:
        raise ValueError("emission model has no stored feature names")
    missing = [name for name in model.feature_names if name not in FEATURE_NAMES]
    if missing:
        raise ValueError(f"emission model needs features that are not computed: {missing}")
    if np.shape(F)[1] != len(FEATURE_NAMES):
        raise ValueError(f"F must have {len(FEATURE_NAMES)} columns (FEATURE_NAMES), got {np.shape(F)[1]}")
    # The model sees exactly the columns it was trained on, picked by name from the full matrix.
    cols = [FEATURE_NAMES.index(name) for name in model.feature_names]
    return model.log_proba(np.asarray(F)[:, cols])


def predict_codes(trk: Tracking, F: np.ndarray, params: dict, tags: dict | None = None) -> np.ndarray:
    """Decode int8 codes (H=0, A=1, D=2, X=3) for every row of the tracking.

    `tags` (optional) holds ``votes``, ``weights`` and ``goal_mask``; it is applied to the emission
    log-probabilities by `apply_tag_prior` (team-only, D untouched except the goal term).
    """
    logp = emission_logp(F, params)
    if tags is not None:
        logp = apply_tag_prior(logp, tags["votes"], tags["weights"], tags["goal_mask"])
    cost = cost_matrix(**{k: float(v) for k, v in params["costs"].items()})
    return decode_periods(logp, trk.periods, cost)


def predict_match_detailed(params: dict, use_tags: bool | None = None) -> tuple[np.ndarray, dict | None, dict]:
    """Run the competition match; returns (codes, calibrated tag weights or None, info).

    `use_tags` overrides ``params["use_tags"]`` (for label-free with/without diagnostics).
    `info` holds aggregates only: ``unknown_team_slots`` and ``calibration`` (per family raw
    agreement ``p`` and support ``n``, or None when tags are off). Raises RuntimeError if the
    attack tags disagree with the tracking (H and A flipped).
    """
    trk = load_competition()
    F = compute_features(trk)
    n = len(trk.ball)
    info = {"unknown_team_slots": int((np.asarray(trk.team) < 0).sum()), "calibration": None}
    tags_in = None
    weights = None
    if params.get("use_tags") if use_tags is None else use_tags:
        games, _ = paths.upstream()
        assets = games.asset_paths(GAME, data_dir=str(paths.data_dir()))
        tags = load_tags(assets["smart_tagging"], fps=trk.fps)
        votes = tag_votes(tags, n)
        confident = confident_team(F, FEATURE_NAMES)
        info["calibration"] = calibration_stats(votes, confident)
        check_orientation(info["calibration"])
        weights = calibrate(votes, confident)
        tags_in = {"votes": votes, "weights": weights,
                   "goal_mask": goal_dead(tags, n, fps=trk.fps)}
    return predict_codes(trk, F, params, tags=tags_in), weights, info

def predict_match(params: dict) -> np.ndarray:
    """int8 codes (199319,) for the competition match."""
    return predict_match_detailed(params)[0]


def write_submission(codes: np.ndarray, path) -> str:
    """Write `frame,label` (letters, \\n endings, ASCII); returns the SHA-256 hex digest."""
    letters = np.array(list(LETTERS))[np.asarray(codes, dtype=np.int64)]
    lines = ["frame,label"] + [f"{i},{c}" for i, c in enumerate(letters.tolist())]
    data = ("\n".join(lines) + "\n").encode("ascii")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
