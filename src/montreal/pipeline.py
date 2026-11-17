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
from montreal.tagging import (apply_tag_prior, calibrate, confident_team, goal_dead, load_tags,
                              tag_votes)
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
    if model.feature_names is None or list(model.feature_names) != list(FEATURE_NAMES):
        raise ValueError("emission model feature names do not match FEATURE_NAMES")
    return model.log_proba(F)


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


def predict_match_detailed(params: dict, use_tags: bool | None = None) -> tuple[np.ndarray, dict | None]:
    """Run the competition match; returns (codes, calibrated tag weights or None).

    `use_tags` overrides ``params["use_tags"]`` (for label-free with/without diagnostics).
    """
    trk = load_competition()
    F = compute_features(trk)
    n = len(trk.ball)
    tags_in = None
    weights = None
    if params.get("use_tags") if use_tags is None else use_tags:
        games, _ = paths.upstream()
        assets = games.asset_paths(GAME, data_dir=str(paths.data_dir()))
        tags = load_tags(assets["smart_tagging"], fps=trk.fps)
        votes = tag_votes(tags, n)
        weights = calibrate(votes, confident_team(F, FEATURE_NAMES))
        tags_in = {"votes": votes, "weights": weights,
                   "goal_mask": goal_dead(tags, n, fps=trk.fps)}
    return predict_codes(trk, F, params, tags=tags_in), weights

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
