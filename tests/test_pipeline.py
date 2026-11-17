"""Pipeline tests: synthetic tracking only (the slow test needs the real data and skips without it)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from montreal import paths
from montreal.emission import SoftmaxRegression
from montreal.features import FEATURE_NAMES, compute_features
from montreal.pipeline import (emission_logp, predict_codes, predict_match, write_submission)
from montreal.tracking import Tracking

HEURISTIC = {
    "emission": "heuristic",
    "costs": {"swap": 20, "dead": 10, "live": 10},
    "use_tags": True,
    "meta": {"source": "test"},
}


def _trk(ball, players, team, periods) -> Tracking:
    return Tracking(
        ball=np.asarray(ball, dtype=np.float64),
        players=np.asarray(players, dtype=np.float64),
        team=np.asarray(team, dtype=np.int8),
        periods=list(periods),
        pitch=(105.0, 68.0),
    )


def test_write_submission_bytes(tmp_path):
    path = tmp_path / "sub" / "out.csv"
    sha = write_submission(np.array([3, 0, 1, 2], dtype=np.int8), path)
    data = path.read_bytes()
    assert data == b"frame,label\n0,X\n1,H\n2,A\n3,D\n"
    assert sha == hashlib.sha256(data).hexdigest()


def _moving_match():
    """Two periods with a gap; home is on the ball for 300 rows of period 1, then away."""
    n = 1000
    t = np.arange(n)
    ang = 0.016 * t  # ~4 m/s on a 10 m circle, so nobody is "slow"
    ball = np.stack([52.0 + 10.0 * np.cos(ang), 34.0 + 10.0 * np.sin(ang)], axis=1)
    team = np.array([0] * 4 + [1] * 4, dtype=np.int8)
    players = np.zeros((n, 8, 2))
    far = np.array([[8.0, 6.0], [-9.0, 5.0], [7.0, -8.0], [-6.0, -9.0]])
    near = np.array([[0.5, 0.0], [4.0, 4.0], [-5.0, 3.0], [3.0, -6.0]])
    home_has = t < 300  # rows 0-299 home, 300-599 away, period 2 (700-999) away
    for s in range(4):
        players[:, s] = ball + np.where(home_has[:, None], near[s], far[s])
        players[:, 4 + s] = ball + np.where(home_has[:, None], far[s], near[s])
    return _trk(ball, players, team, [(0, 600), (700, 1000)])


def test_predict_codes_synthetic_end_to_end():
    trk = _moving_match()
    F = compute_features(trk)
    codes = predict_codes(trk, F, HEURISTIC)
    assert codes.dtype == np.int8 and codes.shape == (1000,)
    assert (codes[600:700] == 3).all()
    assert not (codes[:600] == 3).any() and not (codes[700:] == 3).any()
    # the gap is X, so no possession can span it
    # home first, away after; one H->A switch within 25 rows of row 300
    first = codes[:600]
    changes = np.flatnonzero((first[1:] != first[:-1]) & (first[1:] != 2) & (first[:-1] != 2))
    assert first[10] == 0 and first[590] == 1
    switch = np.flatnonzero((first[:-1] == 0) & (first[1:] == 1))
    assert len(switch) == 1 and abs(int(switch[0]) + 1 - 300) <= 25
    assert len(changes) == 1
    assert (codes[710:990] == 1).mean() > 0.9


def test_static_players_no_ball_is_dead():
    n = 500
    rng = np.random.default_rng(3)
    players = np.broadcast_to(rng.uniform(10, 90, (1, 8, 2)), (n, 8, 2)).copy()
    ball = np.full((n, 2), np.nan)
    trk = _trk(ball, players, [0] * 4 + [1] * 4, [(0, n)])
    codes = predict_codes(trk, compute_features(trk), HEURISTIC)
    assert (codes == 2).mean() > 0.8


def test_predict_codes_applies_goal_term_to_dead(monkeypatch):
    import montreal.pipeline as pl

    trk = _moving_match()
    F = compute_features(trk)
    n = len(F)
    tags = {"votes": {}, "weights": {}, "goal_mask": np.ones(n, dtype=bool)}
    orig = pl.apply_tag_prior
    monkeypatch.setattr(pl, "apply_tag_prior",
                        lambda lp, v, w, g, **k: orig(lp, v, w, g, w_goal=100.0))
    codes = predict_codes(trk, F, HEURISTIC, tags=tags)
    assert (codes[:600] == 2).all() and (codes[600:700] == 3).all()


def test_votes_alone_leave_dead_frames_unchanged_in_a_clear_match():
    """With no goal mask the votes only move H against A; here D frames are the same."""
    trk = _moving_match()
    F = compute_features(trk)
    n = len(F)
    base = predict_codes(trk, F, HEURISTIC)
    votes = {k: np.ones(n, dtype=np.int8) for k in ("attack", "setpiece")}
    tags = {"votes": votes, "weights": {"attack": 1.0, "setpiece": 1.0},
            "goal_mask": np.zeros(n, dtype=bool)}
    with_tags = predict_codes(trk, F, HEURISTIC, tags=tags)
    np.testing.assert_array_equal(base == 2, with_tags == 2)

def test_emission_logp_softmax_dict_and_name_check():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, 60)
    model = SoftmaxRegression().fit(F, y, feature_names=FEATURE_NAMES)
    params = dict(HEURISTIC, emission=model.to_dict())
    assert np.allclose(emission_logp(F, params), model.log_proba(F))
    bad = model.to_dict()
    bad["feature_names"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError):
        emission_logp(F, dict(HEURISTIC, emission=bad))
    none = model.to_dict()
    none["feature_names"] = None
    with pytest.raises(ValueError):
        emission_logp(F, dict(HEURISTIC, emission=none))


@pytest.mark.slow
def test_run_twice_identical():
    try:
        games, _ = paths.upstream()
        assets = games.asset_paths("BAR-ATH", data_dir=str(paths.data_dir()))
        present = all(Path(str(assets[k])).exists() for k in ("atd_meta", "atd_raw", "smart_tagging"))
    except Exception:
        present = False
    if not present:
        pytest.skip("competition data not present")
    params = dict(HEURISTIC)
    a = predict_match(params)
    b = predict_match(params)
    assert a.shape == (199319,)
    assert np.array_equal(a, b)


def test_committed_params_load_through_emission_logp_and_match_feature_names():
    import json
    from montreal.pipeline import load_params
    params = load_params(paths.repo_dir() / "models" / "params.json")
    assert isinstance(params["emission"], dict)
    assert params["emission"]["feature_names"] == FEATURE_NAMES
    F = np.random.default_rng(5).normal(size=(40, len(FEATURE_NAMES)))
    logp = emission_logp(F, params)
    assert logp.shape == (40, 3) and np.isfinite(logp).all()
    np.testing.assert_allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-9)
    assert set(params["costs"]) == {"swap", "dead", "live"}