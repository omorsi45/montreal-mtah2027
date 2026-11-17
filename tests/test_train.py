"""Training / LOGO tuning tests: synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from montreal import train
from montreal.tracking import Tracking

PERIODS = [(0, 4000), (5000, 9500)]


def _trk(n=10_000, slots=6, periods=PERIODS) -> Tracking:
    rng = np.random.default_rng(1)
    ball = rng.uniform(5, 95, size=(n, 2))
    players = rng.uniform(5, 95, size=(n, slots, 2))
    team = np.array([0] * (slots // 2) + [1] * (slots - slots // 2), dtype=np.int8)
    return Tracking(ball=ball, players=players, team=team, periods=list(periods),
                    pitch=(105.0, 68.0))


def _in_period(n, periods):
    m = np.zeros(n, dtype=bool)
    for a, b in periods:
        m[a:b] = True
    return m


def test_simulate_dropout_share_and_determinism():
    trk = _trk()
    n = len(trk.ball)
    inside = _in_period(n, trk.periods)
    ball_before = trk.ball.copy()
    players_before = trk.players.copy()

    out = train.simulate_dropout(trk, np.random.default_rng(7))
    miss = np.isnan(out.ball[:, 0])
    share = miss[inside].mean()
    assert 0.25 <= share <= 0.30
    assert not miss[~inside].any()  # nothing outside the periods is touched
    assert np.array_equal(np.isnan(out.ball[:, 0]), np.isnan(out.ball[:, 1]))

    pshare = np.isnan(out.players[..., 0])[inside].mean(axis=0)  # per slot
    assert (pshare >= 0.05).all() and (pshare <= 0.07).all()

    # same seed -> same mask; input not mutated
    again = train.simulate_dropout(trk, np.random.default_rng(7))
    assert np.array_equal(np.isnan(out.ball), np.isnan(again.ball))
    assert np.array_equal(np.isnan(out.players), np.isnan(again.players))
    other = train.simulate_dropout(trk, np.random.default_rng(8))
    assert not np.array_equal(np.isnan(out.ball), np.isnan(other.ball))
    assert np.array_equal(trk.ball, ball_before) and np.array_equal(trk.players, players_before)
    assert out.ball is not trk.ball and out.players is not trk.players


def test_simulate_dropout_adds_on_top_of_existing_gaps():
    trk = _trk()
    n = len(trk.ball)
    inside = _in_period(n, trk.periods)
    ball = trk.ball.copy()
    ball[(np.arange(n) // 500) % 10 < 3] = np.nan  # 30% missing already, in 500-row blocks
    trk = Tracking(ball=ball, players=trk.players, team=trk.team, periods=trk.periods,
                   pitch=trk.pitch)
    before = np.isnan(trk.ball[:, 0])
    out = train.simulate_dropout(trk, np.random.default_rng(3))
    after = np.isnan(out.ball[:, 0])
    newly = after & ~before
    assert newly[inside].sum() >= 0.25 * inside.sum() - 1  # per-period ceil
    assert (after | ~before).all()  # nothing re-appears


def test_select_tiebreak():
    table = pd.DataFrame(
        {
            "candidate": ["a", "b", "c", "d"],
            "swap": [10, 10, 20, 20],
            "dead": [5, 5, 5, 5],
            "live": [5, 5, 5, 5],
            "spell_f1_mean": [0.5, 0.6, 0.6, 0.6],
            "macro_f1_mean": [0.9, 0.5, 0.7, 0.7],
        }
    )
    row = train.select(table)
    assert row["candidate"] == "c"  # ties on spell F1 -> higher macro; then the earlier row
    assert train.select(table.iloc[[0]])["candidate"] == "a"


def _synthetic_game(seed: int):
    """Ball and players orbit; team labels switch every ~150 rows with short dead gaps."""
    rng = np.random.default_rng(seed)
    n = 1500
    t = np.arange(n)
    ang = 0.016 * t
    ball = np.stack([52.0 + 10.0 * np.cos(ang), 34.0 + 10.0 * np.sin(ang)], axis=1)
    team = np.array([0] * 4 + [1] * 4, dtype=np.int8)
    labels = np.full(n, 3, dtype=np.int8)
    periods = [(0, 700), (800, 1500)]
    seq = [0, 1, 2, 0, 1, 0, 2, 1]
    for a, b in periods:
        for i, s in enumerate(range(a, b, 90)):
            labels[s : min(s + 90, b)] = seq[(i + seed) % len(seq)]
    players = np.zeros((n, 8, 2))
    far = np.array([[8.0, 6.0], [-9.0, 5.0], [7.0, -8.0], [-6.0, -9.0]])
    near = np.array([[0.5, 0.0], [4.0, 4.0], [-5.0, 3.0], [3.0, -6.0]])
    home_has = labels == 0
    for s in range(4):
        players[:, s] = ball + np.where(home_has[:, None], near[s], far[s])
        players[:, 4 + s] = ball + np.where(home_has[:, None], far[s], near[s])
    players = players + rng.normal(0, 0.1, players.shape)
    trk = Tracking(ball=ball, players=players, team=team, periods=periods, pitch=(105.0, 68.0))
    return trk, labels


def test_cost_grid_brackets_the_optimum_in_ascending_order():
    swaps = sorted({s for s, _, _ in train.COST_GRID})
    assert swaps == [2.5, 5, 10, 20, 40, 80, 160]
    assert sorted({d for _, d, _ in train.COST_GRID}) == [2.5, 5, 10, 20, 40]
    assert sorted({v for _, _, v in train.COST_GRID}) == [2.5, 5, 10, 20, 40]
    assert len(train.COST_GRID) == 7 * 5 * 5
    assert train.COST_GRID == sorted(train.COST_GRID)  # grid order is the final tie-break


def test_num_keeps_fractional_costs():
    assert train._num(2.5) == 2.5
    assert train._num(10.0) == 10 and isinstance(train._num(10.0), int)


def test_logo_runs_on_tiny_synthetic_games(monkeypatch):
    monkeypatch.setattr(train, "COST_GRID", [(20.0, 10.0, 10.0), (40.0, 5.0, 5.0)])
    games = [_synthetic_game(k) for k in range(3)]
    table = train.logo(games)
    assert len(table) == len(train.CANDIDATES) * 2
    expected = {"candidate", "swap", "dead", "live", "spell_f1_g1", "spell_f1_g2", "spell_f1_g3",
                "spell_f1_mean", "macro_f1_mean"}
    assert expected <= set(table.columns)
    num = table[["spell_f1_g1", "spell_f1_g2", "spell_f1_g3", "spell_f1_mean", "macro_f1_mean"]]
    assert np.isfinite(num.to_numpy()).all()
    assert list(table["candidate"].unique()) == [c.name for c in train.CANDIDATES]
    again = train.logo(games)
    pd.testing.assert_frame_equal(table, again)  # deterministic


def test_feature_cache_signature_depends_on_code_version(monkeypatch):
    trk = _trk(n=300, periods=[(0, 300)])
    sig = train._signature(trk)
    assert sig == train._signature(trk)
    monkeypatch.setattr(train, "_code_version", lambda: "another version of features.py")
    assert train._signature(trk) != sig