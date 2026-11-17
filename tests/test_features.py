"""Per-frame features: synthetic tracking only (no competition data)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from montreal.features import (FEATURE_NAMES, FEATURE_SETS, V1_NAMES, V2_NAMES, compute_features)
from montreal.tracking import Tracking

N = 100
PERIODS = [(0, 40), (50, 90)]
COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _trk(ball, players, team, periods=PERIODS) -> Tracking:
    return Tracking(
        ball=np.asarray(ball, dtype=np.float64),
        players=np.asarray(players, dtype=np.float64),
        team=np.asarray(team, dtype=np.int8),
        periods=list(periods),
        pitch=(105.0, 68.0),
    )


def _feat(trk: Tracking) -> dict[str, np.ndarray]:
    """Compute features with every warning turned into an error (output must be pristine)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        F = compute_features(trk)
    assert F.shape == (len(trk.ball), len(FEATURE_NAMES))
    assert np.isfinite(F).all()
    return {name: F[:, i] for name, i in COL.items()}


def _empty(slots: int = 4):
    ball = np.full((N, 2), np.nan)
    players = np.full((N, slots, 2), np.nan)
    return ball, players


def test_names_and_shape():
    assert len(V1_NAMES) == 41 and FEATURE_NAMES[:41] == V1_NAMES  # v1 names keep name and order
    assert len(FEATURE_NAMES) == 51 and len(set(FEATURE_NAMES)) == 51
    assert FEATURE_NAMES[41:] == V2_NAMES
    assert list(FEATURE_SETS) == ["v1", "v2"]
    assert FEATURE_SETS["v1"] == V1_NAMES and FEATURE_SETS["v2"] == FEATURE_NAMES
    assert FEATURE_NAMES[0] == "ball_vis" and FEATURE_NAMES[15] == "vel_away"
    assert FEATURE_NAMES[16:23] == [
        "last_diff", "next_diff", "since_seen", "until_seen", "last_out", "last_edge", "next_edge",
    ]
    assert FEATURE_NAMES[23:27] == ["d_diff_w12", "d_diff_w50", "ball_out_w12", "ball_out_w50"]
    assert V1_NAMES[-2:] == ["ball_vis_w12", "ball_vis_w50"]
    assert V2_NAMES == [
        "near_run_home", "near_run_away", "gap_flip", "ball_still", "restart_spot", "ply_accel",
        "ball_still_w50", "restart_spot_w50", "near_run_home_w12", "near_run_away_w12",
    ]
    rng = np.random.default_rng(1)
    ball = rng.uniform(0, 100, (N, 2))
    players = rng.uniform(0, 100, (N, 8, 2))
    players[rng.random((N, 8)) < 0.3] = np.nan
    ball[rng.random(N) < 0.2] = np.nan
    f = _feat(_trk(ball, players, [0] * 4 + [1] * 4))
    assert all(len(v) == N for v in f.values())


def test_ball_proximity():
    ball, players = _empty()
    ball[:] = (50.0, 30.0)
    players[:, 0] = (50.0, 30.0)  # home on the ball
    players[:, 2] = (60.0, 30.0)  # away 10 m away
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    for r in (0, 10, 39, 50, 89):
        assert f["ball_vis"][r] == 1
        assert f["d_home"][r] == pytest.approx(0)
        assert f["d_away"][r] == pytest.approx(10)
        assert f["d_diff"][r] == pytest.approx(10)
        assert f["n_home_3m"][r] == 1
        assert f["n_away_3m"][r] == 0


def test_review_focus_empty_frames():
    ball, players = _empty()
    ball[:] = (50.0, 30.0)
    players[:, 0] = (50.0, 30.0)
    players[:, 2] = (60.0, 30.0)
    ball[40:50] = np.nan  # halftime gap: no ball, no players
    players[40:50] = np.nan
    ball[10:15] = np.nan  # all-NaN rows inside a period
    players[10:15] = np.nan
    ball[60:70] = np.nan
    players[60:70] = np.nan
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    allf = np.stack(list(f.values()), axis=1)
    outside = np.r_[40:50, 90:100]
    assert (allf[outside] == 0).all()
    assert f["ball_vis"][12] == 0 and f["d_home"][12] == 30 and f["d_away"][12] == 30
    assert f["d_diff"][12] == 0 and f["ball_edge"][12] == 10
    assert f["ply_speed"][12] == 0 and f["frac_slow"][12] == 1 and f["block_prog"][12] == 0
    assert f["since_seen"][12] == pytest.approx(3 / 25) and f["until_seen"][12] == pytest.approx(3 / 25)
    assert f["since_seen"][65] == pytest.approx(6 / 25)


def test_review_focus_tracklets():
    ball = np.full((N, 2), 50.0)
    players = np.full((N, 46, 2), np.nan)
    team = np.array([0] * 23 + [1] * 23)
    players[:, 0] = (52.0, 30.0)
    players[:, 1] = (45.0, 30.0)
    players[:, 30] = (49.0, 30.0)
    players[10] = np.nan
    players[10, 0] = (52.0, 30.0)  # only a home player on this frame
    f = _feat(_trk(ball, players, team))
    assert f["d_away"][10] == 30
    assert f["d_home"][10] == pytest.approx(np.hypot(2, 20))
    assert f["d_diff"][10] == pytest.approx(30 - np.hypot(2, 20))
    assert f["spread_x_diff"][10] == 0 and f["vel_away"][10] == 0
    # more than 22 players tracked on every frame stays finite
    rng = np.random.default_rng(2)
    crowd = rng.uniform(0, 100, (N, 46, 2))
    _feat(_trk(ball, crowd, team))


def test_review_focus_ball_far_out():
    ball, players = _empty()
    ball[:] = (107.1, 87.4)
    players[:, 0] = (50.0, 30.0)
    players[:, 2] = (60.0, 30.0)
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    assert f["ball_out"][5] == 1 and f["ball_edge"][5] == 0
    assert f["d_home"][5] == 30 and f["last_out"][5] == 1 and f["ball_out_w12"][5] == 1


def test_ball_edge_and_out_thresholds():
    ball, players = _empty()
    ball[:] = (3.0, 30.0)
    ball[1] = (-0.4, 30.0)  # just outside the line but inside the 0.5 m margin
    ball[2] = (-0.6, 30.0)
    ball[3] = (50.0, 34.0)
    ball[4] = (50.0, 66.0)
    ball[5] = (50.0, 30.0)
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    assert f["ball_edge"][0] == pytest.approx(3.0)
    assert f["ball_out"][1] == 0 and f["ball_edge"][1] == 0
    assert f["ball_out"][2] == 1 and f["ball_edge"][2] == 0
    assert f["ball_edge"][3] == 10  # clipped
    assert f["ball_edge"][4] == pytest.approx(2.0)


def test_carry_does_not_cross_period():
    ball, players = _empty()
    ball[35] = (50.0, 30.0)
    players[35, 0] = (50.0, 30.0)
    players[35, 2] = (60.0, 30.0)
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    assert f["d_diff"][35] == pytest.approx(10)
    assert (f["last_diff"][36:40] == 10).all()
    assert (f["last_diff"][50:60] == 0).all()
    assert (f["next_diff"][0:35] == 10).all()
    assert (f["next_diff"][36:40] == 0).all()
    assert f["since_seen"][35] == 0 and f["since_seen"][38] == pytest.approx(3 / 25)
    assert f["since_seen"][50] == 30 and f["until_seen"][36] == 30
    assert f["until_seen"][30] == pytest.approx(5 / 25)
    assert f["last_edge"][38] == pytest.approx(10) and f["last_edge"][50] == 10


def test_carry_clips_to_30_seconds():
    n = 2000
    ball = np.full((n, 2), np.nan)
    ball[0] = (50.0, 30.0)
    players = np.full((n, 4, 2), np.nan)
    t = _trk(ball, players, [0, 0, 1, 1], periods=[(0, n)])
    f = _feat(t)
    assert f["since_seen"][1999] == 30 and f["since_seen"][100] == pytest.approx(4.0)


def test_attack_direction_flip():
    ball, players = _empty(slots=4)
    team = [0, 0, 1, 1]
    t = np.arange(N)[:, None] / 25.0  # 1 m/s towards +x
    base1 = {0: 30.0, 2: 80.0}  # period 1: home at 30 m, away mirrored at 80 m
    base2 = {0: 80.0, 2: 30.0}  # period 2: home at 80 m, away at 30 m
    for lo, hi, base in ((0, 40, base1), (50, 90, base2)):
        for s in range(4):
            x0 = base[0] if s < 2 else base[2]
            players[lo:hi, s, 0] = x0 + (s % 2) * 3.0 + t[lo:hi, 0]
            players[lo:hi, s, 1] = 20.0 + 10.0 * s
    f = _feat(_trk(ball, players, team))
    assert (f["vel_home"][5:35] > 0.9).all() and (f["vel_home"][5:35] < 1.1).all()
    assert (f["vel_home"][55:85] < -0.9).all()
    assert (f["vel_away"][5:35] < -0.9).all()
    assert (f["vel_away"][55:85] > 0.9).all()
    assert (f["vel_home"][0:40] > 0).all() and (f["vel_home"][50:90] < 0).all()  # also at the edges
    # block_prog is mirrored in period 2 (home attacks -x): (L - cx) / L instead of cx / L
    assert f["block_prog"][20] == pytest.approx((56.5 + 20 / 25) / 105.0)
    assert f["block_prog"][70] == pytest.approx((105.0 - (56.5 + 70 / 25)) / 105.0)


def test_player_speed_and_slow_share():
    ball, players = _empty(slots=4)
    t = np.arange(N) / 25.0
    players[:, 0, 0] = 10.0 + 5.0 * t  # 5 m/s
    players[:, 0, 1] = 10.0
    players[:, 1] = (20.0, 20.0)  # standing
    players[:, 2] = (30.0, 20.0)  # standing
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    assert f["ply_speed"][20] == pytest.approx(5 / 3)
    assert f["frac_slow"][20] == pytest.approx(2 / 3)
    assert f["ply_speed"][0] == pytest.approx(5 / 3)  # one-sided difference at the period edge
    assert f["ply_speed"][39] == pytest.approx(5 / 3)
    assert f["ball_speed"][20] == 0  # ball never visible


def test_ball_speed_and_nan_endpoints():
    ball, players = _empty()
    t = np.arange(N) / 25.0
    ball[:, 0] = 10.0 + 10.0 * t
    ball[:, 1] = 30.0
    ball[20] = np.nan
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    assert f["ball_speed"][30] == pytest.approx(10.0)
    assert f["ball_speed"][20] == pytest.approx(10.0)  # row itself missing, endpoints fine
    assert f["ball_speed"][18] == 0 and f["ball_speed"][22] == 0  # an endpoint is missing
    assert f["ball_speed"][25] == pytest.approx(10.0)


def test_spread_and_windows():
    ball, players = _empty(slots=4)
    players[:, 0] = (40.0, 10.0)
    players[:, 1] = (50.0, 10.0)  # home x std 5, y std 0
    players[:, 2] = (60.0, 20.0)
    players[:, 3] = (60.0, 40.0)  # away x std 0, y std 10
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    assert f["spread_x_diff"][10] == pytest.approx(5.0)
    assert f["spread_y_diff"][10] == pytest.approx(-10.0)
    assert f["spread_x_diff_w12"][20] == pytest.approx(5.0)
    assert f["spread_y_diff_w50"][20] == pytest.approx(-10.0)


def test_window_is_truncated_at_period_edges():
    ball, players = _empty()
    ball[:] = (50.0, 30.0)
    ball[0:20] = np.nan  # visible only in rows 20..39 of the first period
    f = _feat(_trk(ball, players, [0, 0, 1, 1]))
    # row 0: window [0, 12] -> all missing; row 39: window [27, 39] -> all visible
    assert f["ball_vis_w12"][0] == 0 and f["ball_vis_w12"][39] == 1
    # row 20 with +-12: rows 8..32 -> 13 of 25 visible
    assert f["ball_vis_w12"][20] == pytest.approx(13 / 25)
    # +-50 is truncated to the 40-row period (not the other period): 20 of 40
    assert f["ball_vis_w50"][20] == pytest.approx(0.5)
    assert f["ball_vis_w50"][50] == 1  # second period is fully visible: nothing leaks in from period 1
    assert f["ball_vis_w50"][39] == pytest.approx(0.5)


def test_periods_clipped_to_tracking_length():
    ball, players = _empty()
    ball[:] = (50.0, 30.0)
    f = _feat(_trk(ball, players, [0, 0, 1, 1], periods=[(80, 500), (95, 95), (-5, 3)]))
    assert f["ball_vis"][80:].sum() == 20 and f["ball_vis"][0:3].sum() == 3
    assert f["ball_vis"][3:80].sum() == 0


def test_unknown_team_slots_are_ignored_by_every_feature():
    from montreal.tracking import Tracking
    n = 200
    rng = np.random.default_rng(11)
    ball = np.stack([40.0 + 0.05 * np.arange(n), np.full(n, 30.0)], axis=1)
    base = np.stack([ball + rng.normal(0, 6.0, (n, 2)) for _ in range(6)], axis=1)
    team = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    ref = np.stack([ball + np.array([0.1, 0.1]) for _ in range(1)], axis=1)  # on the ball
    with_ref = Tracking(ball=ball, players=np.concatenate([base, ref], axis=1),
                        team=np.concatenate([team, [-1]]).astype(np.int8),
                        periods=[(0, n)], pitch=(105.0, 68.0))
    without = Tracking(ball=ball, players=base, team=team, periods=[(0, n)], pitch=(105.0, 68.0))
    Fa, Fb = compute_features(with_ref), compute_features(without)
    assert np.isfinite(Fa).all()
    np.testing.assert_allclose(Fa, Fb, atol=1e-12)

# ---------------------------------------------------------------------------------------
# features v2
# ---------------------------------------------------------------------------------------


def _cmean(x, half):
    """Reference centred mean of x over +-half rows, truncated at the ends of x (one period)."""
    x = np.asarray(x, dtype=float)
    return np.array([x[max(0, i - half): i + half + 1].mean() for i in range(len(x))])


def _pair(m, ball, hd, ad, periods=None):
    """One home and one away player at x-offsets hd / ad from the ball (d_diff = ad - hd)."""
    ball = np.asarray(ball, dtype=float)
    players = np.zeros((m, 2, 2))
    players[:, 0] = ball + np.stack([np.broadcast_to(hd, m), np.zeros(m)], axis=1)
    players[:, 1] = ball + np.stack([np.broadcast_to(ad, m), np.zeros(m)], axis=1)
    return _trk(ball, players, [0, 1], periods=periods or [(0, m)])


def test_near_run_counts_rows_since_the_team_became_nearest():
    m = 300
    hd = np.where(np.arange(m) < 120, 2.0, 6.0)  # home 2 m, then 6 m from the ball
    hd[200:210] = 4.0  # equal to the away player (4 m): d_diff == 0, nobody is nearest
    f = _feat(_pair(m, np.tile([50.0, 30.0], (m, 1)), hd, 4.0))
    h, a = f["near_run_home"], f["near_run_away"]
    assert h[0] == pytest.approx(1 / 25) and h[9] == pytest.approx(10 / 25)
    assert h[119] == pytest.approx(120 / 25) and (h[120:] == 0).all()
    assert (a[:120] == 0).all()
    assert a[120] == pytest.approx(1 / 25) and a[199] == pytest.approx(80 / 25)
    assert (a[200:210] == 0).all()  # d_diff == 0 breaks the run
    assert a[210] == pytest.approx(1 / 25) and a[299] == pytest.approx(90 / 25)


def test_near_run_resets_on_missing_ball_and_period_start_and_clips_at_10_s():
    m = 600
    ball = np.tile([50.0, 30.0], (m, 1))
    ball[100:110] = np.nan
    f = _feat(_pair(m, ball, 1.0, 9.0, periods=[(0, 300), (320, 600)]))
    h, a = f["near_run_home"], f["near_run_away"]
    assert h[99] == pytest.approx(100 / 25) and (h[100:110] == 0).all()
    assert h[110] == pytest.approx(1 / 25) and h[299] == pytest.approx(190 / 25)
    assert (h[300:320] == 0).all()
    assert h[320] == pytest.approx(1 / 25)  # a new period starts a new run
    assert h[568] == pytest.approx(249 / 25) and h[569] == 10.0 and h[599] == 10.0  # clipped
    assert (a == 0).all()


def test_gap_flip_when_the_ball_changes_hands_during_a_gap():
    m = 200
    ball = np.tile([50.0, 30.0], (m, 1))
    hd = np.where(np.arange(m) < 75, 0.0, 5.0)  # home on the ball, then 5 m away
    ad = np.where(np.arange(m) < 75, 5.0, 0.0)  # (away the other way round)
    ball[50:100] = np.nan  # gap 50..99: last_diff = +5 (home), next_diff = -5 (away)
    f = _feat(_pair(m, ball, hd, ad))
    flip = f["gap_flip"]
    assert (flip[50:100] == 1).all()
    assert flip[:50].sum() == 0 and flip[100:].sum() == 0  # only while the ball is missing


def test_gap_flip_needs_opposite_signs_and_two_metres_each():
    m = 200
    base = np.tile([50.0, 30.0], (m, 1))
    gap = base.copy()
    gap[50:100] = np.nan
    same = _feat(_pair(m, gap, 0.0, 5.0))["gap_flip"]  # home before and after: no flip
    assert same.sum() == 0
    t = np.arange(m)
    weak = _feat(_pair(m, gap, np.where(t < 75, 0.0, 1.5), np.where(t < 75, 1.5, 0.0)))["gap_flip"]
    assert weak.sum() == 0  # |d_diff| = 1.5 < 2 m on both sides
    edge = _feat(_pair(m, gap, np.where(t < 75, 0.0, 2.0), np.where(t < 75, 2.0, 0.0)))["gap_flip"]
    assert (edge[50:100] == 1).all()  # exactly 2 m counts
    lead = base.copy()
    lead[:60] = np.nan  # gap at the start of the period: no last visible row
    start = _feat(_pair(m, lead, 0.0, 5.0))["gap_flip"]
    assert start.sum() == 0


def test_ball_still():
    m = 300
    ball = np.tile([50.0, 30.0], (m, 1))
    ball[100:200, 0] += 0.04 * np.arange(100)  # 0.04 m / row = 1 m/s
    ball[120:126] = np.nan
    ball[245:250] = np.nan
    ball[251:256] = np.nan  # row 250 is visible but isolated: no speed -> not still
    f = _feat(_pair(m, ball, 1.0, 5.0))
    s = f["ball_still"]
    assert (s[:96] == 1).all() and (s[205:243] == 1).all() and (s[258:] == 1).all()
    assert (s[105:195][np.isfinite(ball[105:195, 0])] == 0).all()
    assert (s[120:126] == 0).all()  # missing ball
    assert s[250] == 0
    assert (s[243:258] == 0).all()  # the speed needs the rows 2 before and 2 after

RESTART_YES = [
    (0.5, 0.5), (104.0, 67.0), (0.0, 67.5),  # corner arcs (2 m)
    (52.5, 34.0), (54.4, 34.0),  # centre spot (2 m)
    (50.0, 1.2), (50.0, -1.4), (50.0, 69.3),  # within 1.5 m of a touchline, inside or outside
    (-1.0, 20.0), (105.9, 20.0),  # within 1.5 m of a goal line, inside or outside
    (3.0, 34.0), (3.0, 42.9), (102.0, 34.0), (5.4, 25.0),  # inside a goal area
]
RESTART_NO = [
    (52.5, 36.5), (50.0, 3.0), (50.0, -3.0), (-5.0, 20.0), (6.0, 34.0), (3.0, 43.5),
    (30.0, 20.0), (50.0, -20.0), (4.0, 3.0), (-50.0, 0.5),
]


def test_restart_spot():
    pts = np.array(RESTART_YES + RESTART_NO + [(np.nan, np.nan)], dtype=float)
    m = len(pts)
    players = np.full((m, 2, 2), np.nan)
    f = _feat(_trk(pts, players, [0, 1], periods=[(0, m)]))
    r = f["restart_spot"]
    ny = len(RESTART_YES)
    assert (r[:ny] == 1).all(), r[:ny]
    assert (r[ny:] == 0).all(), r[ny:]  # includes the missing ball (last row)


def _speed_profile(m, v_slow, v_fast, switch):
    v = np.where(np.arange(m) < switch, v_slow, v_fast)
    x = 20.0 + np.concatenate([[0.0], np.cumsum(v[:-1] / 25.0)])
    players = np.zeros((m, 2, 2))
    players[:, 0, 0] = x
    players[:, 0, 1] = 10.0
    players[:, 1, 0] = x
    players[:, 1, 1] = 50.0
    ball = np.full((m, 2), np.nan)
    return _trk(ball, players, [0, 1], periods=[(0, m)])


def test_ply_accel_ahead_minus_behind():
    f = _feat(_speed_profile(400, 0.5, 3.5, 200))["ply_accel"]
    assert 2.7 < f[200] < 3.1
    assert abs(f[100]) < 1e-6 and abs(f[350]) < 1e-6
    assert f[0] == 0.0 and f[399] == 0.0  # no rows behind / ahead: 0
    slow_down = _feat(_speed_profile(400, 3.5, 0.5, 200))["ply_accel"]
    assert -3.1 < slow_down[200] < -2.7


def test_ply_accel_is_clipped_to_six():
    up = _feat(_speed_profile(400, 0.0, 12.0, 200))["ply_accel"]
    down = _feat(_speed_profile(400, 12.0, 0.0, 200))["ply_accel"]
    assert up[200] == 6.0 and down[200] == -6.0
    assert up.max() <= 6.0 and down.min() >= -6.0


def test_v2_window_means_match_the_centred_means_of_their_sources():
    rng = np.random.default_rng(21)
    m = 300
    blocks = rng.uniform([-3.0, -3.0], [108.0, 71.0], size=(m // 30 + 1, 2))
    ball = np.repeat(blocks, 30, axis=0)[:m]  # still for 30 rows at a time
    ball[rng.random(m) < 0.15] = np.nan
    players = rng.uniform(0, 100, (m, 6, 2))
    f = _feat(_one_period_trk(m, ball, players))
    for name, src, half in (("ball_still_w50", "ball_still", 50), ("restart_spot_w50", "restart_spot", 50),
                            ("near_run_home_w12", "near_run_home", 12),
                            ("near_run_away_w12", "near_run_away", 12)):
        np.testing.assert_allclose(f[name], _cmean(f[src], half), atol=1e-9, err_msg=name)
    assert f["ball_still"].sum() > 0 and f["near_run_home"].max() > 0


def _one_period_trk(m, ball, players):
    return _trk(ball, players, [0, 0, 0, 1, 1, 1], periods=[(0, m)])


def test_v2_features_are_finite_and_zero_outside_the_periods():
    rng = np.random.default_rng(22)
    ball = rng.uniform(-5, 110, (N, 2))
    ball[rng.random(N) < 0.3] = np.nan
    players = rng.uniform(0, 100, (N, 6, 2))
    players[rng.random((N, 6)) < 0.3] = np.nan
    f = _feat(_trk(ball, players, [0, 0, 0, 1, 1, 1]))
    outside = np.r_[40:50, 90:100]
    for name in V2_NAMES:
        assert (f[name][outside] == 0).all(), name