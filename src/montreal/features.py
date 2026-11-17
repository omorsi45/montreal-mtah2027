"""Per-frame, team-level feature matrix computed from a Tracking object.

Everything is computed independently inside each period (no difference, carry or window crosses a
period boundary) and rows outside the periods are 0. All values are finite. The code is vectorised
over frames; the only Python loop is over periods.

Conventions
-----------
* Tracked player: a slot whose x and y are both finite on that row.
* Speed (players and ball): centred difference over +-2 rows, ``|x[t+2] - x[t-2]| * fps / 4``.
  Near a period edge the difference is one-sided over the rows that are available (e.g. at the
  first row ``x[2] - x[0]`` over 2 rows), so the divisor is always the real row gap. It is NaN if
  either endpoint is NaN; NaN speeds are excluded from means and shares. A one-row period has no
  speed.
* Team centroid velocity: same scheme over +-12 rows; 0 if either endpoint has no tracked player.
* ``block_prog`` is a progress coordinate in [0, 1]: centroid x of all tracked players divided by
  L when home attacks +x, and ``(L - x) / L`` when home attacks -x, so that larger always means
  "further up the pitch for home". 0 when nobody is tracked.
* Home attacks +x in a period iff home's mean x over the period is smaller than away's; if either
  team has no tracked position in the period, +1.
* "previous / next visible row" in the carry features includes the current row (so on a visible
  row the carry value equals the row's own value and ``since_seen`` / ``until_seen`` are 0).
* Distances to the ball are Euclidean; "within 3 m" is ``<= 3``. Spread is the population std (ddof
  0) of the tracked players' coordinate; the diff is 0 when either team has nobody tracked.
"""

from __future__ import annotations

import warnings

import numpy as np

from montreal.tracking import Tracking

BASE_NAMES = [
    "ball_vis",
    "d_home",
    "d_away",
    "d_diff",
    "n_home_3m",
    "n_away_3m",
    "ball_out",
    "ball_edge",
    "ball_speed",
    "ply_speed",
    "frac_slow",
    "block_prog",
    "spread_x_diff",
    "spread_y_diff",
    "vel_home",
    "vel_away",
]
CARRY_NAMES = [
    "last_diff",
    "next_diff",
    "since_seen",
    "until_seen",
    "last_out",
    "last_edge",
    "next_edge",
]
WINDOW_SOURCES = [
    "d_diff",
    "ball_out",
    "ply_speed",
    "frac_slow",
    "spread_x_diff",
    "spread_y_diff",
    "vel_home",
    "vel_away",
    "ball_vis",
]
WINDOWS = (12, 50)
FEATURE_NAMES: list[str] = (
    BASE_NAMES
    + CARRY_NAMES
    + [f"{name}_w{w}" for name in WINDOW_SOURCES for w in WINDOWS]
)
_COL = {name: i for i, name in enumerate(FEATURE_NAMES)}

MAX_DIST = 30.0
NEAR_M = 3.0
SLOW_MPS = 1.0
SEEN_CAP_S = 30.0


def _gap_diff(X: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    """``X[hi] - X[lo]`` and the row gap ``hi - lo`` with lo/hi = t -/+ half clamped to the period."""
    m = X.shape[0]
    t = np.arange(m)
    lo = np.clip(t - half, 0, m - 1)
    hi = np.clip(t + half, 0, m - 1)
    return X[hi] - X[lo], (hi - lo).astype(np.float64)


def _speed(X: np.ndarray, fps: int) -> np.ndarray:
    """Speed (m/s) of positions X (m, ..., 2) by the centred +-2 difference; NaN if not computable."""
    d, gap = _gap_diff(X, 2)
    gap = gap.reshape((-1,) + (1,) * (X.ndim - 2))
    with np.errstate(all="ignore"):
        s = np.hypot(d[..., 0], d[..., 1]) * fps / np.where(gap > 0, gap, np.nan)
    return s


def _centered_mean(X: np.ndarray, half: int) -> np.ndarray:
    """Centred mean of the columns of X (m, k) over +-half rows, truncated at the period edges."""
    m, k = X.shape
    c = np.concatenate([np.zeros((1, k)), np.cumsum(X, axis=0)], axis=0)
    t = np.arange(m)
    lo = np.clip(t - half, 0, m)
    hi = np.clip(t + half + 1, 0, m)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def _team_stats(P: np.ndarray):
    """Tracked count, coordinate sums, centroid (NaN if none) and per-axis std (NaN if none)."""
    tr = np.isfinite(P).all(axis=-1)
    cnt = tr.sum(axis=1)
    sums = np.where(tr[..., None], P, 0.0).sum(axis=1)
    with np.errstate(all="ignore"):
        cen = sums / cnt[:, None]
        dev = np.where(tr[..., None], P - cen[:, None, :], 0.0)
        std = np.sqrt((dev**2).sum(axis=1) / cnt[:, None])
    return cnt, sums, cen, std


def _min_dist(D: np.ndarray) -> np.ndarray:
    """Row-wise min of D ignoring NaN, clipped to [0, 30]; 30 if nothing is finite."""
    d = np.where(np.isnan(D), np.inf, D).min(axis=1) if D.shape[1] else np.full(len(D), np.inf)
    return np.clip(np.where(np.isinf(d), MAX_DIST, d), 0.0, MAX_DIST)


def _period_features(
    ball: np.ndarray,
    players: np.ndarray,
    team: np.ndarray,
    pitch: tuple[float, float],
    fps: int,
) -> np.ndarray:
    m = len(ball)
    L, W = pitch
    idx = np.arange(m)
    is_home = team == 0
    Ph, Pa = players[:, is_home, :], players[:, team == 1, :]
    col: dict[str, np.ndarray] = {}

    # ---- ball ---------------------------------------------------------------------------
    bx, by = ball[:, 0], ball[:, 1]
    vis = np.isfinite(ball).all(axis=1)
    col["ball_vis"] = vis.astype(np.float64)
    out = vis & ((bx < -0.5) | (bx > L + 0.5) | (by < -0.5) | (by > W + 0.5))
    col["ball_out"] = out.astype(np.float64)
    edge = np.minimum(np.minimum(bx, L - bx), np.minimum(by, W - by))
    edge = np.clip(np.where(vis, edge, 10.0), 0.0, 10.0)  # outside -> negative -> 0
    col["ball_edge"] = np.where(out, 0.0, edge)
    bs = _speed(ball, fps)
    col["ball_speed"] = np.clip(np.where(np.isfinite(bs), bs, 0.0), 0.0, 40.0)

    # ---- distances to the ball ----------------------------------------------------------
    def dist(P):
        return np.hypot(P[..., 0] - bx[:, None], P[..., 1] - by[:, None])

    Dh, Da = dist(Ph), dist(Pa)
    col["d_home"] = _min_dist(Dh)
    col["d_away"] = _min_dist(Da)
    col["d_diff"] = np.where(vis, np.clip(col["d_away"] - col["d_home"], -15.0, 15.0), 0.0)
    col["n_home_3m"] = (Dh <= NEAR_M).sum(axis=1).astype(np.float64)
    col["n_away_3m"] = (Da <= NEAR_M).sum(axis=1).astype(np.float64)

    # ---- player speeds ------------------------------------------------------------------
    tracked = np.isfinite(players).all(axis=-1)
    sp = _speed(players, fps)
    ok = tracked & np.isfinite(sp)
    n_ok = ok.sum(axis=1)
    sp0 = np.where(ok, sp, 0.0)
    with np.errstate(all="ignore"):
        mean_sp = sp0.sum(axis=1) / n_ok
        slow = (ok & (sp < SLOW_MPS)).sum(axis=1) / n_ok
    col["ply_speed"] = np.clip(np.where(n_ok > 0, mean_sp, 0.0), 0.0, 12.0)
    col["frac_slow"] = np.where(n_ok > 0, slow, 1.0)

    # ---- team shape ---------------------------------------------------------------------
    cnt_h, sum_h, cen_h, std_h = _team_stats(Ph)
    cnt_a, sum_a, cen_a, std_a = _team_stats(Pa)

    ch, ca = cnt_h.sum(), cnt_a.sum()
    if ch > 0 and ca > 0:
        direction = 1.0 if sum_h[:, 0].sum() / ch < sum_a[:, 0].sum() / ca else -1.0
    else:
        direction = 1.0

    cnt_all = cnt_h + cnt_a
    with np.errstate(all="ignore"):
        cx = (sum_h[:, 0] + sum_a[:, 0]) / cnt_all
    prog = (cx if direction > 0 else L - cx) / L
    col["block_prog"] = np.where(cnt_all > 0, prog, 0.0)

    both = (cnt_h > 0) & (cnt_a > 0)
    for k, name in enumerate(("spread_x_diff", "spread_y_diff")):
        diff = np.where(both, std_h[:, k] - std_a[:, k], 0.0)
        col[name] = np.clip(np.nan_to_num(diff), -20.0, 20.0)

    def centroid_vel(cen: np.ndarray, sign: float) -> np.ndarray:
        d, gap = _gap_diff(cen[:, 0], 12)
        with np.errstate(all="ignore"):
            v = sign * d * fps / np.where(gap > 0, gap, np.nan)
        return np.clip(np.where(np.isfinite(v), v, 0.0), -8.0, 8.0)

    col["vel_home"] = centroid_vel(cen_h, direction)
    col["vel_away"] = centroid_vel(cen_a, -direction)

    # ---- carry (previous / next visible row, inside the period) -------------------------
    last = np.maximum.accumulate(np.where(vis, idx, -1))
    nxt = np.minimum.accumulate(np.where(vis, idx, m)[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < m
    li, ni = np.where(has_last, last, 0), np.where(has_next, nxt, 0)
    col["last_diff"] = np.where(has_last, col["d_diff"][li], 0.0)
    col["next_diff"] = np.where(has_next, col["d_diff"][ni], 0.0)
    col["since_seen"] = np.clip(np.where(has_last, (idx - last) / fps, SEEN_CAP_S), 0.0, SEEN_CAP_S)
    col["until_seen"] = np.clip(np.where(has_next, (nxt - idx) / fps, SEEN_CAP_S), 0.0, SEEN_CAP_S)
    col["last_out"] = np.where(has_last, col["ball_out"][li], 0.0)
    col["last_edge"] = np.where(has_last, col["ball_edge"][li], 10.0)
    col["next_edge"] = np.where(has_next, col["ball_edge"][ni], 10.0)

    # ---- windows ------------------------------------------------------------------------
    src = np.stack([col[name] for name in WINDOW_SOURCES], axis=1)
    for w in WINDOWS:
        wm = _centered_mean(src, w)
        for j, name in enumerate(WINDOW_SOURCES):
            col[f"{name}_w{w}"] = wm[:, j]

    return np.stack([col[name] for name in FEATURE_NAMES], axis=1)


def compute_features(trk: Tracking) -> np.ndarray:
    """Feature matrix (n, len(FEATURE_NAMES)) float64; finite, 0 outside the periods."""
    n = len(trk.ball)
    F = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    team = np.asarray(trk.team)
    players = trk.players[:, team >= 0]  # slots of an unknown team (-1) are ignored
    team = team[team >= 0]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start, stop in trk.periods:
            a, b = max(0, int(start)), min(n, int(stop))
            if b <= a:
                continue
            F[a:b] = _period_features(
                trk.ball[a:b], players[a:b], team, trk.pitch, trk.fps
            )
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
