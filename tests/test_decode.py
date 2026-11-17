import time

import numpy as np

from montreal.decode import cost_matrix, decode_periods, viterbi


def test_cost_matrix_layout():
    c = cost_matrix(swap=1.0, dead=2.0, live=3.0)
    expected = np.array([[0, 1, 2], [1, 0, 2], [3, 3, 0]], dtype=float)
    assert c.shape == (3, 3)
    assert np.array_equal(c, expected)


def test_high_cost_suppresses_flicker():
    logp = np.tile([1.0, 0.0, -5.0], (60, 1))
    for t in (10, 30, 50):
        logp[t] = [0.0, 0.5, -5.0]  # single-frame A blip
    out_hi = viterbi(logp, cost_matrix(10, 10, 10))
    assert out_hi.dtype == np.int8
    assert np.all(out_hi == 0)
    out_lo = viterbi(logp, cost_matrix(0, 0, 0))
    assert [int(out_lo[t]) for t in (10, 30, 50)] == [1, 1, 1]
    assert int((out_lo == 1).sum()) == 3


def test_long_switch_survives():
    logp = np.zeros((400, 3))
    logp[:200] = [1.0, 0.0, -5.0]
    logp[200:] = [0.0, 1.0, -5.0]
    out = viterbi(logp, cost_matrix(20, 20, 20))
    switches = np.flatnonzero(np.diff(out.astype(int)) != 0)
    assert len(switches) == 1
    assert abs(int(switches[0]) + 1 - 200) <= 1
    assert out[0] == 0 and out[-1] == 1


def test_periods_independent():
    n = 40
    logp = np.zeros((n, 3))
    logp[:, 0] = 1.0  # favour H everywhere
    cost = cost_matrix(5, 5, 0)
    out = decode_periods(logp, [(0, 10), (20, 30)], cost)
    assert out.dtype == np.int8
    assert np.all(out[10:20] == 3)
    assert np.all(out[:10] == 0) and np.all(out[20:30] == 0)
    assert np.all(out[30:] == 3)
    # Frame 9 strongly favours A; the second period must not be influenced by it.
    logp2 = logp.copy()
    logp2[5:10] = [-9.0, 5.0, -9.0]
    out2 = decode_periods(logp2, [(0, 10), (20, 30)], cost)
    assert np.array_equal(out2[20:30], out[20:30])
    assert not np.array_equal(out2[:10], out[:10])


def test_period_starts_from_dead_virtual_state():
    logp = np.tile([1.0, -9.0, 0.0], (5, 1))
    out = decode_periods(logp, [(0, 5)], cost_matrix(20, 10, 10))
    assert np.all(out == 2)
    out = decode_periods(logp, [(0, 5)], cost_matrix(20, 10, 0))
    assert np.all(out == 0)


def test_init_changes_frame_zero():
    logp = np.tile([1.0, 0.0, 0.0], (3, 1))
    cost = cost_matrix(20, 10, 10)
    assert np.all(viterbi(logp, cost) == 0)
    assert np.all(viterbi(logp, cost, init=2) == 2)  # H would pay live=10 for a gain of 3
    # init=0 makes staying in H free, so H wins
    assert np.all(viterbi(logp, cost, init=0) == 0)


def test_deterministic_ties():
    logp = np.zeros((50, 3))
    cost = cost_matrix(3, 2, 1)
    a = viterbi(logp, cost)
    b = viterbi(logp, cost)
    assert np.all(a == 0)
    assert np.array_equal(a, b)


def test_ties_prefer_stay_then_lower_index():
    # Everything ties except the last frame, which picks D. Stay-preference traces D back
    # through every frame; plain lowest-index backtracking would give [0, 0, 2].
    logp = np.zeros((3, 3))
    logp[2] = [0.0, 0.0, 1.0]
    out = viterbi(logp, cost_matrix(0, 0, 0))
    assert list(out) == [2, 2, 2]
    # Predecessor tie between H and A (lower index wins) for a switch into D.
    logp = np.zeros((2, 3))
    logp[1] = [0.0, 0.0, 1.0]
    logp[0] = [0.0, 0.0, -9.0]
    out = viterbi(logp, cost_matrix(0, 0, 0))
    assert list(out) == [0, 2]


def test_empty_and_rounding():
    cost = cost_matrix(1, 1, 1)
    assert viterbi(np.zeros((0, 3)), cost).shape == (0,)
    # Differences below 1e-6 vanish after rounding -> tie -> state 0
    logp = np.tile([0.0, 1e-9, 0.0], (10, 1))
    assert np.all(viterbi(logp, cost) == 0)


def test_speed_200k_frames():
    rng = np.random.default_rng(20261117)
    logp = rng.normal(size=(200_000, 3))
    t0 = time.perf_counter()
    out = viterbi(logp, cost_matrix(2.0, 1.0, 1.0))
    elapsed = time.perf_counter() - t0
    assert out.shape == (200_000,)
    assert elapsed < 2.0
