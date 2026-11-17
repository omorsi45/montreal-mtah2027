import json

import numpy as np
import pytest

from montreal.emission import SoftmaxRegression, heuristic_logp
from montreal.features import FEATURE_NAMES

H, A, D, X = 0, 1, 2, 3


def _neutral(n=1):
    """Feature rows where the heuristic is indifferent: ball visible, team signal zero."""
    F = np.zeros((n, len(FEATURE_NAMES)))
    c = FEATURE_NAMES.index
    F[:, c("ball_vis")] = 1.0
    F[:, c("ball_vis_w50")] = 1.0
    F[:, c("frac_slow_w50")] = 0.6
    return F


def _lse(x):
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]


def test_heuristic_shapes_and_direction():
    c = FEATURE_NAMES.index
    F = _neutral(3)
    F[0, c("d_diff_w12")] = 10.0  # ball nearer home -> H
    F[1, c("d_diff_w12")] = -10.0  # nearer away -> A
    F[2, c("ball_out_w12")] = 1.0  # ball out, not visible, last seen out -> D
    F[2, c("ball_vis")] = 0.0
    F[2, c("last_out")] = 1.0
    logp = heuristic_logp(F, FEATURE_NAMES)
    assert logp.shape == (3, 3)
    assert np.allclose(_lse(logp), 0.0, atol=1e-12)
    assert list(logp.argmax(axis=1)) == [H, A, D]


def test_heuristic_neutral_splits_teams_evenly():
    logp = heuristic_logp(_neutral(2), FEATURE_NAMES)
    assert np.allclose(logp[:, 0], logp[:, 1])
    assert (logp[:, 2] < logp[:, 0]).all()  # z_dead = -1.5 at neutral


def test_heuristic_exact_formula():
    c = FEATURE_NAMES.index
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1, size=(5, len(FEATURE_NAMES)))
    gap = 1 - F[:, c("ball_vis")]
    zt = 0.35 * F[:, c("d_diff_w12")] + 0.15 * gap * (
        F[:, c("last_diff")] * np.exp(-F[:, c("since_seen")] / 8)
        + F[:, c("next_diff")] * np.exp(-F[:, c("until_seen")] / 8)
    )
    zd = (
        -1.5
        + 3.0 * F[:, c("ball_out_w12")]
        + 2.5 * gap * F[:, c("last_out")]
        + 2.0 * (F[:, c("frac_slow_w50")] - 0.6)
        + 1.0 * (1 - F[:, c("ball_vis_w50")])
    )
    z = np.stack([zt / 2, -zt / 2, zd], axis=1)
    expected = z - _lse(z)[:, None]
    assert np.allclose(heuristic_logp(F, FEATURE_NAMES), expected, atol=1e-12)


def test_heuristic_is_column_order_independent():
    rng = np.random.default_rng(1)
    F = rng.uniform(0, 1, size=(6, len(FEATURE_NAMES)))
    perm = rng.permutation(len(FEATURE_NAMES))
    names2 = [FEATURE_NAMES[i] for i in perm]
    assert np.array_equal(
        heuristic_logp(F, FEATURE_NAMES), heuristic_logp(F[:, perm], names2)
    )


def _blobs(n_per=100, seed=3):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    Xs = np.concatenate([c + rng.normal(size=(n_per, 2)) for c in centres])
    y = np.repeat([H, A, D], n_per)
    return Xs, y


def test_softmax_learns_separable():
    Xs, y = _blobs()
    m = SoftmaxRegression().fit(Xs, y)
    lp = m.log_proba(Xs)
    assert lp.shape == (300, 3)
    assert np.allclose(_lse(lp), 0.0, atol=1e-12)
    assert (lp.argmax(axis=1) == y).mean() > 0.95


def test_softmax_deterministic_and_roundtrip():
    Xs, y = _blobs()
    a = SoftmaxRegression(l2=1e-2).fit(Xs, y, feature_names=["u", "v"])
    b = SoftmaxRegression(l2=1e-2).fit(Xs, y, feature_names=["u", "v"])
    assert a.to_dict() == b.to_dict()
    d = json.loads(json.dumps(a.to_dict()))  # JSON-safe
    c = SoftmaxRegression.from_dict(d)
    assert np.array_equal(a.log_proba(Xs), c.log_proba(Xs))
    assert d["feature_names"] == ["u", "v"]
    assert d["class_weight"] is None and d["l2"] == 1e-2


def test_ignores_X_rows():
    Xs, y = _blobs()
    base = SoftmaxRegression().fit(Xs, y)
    rng = np.random.default_rng(4)
    Xx = rng.normal(size=(50, 2)) * 10
    both = SoftmaxRegression().fit(np.vstack([Xs, Xx]), np.concatenate([y, np.full(50, X)]))
    assert base.to_dict() == both.to_dict()


def test_balanced_weights_shift_prior_towards_minority():
    rng = np.random.default_rng(5)
    Xs = rng.normal(size=(1000, 2))  # uninformative features
    y = np.where(rng.uniform(size=1000) < 0.9, H, A)
    plain = SoftmaxRegression().fit(Xs, y).log_proba(np.zeros((1, 2)))[0]
    bal = SoftmaxRegression(class_weight="balanced").fit(Xs, y).log_proba(np.zeros((1, 2)))[0]
    assert plain[H] - plain[A] > 1.5  # follows the 9:1 prior
    assert abs(bal[H] - bal[A]) < 0.3  # balanced removes it


def test_constant_column_and_missing_class_are_stable():
    Xs, y = _blobs()
    Xs = np.hstack([Xs, np.full((len(Xs), 1), 7.0)])
    m = SoftmaxRegression().fit(Xs, y)
    assert np.isfinite(m.log_proba(Xs)).all()
    m2 = SoftmaxRegression().fit(Xs[y != D], y[y != D])  # class D never seen
    lp = m2.log_proba(Xs)
    assert np.isfinite(lp).all()
    assert (lp.argmax(axis=1)[y == H] == H).mean() > 0.95


def test_fit_without_rows_raises():
    with pytest.raises(ValueError):
        SoftmaxRegression().fit(np.zeros((3, 2)), np.full(3, X))


def test_log_proba_stable_for_huge_inputs():
    Xs, y = _blobs()
    m = SoftmaxRegression().fit(Xs, y)
    lp = m.log_proba(Xs * 1e6)
    assert np.isfinite(lp).all()
    assert np.allclose(_lse(lp), 0.0, atol=1e-9)


def test_sample_weight_zero_equals_dropping():
    Xs, y = _blobs()
    w = np.ones(len(y))
    w[:40] = 0.0
    a = SoftmaxRegression().fit(Xs, y, sample_weight=w)
    b = SoftmaxRegression().fit(Xs[40:], y[40:])
    assert np.allclose(a.log_proba(Xs), b.log_proba(Xs), atol=1e-4)
