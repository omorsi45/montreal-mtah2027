"""Emission models: per-frame log P(H), log P(A), log P(D) from the feature matrix.

Two interchangeable models, both returning an (n, 3) array of log-probabilities in the label order
H=0, A=1, D=2:

* ``heuristic_logp`` - a hand-set formula (Milestone 1), no training.
* ``SoftmaxRegression`` - multinomial logistic regression fitted by Newton/IRLS in pure numpy.
  Fully deterministic (no randomness anywhere).
"""

from __future__ import annotations

import numpy as np

X_LABEL = 3  # label code X (ungraded); H=0, A=1, D=2
N_CLASSES = 3


def _log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise log-softmax."""
    m = z.max(axis=1, keepdims=True)
    s = z - m
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def heuristic_logp(F: np.ndarray, names: list[str]) -> np.ndarray:
    """Hand-set emission: log-softmax of ``[z_team/2, -z_team/2, z_dead]``.

    ``d_diff = d_away - d_home`` so positive (ball nearer home) favours H. Columns are looked up by
    name, so any column order works. ``gap = 1 - ball_vis`` switches on the carried-over evidence
    from the last / next visible frame, which decays with the time since / until it was seen.
    """
    F = np.asarray(F, dtype=np.float64)

    def col(name: str) -> np.ndarray:
        return F[:, names.index(name)]

    gap = 1.0 - col("ball_vis")
    z_team = 0.35 * col("d_diff_w12") + 0.15 * gap * (
        col("last_diff") * np.exp(-col("since_seen") / 8.0)
        + col("next_diff") * np.exp(-col("until_seen") / 8.0)
    )
    z_dead = (
        -1.5
        + 3.0 * col("ball_out_w12")
        + 2.5 * gap * col("last_out")
        + 2.0 * (col("frac_slow_w50") - 0.6)
        + 1.0 * (1.0 - col("ball_vis_w50"))
    )
    return _log_softmax(np.stack([z_team / 2.0, -z_team / 2.0, z_dead], axis=1))


class SoftmaxRegression:
    """Multinomial logistic regression (K=3) with an L2 penalty on the weights (not the intercept).

    Features are standardised with the training mean / std (std 0 -> 1). Rows labelled X are dropped
    before fitting. Fitted by full-batch Newton/IRLS with a backtracking line search; the
    objective is the weighted mean negative log-likelihood plus ``0.5 * l2 * ||W||^2``.
    """

    MAX_ITER = 25
    TOL = 1e-6

    def __init__(self, l2: float = 1e-2, class_weight: str | None = None):
        if class_weight not in (None, "balanced"):
            raise ValueError(f"class_weight must be None or 'balanced', got {class_weight!r}")
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.feature_names: list[str] | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.W: np.ndarray | None = None  # (d, 3)
        self.b: np.ndarray | None = None  # (3,)

    # ---- fitting ----------------------------------------------------------------------------
    def _objective(self, Z: np.ndarray, Y: np.ndarray, w: np.ndarray, Theta: np.ndarray) -> float:
        lp = _log_softmax(Z @ Theta)
        nll = -(w * (Y * lp).sum(axis=1)).sum()
        return float(nll + 0.5 * self.l2 * (Theta[:-1] ** 2).sum())

    def fit(self, X, y, sample_weight=None, feature_names: list[str] | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be (n, d) and y must have length n")
        keep = y != X_LABEL
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            keep &= sample_weight > 0
            sw = sample_weight[keep]
        else:
            sw = np.ones(int(keep.sum()))
        Xk, yk = X[keep], y[keep].astype(np.int64)
        n, d = Xk.shape
        if n == 0:
            raise ValueError("no rows with label H/A/D to fit on")
        if yk.min() < 0 or yk.max() >= N_CLASSES:
            raise ValueError("labels must be in {0, 1, 2} (or 3 = X, dropped)")

        w = sw.copy()
        if self.class_weight == "balanced":
            counts = np.bincount(yk, minlength=N_CLASSES).astype(np.float64)
            cw = np.where(counts > 0, n / (N_CLASSES * np.maximum(counts, 1.0)), 0.0)
            w = w * cw[yk]
        w = w / w.sum()  # weighted mean objective

        self.feature_names = None if feature_names is None else [str(s) for s in feature_names]
        self.mean = Xk.mean(axis=0)
        std = Xk.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        Z = np.hstack([(Xk - self.mean) / self.std, np.ones((n, 1))])  # (n, d+1), last col = bias
        Y = np.zeros((n, N_CLASSES))
        Y[np.arange(n), yk] = 1.0

        p1 = d + 1
        ridge = np.full(p1, self.l2)
        ridge[-1] = 0.0  # intercept unpenalised
        Theta = np.zeros((p1, N_CLASSES))
        obj = self._objective(Z, Y, w, Theta)
        for _ in range(self.MAX_ITER):
            P = np.exp(_log_softmax(Z @ Theta))
            G = Z.T @ (w[:, None] * (P - Y)) + ridge[:, None] * Theta  # (p1, 3)
            Hs = np.zeros((N_CLASSES * p1, N_CLASSES * p1))
            for k in range(N_CLASSES):
                for m in range(k, N_CLASSES):
                    s = w * P[:, k] * ((1.0 if k == m else 0.0) - P[:, m])
                    blk = Z.T @ (Z * s[:, None])
                    Hs[k * p1 : (k + 1) * p1, m * p1 : (m + 1) * p1] = blk
                    if m != k:
                        Hs[m * p1 : (m + 1) * p1, k * p1 : (k + 1) * p1] = blk.T
            # Penalty curvature, plus a tiny jitter: adding the same constant to every class's
            # intercept leaves the likelihood unchanged, so the Hessian is singular there.
            Hs[np.diag_indices_from(Hs)] += np.tile(ridge, N_CLASSES) + 1e-10
            step = np.linalg.solve(Hs, G.T.reshape(-1)).reshape(N_CLASSES, p1).T  # (p1, 3)

            t = 1.0
            for _ in range(30):  # backtracking: accept the first step that does not raise the loss
                cand = Theta - t * step
                cand_obj = self._objective(Z, Y, w, cand)
                if cand_obj <= obj + 1e-12:
                    break
                t *= 0.5
            else:
                break
            delta = float(np.abs(cand - Theta).max())
            Theta, obj = cand, cand_obj
            if delta < self.TOL:
                break

        self.W = Theta[:-1].copy()
        self.b = Theta[-1].copy()
        return self

    # ---- inference --------------------------------------------------------------------------
    def log_proba(self, X) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("model is not fitted")
        Xs = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        return _log_softmax(Xs @ self.W + self.b)

    # ---- serialisation ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        if self.W is None:
            raise RuntimeError("model is not fitted")
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "l2": self.l2,
            "class_weight": self.class_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoftmaxRegression":
        m = cls(l2=float(d["l2"]), class_weight=d["class_weight"])
        m.feature_names = None if d["feature_names"] is None else list(d["feature_names"])
        m.mean = np.asarray(d["mean"], dtype=np.float64)
        m.std = np.asarray(d["std"], dtype=np.float64)
        m.W = np.asarray(d["W"], dtype=np.float64)
        m.b = np.asarray(d["b"], dtype=np.float64)
        return m
