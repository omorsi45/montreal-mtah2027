"""Three-state (H/A/D) Viterbi decoding with switch costs, decoded per period."""

from __future__ import annotations

import numpy as np

H, A, D, X = 0, 1, 2, 3

# For each current state s: predecessor candidates in tie-break order (stay first, then lower index).
_ORDER = tuple(
    (s,) + tuple(p for p in range(3) if p != s) for s in range(3)
)


def cost_matrix(swap: float, dead: float, live: float) -> np.ndarray:
    """Transition costs; rows are from H/A/D, columns are to H/A/D."""
    return np.array(
        [[0.0, swap, dead], [swap, 0.0, dead], [live, live, 0.0]], dtype=float
    )


def viterbi(logp: np.ndarray, cost: np.ndarray, init: int | None = None) -> np.ndarray:
    """Maximise sum(logp[t, s_t]) - sum(cost[s_{t-1}, s_t]); returns int8 states (n,).

    With `init`, frame 0 pays cost[init, s] (a virtual previous state `init`); without it frame 0
    scores are logp[0]. Ties: prefer keeping the previous state, then the lower state index
    (at frame 0 / the final frame: the lowest index).
    """
    logp = np.round(np.asarray(logp, dtype=float), 6)
    n = logp.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    c = [[float(cost[i][j]) for j in range(3)] for i in range(3)]
    rows = logp.tolist()
    r0 = rows[0]
    if init is None:
        s0, s1, s2 = r0
    else:
        ci = c[init]
        s0, s1, s2 = r0[0] - ci[0], r0[1] - ci[1], r0[2] - ci[2]

    # Per current state: (stay-candidate, other1, other2) predecessor indices and their costs.
    o0, o1, o2 = _ORDER
    c00, c01, c02 = c[o0[0]][0], c[o0[1]][0], c[o0[2]][0]
    c10, c11, c12 = c[o1[0]][1], c[o1[1]][1], c[o1[2]][1]
    c20, c21, c22 = c[o2[0]][2], c[o2[1]][2], c[o2[2]][2]
    p00, p01, p02 = o0
    p10, p11, p12 = o1
    p20, p21, p22 = o2

    bp = [(0, 0, 0)] * n  # backpointers per frame: best predecessor for each current state
    for t in range(1, n):
        prev = (s0, s1, s2)
        r = rows[t]

        b = p00
        v = prev[p00] - c00
        w = prev[p01] - c01
        if w > v:
            v = w
            b = p01
        w = prev[p02] - c02
        if w > v:
            v = w
            b = p02
        n0 = v + r[0]
        b0 = b

        b = p10
        v = prev[p10] - c10
        w = prev[p11] - c11
        if w > v:
            v = w
            b = p11
        w = prev[p12] - c12
        if w > v:
            v = w
            b = p12
        n1 = v + r[1]
        b1 = b

        b = p20
        v = prev[p20] - c20
        w = prev[p21] - c21
        if w > v:
            v = w
            b = p21
        w = prev[p22] - c22
        if w > v:
            v = w
            b = p22
        n2 = v + r[2]

        s0, s1, s2 = n0, n1, n2
        bp[t] = (b0, b1, b)

    # Final state: highest score, lowest index on ties.
    s = 0
    best = s0
    if s1 > best:
        s, best = 1, s1
    if s2 > best:
        s = 2
    out = [0] * n
    for t in range(n - 1, -1, -1):
        out[t] = s
        if t:
            s = bp[t][s]
    return np.asarray(out, dtype=np.int8)


def decode_periods(logp: np.ndarray, periods, cost: np.ndarray) -> np.ndarray:
    """Viterbi run independently inside each half-open period, X (3) elsewhere.

    Every period starts as if the previous state were D (kick-off), so the first alive frame pays
    `live` once.
    """
    logp = np.asarray(logp)
    out = np.full(logp.shape[0], X, dtype=np.int8)
    for a, b in periods:
        a, b = int(a), int(b)
        if b > a:
            out[a:b] = viterbi(logp[a:b], cost, init=D)
    return out
