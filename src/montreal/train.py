"""Leave-one-game-out (LOGO) training and switch-cost tuning on Metrica's public sample games.

CLI: ``python -m montreal.train [--params models/params.json] [--workers N]``.

Every choice (emission model, switch costs) is made on the three public sample games only; the
competition match and its ground truth are never touched here. Each fold fits on two games and
scores on the third, with ball/player dropout simulated on both sides so the models learn on
tracking with realistic long gaps (label-independent). The ball-gap lengths are resampled from the
public sample games' own missing-ball runs (external_gap_lengths), never from the competition match.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from montreal import paths
from montreal.decode import cost_matrix, decode_periods
from montreal.emission import SoftmaxRegression, heuristic_logp
from montreal.features import FEATURE_NAMES, FEATURE_SETS, compute_features
from montreal.metrics import score
from montreal.tracking import Tracking

SEED = 20261117
# Design constants, set a priori and not tuned on any data:
BALL_SHARE = 0.25  # newly masked ball rows, as a share of each period's rows
PLAYER_SHARE = 0.05  # newly masked rows per player slot, as a share of each period's rows
PLAYER_BLOCK = (25, 250)  # uniform block length (rows)
MIN_GAP = 50  # shortest external missing-ball run (rows) that counts as a gap to resample
TRAIN_STRIDE = 5  # train on every 5th row
_MAX_BLOCKS = 20_000  # safety valve against pathological (almost fully missing) inputs

COST_GRID: list[tuple[float, float, float]] = list(
    itertools.product((2.5, 5, 10, 20, 40, 80, 160), (2.5, 5, 10, 20, 40), (2.5, 5, 10, 20, 40))
)  # (swap, dead, live)


@dataclass(frozen=True)
class Candidate:
    name: str
    l2: float | None = None  # None -> the heuristic
    class_weight: str | None = None

    def make(self) -> SoftmaxRegression:
        return SoftmaxRegression(l2=self.l2, class_weight=self.class_weight)


CANDIDATES: list[Candidate] = [Candidate("heuristic")] + [
    Candidate(f"softmax(l2={l2:g},cw={cw or 'none'})", l2, cw)
    for l2 in (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    for cw in (None, "balanced")
]


# --------------------------------------------------------------------------------------
# Dropout simulation
# --------------------------------------------------------------------------------------


def _mask_blocks(visible: np.ndarray, target: int, draw_len, rng) -> np.ndarray:
    """Boolean mask of newly hidden rows: random blocks until `target` visible rows are covered.

    `draw_len(rng)` gives a block length; blocks start uniformly. The last block is truncated so
    the count lands exactly on `target` (capped at the number of visible rows).
    """
    length = len(visible)
    masked = np.zeros(length, dtype=bool)
    target = min(int(target), int(visible.sum()))
    newly = 0
    for _ in range(_MAX_BLOCKS):
        if newly >= target:
            break
        blk = min(max(int(draw_len(rng)), 1), length)
        start = int(rng.integers(0, length - blk + 1))
        fresh = np.flatnonzero(visible[start : start + blk] & ~masked[start : start + blk])
        if fresh.size == 0:
            continue
        fresh = fresh[: target - newly]
        masked[start + fresh] = True
        newly += fresh.size
    return masked


def external_gap_lengths(trks: list[Tracking]) -> np.ndarray:
    """Lengths (rows, int64, sorted) of every missing-ball run of at least ``MIN_GAP`` rows inside
    the periods of the given (clean, public sample) games, pooled over the games."""
    out: list[int] = []
    for trk in trks:
        n = len(trk.ball)
        for a, b in trk.periods:
            a, b = max(int(a), 0), min(int(b), n)
            if b <= a:
                continue
            miss = np.isnan(trk.ball[a:b, 0]).astype(np.int8)
            d = np.diff(np.concatenate(([0], miss, [0])))
            out.extend(int(e - s) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
                       if e - s >= MIN_GAP)
    return np.sort(np.asarray(out, dtype=np.int64))


def _gap_tag(ball_lengths) -> str:
    """Short digest of the gap lengths plus the code version; part of every dropout cache key."""
    h = hashlib.sha256()
    h.update(np.asarray(ball_lengths, dtype=np.int64).tobytes())
    h.update(_code_version().encode())
    return h.hexdigest()[:12]


def simulate_dropout(trk: Tracking, rng, ball_lengths) -> Tracking:
    """Copy of `trk` with extra label-independent gaps (inputs are never mutated).

    Ball: inside each period, blocks at uniform starts until the *newly* hidden rows cover 25% of
    the period's rows; each block length is drawn uniformly from the array `ball_lengths` (the
    public sample games' own gap lengths). Players: per slot, blocks of ``int(U(25, 250))`` rows
    until 5% of the period's rows are newly hidden.
    """
    ball_lengths = np.asarray(ball_lengths, dtype=np.int64)
    if ball_lengths.size == 0:
        raise ValueError("ball_lengths is empty: no external gaps to resample")
    ball = trk.ball.copy()
    players = trk.players.copy()
    n = len(ball)
    for a, b in trk.periods:
        a, b = max(int(a), 0), min(int(b), n)
        if b <= a:
            continue
        rows = b - a
        vis = ~np.isnan(ball[a:b, 0])
        gone = _mask_blocks(vis, int(np.ceil(BALL_SHARE * rows)),
                            lambda r: ball_lengths[r.integers(len(ball_lengths))], rng)
        ball[a:b][gone] = np.nan
        for s in range(players.shape[1]):
            vis = ~np.isnan(players[a:b, s, 0])
            gone = _mask_blocks(vis, int(np.ceil(PLAYER_SHARE * rows)),
                                lambda r: r.uniform(*PLAYER_BLOCK), rng)
            players[a:b, s][gone] = np.nan
    return replace(trk, ball=ball, players=players)


# --------------------------------------------------------------------------------------
# Features with an on-disk cache
# --------------------------------------------------------------------------------------


def _code_version() -> str:
    """Digest of the code that makes the cached features: features.py and tracking.py."""
    here = Path(__file__).resolve().parent
    return paths.source_digest(here / "features.py", here / "tracking.py")


def _signature(trk: Tracking) -> str:
    """Digest of the missingness pattern, feature names and code version; guards cached features."""
    h = hashlib.sha256()
    h.update(_code_version().encode())
    h.update(np.packbits(np.isnan(trk.ball[:, 0])).tobytes())
    h.update(np.packbits(np.isnan(trk.players[..., 0])).tobytes())
    h.update(repr((trk.ball.shape, trk.players.shape, trk.periods, FEATURE_NAMES)).encode())
    return h.hexdigest()


def _features(trk: Tracking, name: str, use_cache: bool) -> np.ndarray:
    """`compute_features(trk)`, cached as .cache/<name>.npz when `use_cache`."""
    if not use_cache:
        return compute_features(trk)
    path = paths.cache_dir() / f"{name}.npz"
    sig = _signature(trk)
    if path.exists():
        try:
            with np.load(path) as z:
                if str(z["sig"]) == sig:
                    return z["F"]
        except Exception:  # corrupt or foreign file -> recompute
            pass
    F = compute_features(trk)
    np.savez(path, F=F, sig=np.array(sig))
    return F


def _seed_tag(*key: int) -> str:
    return "_".join(str(int(k)) for k in (SEED, *key))


def _dropout_features(trk, k, key, use_cache, rng, ball_lengths):
    """Dropout-copy features for game k (rng given by the caller); the cache key carries the gaps."""
    copy = simulate_dropout(trk, rng, ball_lengths)
    return _features(copy, f"montreal_sample_g{k}_drop_{_seed_tag(*key)}_{_gap_tag(ball_lengths)}",
                     use_cache)


def _train_rows(F: np.ndarray, labels: np.ndarray):
    idx = np.arange(0, len(F), TRAIN_STRIDE)
    return F[idx], labels[idx]


# --------------------------------------------------------------------------------------
# LOGO
# --------------------------------------------------------------------------------------


def _columns(names) -> list[int]:
    """Column indices in the full feature matrix of the feature names a model uses."""
    return [FEATURE_NAMES.index(n) for n in names]


def _fold(g: int, games, use_cache: bool, names=None, ball_lengths=None) -> list[dict]:
    """Held-out game g (1-based): returns one row per (candidate, grid point).

    `names` is the feature subset the models see (default: all of FEATURE_NAMES); the full matrix
    is what is cached, the subset is cut out by name.
    """
    names = list(FEATURE_NAMES if names is None else names)
    cols = _columns(names)
    held = g - 1
    rng = np.random.default_rng([SEED, g])
    Xs, ys = [], []
    for j, (trk, labels) in enumerate(games):
        if j == held:
            continue
        copy = simulate_dropout(trk, rng, ball_lengths)  # always drawn: the stream never depends on the cache
        F = _features(copy, f"montreal_sample_g{j + 1}_drop_{_seed_tag(g)}_{_gap_tag(ball_lengths)}", use_cache)
        X, y = _train_rows(F[:, cols], labels)
        Xs.append(X)
        ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)

    trk, gt = games[held]
    F_eval = _dropout_features(trk, g, (100 + g,), use_cache, np.random.default_rng([SEED, 100 + g]),
                              ball_lengths)[:, cols]
    periods = trk.periods

    rows = []
    for cand in CANDIDATES:
        if cand.l2 is None:
            logp = heuristic_logp(F_eval, names)
        else:
            logp = cand.make().fit(X, y, feature_names=names).log_proba(F_eval)
        for swap, dead, live in COST_GRID:
            pred = decode_periods(logp, periods, cost_matrix(swap, dead, live))
            r = score(gt, pred)
            rows.append({"candidate": cand.name, "swap": swap, "dead": dead, "live": live,
                         "g": g, "spell_f1": r["spell_f1"], "macro_f1": r["macro_f1"]})
    return rows


def logo(games, cache: bool = False, workers: int = 1, feature_names=None,
         ball_lengths=None) -> pd.DataFrame:
    """Leave-one-game-out table. `games` is a list of (Tracking, label codes) for games 1..N.

    One row per (candidate, swap, dead, live), columns ``spell_f1_g<k>`` / ``macro_f1_g<k>`` for
    the game held out in each fold, plus ``spell_f1_mean`` and ``macro_f1_mean``. Folds are
    independent and are assembled in game order, so `workers` never changes the result.
    """
    games = [(t, np.asarray(lab)) for t, lab in games]
    if ball_lengths is None:  # the gaps of the (clean) games themselves
        ball_lengths = external_gap_lengths([t for t, _ in games])
    folds = list(range(1, len(games) + 1))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_fold, folds, [games] * len(folds), [cache] * len(folds),
                                    [feature_names] * len(folds), [ball_lengths] * len(folds)))
    else:
        results = [_fold(g, games, cache, feature_names, ball_lengths) for g in folds]

    base = [(c.name, *pt) for c in CANDIDATES for pt in COST_GRID]
    table = pd.DataFrame(base, columns=["candidate", "swap", "dead", "live"])
    for g, rows in zip(folds, results):
        assert [(r["candidate"], r["swap"], r["dead"], r["live"]) for r in rows] == base
        table[f"spell_f1_g{g}"] = [r["spell_f1"] for r in rows]
        table[f"macro_f1_g{g}"] = [r["macro_f1"] for r in rows]
    table["spell_f1_mean"] = table[[f"spell_f1_g{g}" for g in folds]].mean(axis=1)
    table["macro_f1_mean"] = table[[f"macro_f1_g{g}" for g in folds]].mean(axis=1)
    return table


def select(table: pd.DataFrame) -> pd.Series:
    """Best row: highest spell_f1_mean, then higher macro_f1_mean, then the earliest row."""
    order = np.lexsort((np.arange(len(table)), -table["macro_f1_mean"].to_numpy(),
                        -table["spell_f1_mean"].to_numpy()))
    return table.iloc[int(order[0])]


def ranked(table: pd.DataFrame) -> pd.DataFrame:
    """Table sorted by the same key as `select` (best first)."""
    order = np.lexsort((np.arange(len(table)), -table["macro_f1_mean"].to_numpy(),
                        -table["spell_f1_mean"].to_numpy()))
    return table.iloc[order]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _num(v) -> int | float:
    """A cost as int when it is whole (10.0 -> 10), else float (2.5); never truncates."""
    v = float(v)
    return int(v) if v.is_integer() else v


def _row_dict(row: pd.Series) -> dict:
    out = {"candidate": str(row["candidate"])}
    for k in ("swap", "dead", "live"):
        out[k] = _num(row[k])
    for k in row.index:
        if k.startswith(("spell_f1_", "macro_f1_")):
            out[k] = round(float(row[k]), 6)
    return out


def _fmt(table: pd.DataFrame) -> str:
    cols = ["candidate", "swap", "dead", "live"] + [
        c for c in table.columns if c.startswith("spell_f1_")] + ["macro_f1_mean"]
    return table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")


def choose_feature_set(tables: dict[str, pd.DataFrame]) -> tuple[str, str]:
    """('v1' | 'v2', decision text). v2 is kept only if its best spell_f1_mean is strictly higher."""
    b1, b2 = select(tables["v1"]), select(tables["v2"])
    s1, s2 = float(b1["spell_f1_mean"]), float(b2["spell_f1_mean"])
    keep = "v2" if s2 > s1 else "v1"
    verb = "higher than" if s2 > s1 else "not higher than"
    text = (f"features {keep} selected: best v2 spell_f1_mean {s2:.4f} is {verb} best v1 {s1:.4f} "
            f"(v1: {b1['candidate']} {_num(b1['swap'])}/{_num(b1['dead'])}/{_num(b1['live'])}; "
            f"v2: {b2['candidate']} {_num(b2['swap'])}/{_num(b2['dead'])}/{_num(b2['live'])})")
    return keep, text


def _refit(games, cand: Candidate, names, ball_lengths) -> dict:
    """Softmax winner refit on all games' dropout copies (per-game seed [SEED, 200 + k])."""
    cols = _columns(names)
    Xs, ys = [], []
    for k, (trk, labels) in enumerate(games, start=1):
        copy = simulate_dropout(trk, np.random.default_rng([SEED, 200 + k]), ball_lengths)
        F = _features(copy, f"montreal_sample_g{k}_drop_{_seed_tag(200 + k)}_{_gap_tag(ball_lengths)}", True)
        X, y = _train_rows(F[:, cols], labels)
        Xs.append(X)
        ys.append(y)
    model = cand.make().fit(np.vstack(Xs), np.concatenate(ys), feature_names=list(names))
    return model.to_dict()


def main(argv=None) -> int:
    from montreal.events import derive_labels, load_sample_game

    repo = paths.repo_dir()
    ap = argparse.ArgumentParser(prog="python -m montreal.train", description=__doc__)
    ap.add_argument("--params", default=str(repo / "models" / "params.json"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-set", choices=["both", "v1", "v2"], default="both",
                    help="both: run LOGO for v1 (41 features) and v1+v2 (51) and keep v2 only if "
                         "its best spell_f1_mean is higher; v1 / v2: force one set")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    games = []
    for k in (1, 2, 3):
        trk, events = load_sample_game(k)
        labels = derive_labels(events, len(trk.ball), trk.periods)
        _features(trk, f"montreal_sample_g{k}_clean", True)  # warm the clean-feature cache
        games.append((trk, labels))
        share = [round(float((labels == c).mean()), 3) for c in range(4)]
        print(f"game {k}: rows={len(labels)} label shares H/A/D/X={share}")
    t_load = time.perf_counter() - t0
    ball_lengths = external_gap_lengths([t for t, _ in games])
    q = [int(v) for v in np.percentile(ball_lengths, [10, 50, 90])]
    gap_summary = {"n": int(len(ball_lengths)), "p10": q[0], "p50": q[1], "p90": q[2],
                   "max": int(ball_lengths.max()), "min_gap_rows": MIN_GAP}
    print(f"external ball-gap lengths (rows, public sample games, >= {MIN_GAP}): {gap_summary}")

    sets = ["v1", "v2"] if args.feature_set == "both" else [args.feature_set]
    tables: dict[str, pd.DataFrame] = {}
    for fs in sets:
        t1 = time.perf_counter()
        tables[fs] = table = logo(games, cache=True, workers=args.workers,
                                  feature_names=FEATURE_SETS[fs], ball_lengths=ball_lengths)
        t_logo = time.perf_counter() - t1
        n_eval = len(table) * len(games)
        print(f"\n=== features {fs} ({len(FEATURE_SETS[fs])} columns) ===")
        print(f"LOGO: {len(table)} rows x {len(games)} folds = {n_eval} decode+score evaluations "
              f"in {t_logo:.1f} s ({t_logo / n_eval:.3f} s each, incl. fits)")
        print(f"Top 5 (LOGO, features {fs}, sorted by spell_f1_mean):")
        print(_fmt(ranked(table).head(5)))
        heur = table[table["candidate"] == "heuristic"]
        print(f"Heuristic best (features {fs}):")
        print(_fmt(heur.loc[[select(heur).name]]))

    if len(sets) == 2:
        chosen, decision = choose_feature_set(tables)
    else:
        chosen, decision = sets[0], (f"features {sets[0]} forced by --feature-set; the v1-vs-v2 decision "
                                     "was made earlier by LOGO on the public sample games (external data)")
    print(f"\ndecision: {decision}")

    table = tables[chosen]
    top = ranked(table)
    heur_best = select(table[table["candidate"] == "heuristic"])
    best = select(table)
    cand = next(c for c in CANDIDATES if c.name == best["candidate"])
    names = list(FEATURE_SETS[chosen])
    emission = "heuristic" if cand.l2 is None else _refit(games, cand, names, ball_lengths)
    costs = {"swap": _num(best["swap"]), "dead": _num(best["dead"]), "live": _num(best["live"])}
    params = {
        "emission": emission,
        "costs": costs,
        "use_tags": True,
        "meta": {
            "source": "LOGO on Metrica sample games 1-3",
            "sample_commit": paths.SAMPLE_COMMIT,
            "seed": SEED,
            "feature_set": chosen,
            "n_features": len(names),
            "decision": decision,
            "feature_set_decision_data": "external only: LOGO on the public Metrica sample games (v1 vs v2)",
            "dropout_ball_gaps": gap_summary,
            "logo_best": {fs: _row_dict(select(tables[fs])) for fs in sets},
            "top5_by_feature_set": {fs: [_row_dict(r) for _, r in ranked(tables[fs]).head(5).iterrows()]
                                    for fs in sets},
            "selected": {
                "candidate": cand.name,
                "costs": dict(costs),
                "spell_f1_mean": round(float(best["spell_f1_mean"]), 6),
                "macro_f1_mean": round(float(best["macro_f1_mean"]), 6),
            },
            "top5": [_row_dict(r) for _, r in top.head(5).iterrows()],
            "heuristic_best": _row_dict(heur_best),
        },
    }
    out = Path(args.params)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected: {cand.name} features={chosen} swap={costs['swap']} dead={costs['dead']} "
          f"live={costs['live']} spell_f1_mean={best['spell_f1_mean']:.4f} "
          f"macro_f1_mean={best['macro_f1_mean']:.4f}")
    print(f"wrote: {out}")
    print(f"runtime: load+features {t_load:.1f} s, total {time.perf_counter() - t0:.1f} s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
