"""Event loading and possession label derivation: synthetic event frames only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from montreal import events, paths
from montreal.events import derive_labels

H, A, D, X = 0, 1, 2, 3
N = 100
PERIODS = [(0, 50), (60, 100)]


def _ev(*rows):
    """rows: (team, type, start[, end[, subtype]])."""
    out = []
    for r in rows:
        team, typ, start = r[0], r[1], r[2]
        end = r[3] if len(r) > 3 else start
        sub = r[4] if len(r) > 4 else ""
        out.append(
            {"team": team, "type": typ, "subtype": sub, "start": start, "end": end, "period": 1}
        )
    return pd.DataFrame(out, columns=["team", "type", "subtype", "start", "end", "period"])


def test_kickoff_then_switch():
    ev = _ev(
        ("H", "SET PIECE", 0),
        ("H", "PASS", 5),
        ("A", "RECOVERY", 20),
        ("A", "PASS", 22),
    )
    labels = derive_labels(ev, N, PERIODS)
    assert labels.dtype == np.int8 and len(labels) == N
    assert (labels[0:20] == H).all()
    assert (labels[20:50] == A).all()
    assert (labels[50:60] == X).all()
    assert (labels[60:] == D).all()


def test_period_starts_dead_until_first_event():
    labels = derive_labels(_ev(("A", "PASS", 10)), N, PERIODS)
    assert (labels[0:10] == D).all()
    assert (labels[10:50] == A).all()


def test_challenge_does_not_switch():
    ev = _ev(
        ("H", "SET PIECE", 0),
        ("A", "CHALLENGE", 10),
        ("A", "BALL LOST", 11),
        ("A", "CARD", 11),
        ("H", "PASS", 12),
    )
    labels = derive_labels(ev, N, PERIODS)
    assert (labels[0:50] == H).all()


def test_ball_out_until_restart():
    ev = _ev(
        ("H", "SET PIECE", 0),
        ("H", "BALL OUT", 30),
        ("A", "SET PIECE", 40),
    )
    labels = derive_labels(ev, N, PERIODS)
    assert (labels[0:30] == H).all()
    assert (labels[30:40] == D).all()
    assert (labels[40:50] == A).all()


def test_fault_received_is_dead():
    ev = _ev(("H", "PASS", 0), ("A", "FAULT RECEIVED", 20), ("A", "SET PIECE", 25))
    labels = derive_labels(ev, N, PERIODS)
    assert (labels[20:25] == D).all()
    assert (labels[25:50] == A).all()


def test_goal_shot_dead_from_end():
    ev = _ev(
        ("H", "PASS", 0),
        ("H", "SHOT", 10, 14, "ON TARGET-GOAL"),
        ("A", "SET PIECE", 30),
    )
    labels = derive_labels(ev, N, PERIODS)
    assert (labels[0:14] == H).all()
    assert (labels[14:30] == D).all()
    assert (labels[30:50] == A).all()


def test_shot_out_is_dead_but_saved_shot_is_not():
    out = derive_labels(
        _ev(("H", "PASS", 0), ("H", "SHOT", 10, 12, "OFF TARGET-OUT"), ("A", "PASS", 30)),
        N,
        PERIODS,
    )
    assert (out[12:30] == D).all()
    saved = derive_labels(
        _ev(("H", "PASS", 0), ("H", "SHOT", 10, 12, "ON TARGET-SAVED"), ("A", "RECOVERY", 30)),
        N,
        PERIODS,
    )
    assert (saved[0:30] == H).all() and (saved[30:50] == A).all()


def test_switch_and_dead_ordered_by_effective_time():
    # the goal is only dead at its end frame, so a recovery before it still counts
    ev = _ev(
        ("H", "SHOT", 5, 20, "ON TARGET-GOAL"),
        ("A", "RECOVERY", 10),
        ("H", "SET PIECE", 35),
    )
    labels = derive_labels(ev, N, PERIODS)
    assert (labels[0:5] == D).all()
    assert (labels[5:10] == H).all()
    assert (labels[10:20] == A).all()
    assert (labels[20:35] == D).all()
    assert (labels[35:50] == H).all()


def test_state_does_not_leak_across_periods():
    ev = _ev(("H", "PASS", 0), ("A", "PASS", 70))
    labels = derive_labels(ev, N, PERIODS)
    assert (labels[0:50] == H).all()
    assert (labels[60:70] == D).all()
    assert (labels[70:] == A).all()


def test_odd_frames():
    ev = _ev(
        ("H", "SET PIECE", 0, -1, "KICK OFF"),  # CSV End Frame 0 -> -1
        ("H", "PASS", 120),  # beyond n
        ("A", "PASS", 55),  # in the gap between periods
        ("A", "SHOT", 40, 30, "ON TARGET-GOAL"),  # end < start
        ("A", "PASS", -3),  # before the first row
    )
    labels = derive_labels(ev, N, PERIODS)
    assert len(labels) == N
    assert (labels[50:60] == X).all()
    assert (labels[0:40] == H).all()
    assert (labels[40:50] == D).all()
    assert (labels[60:] == D).all()


def test_shot_dead_trigger_beyond_period_is_clipped():
    ev = _ev(("H", "SHOT", 45, 200, "HEAD-WOODWORK-OUT"))
    labels = derive_labels(ev, N, PERIODS)
    assert len(labels) == N
    assert (labels[45:50] == H).all()
    assert (labels[50:60] == X).all()


def test_empty_events():
    labels = derive_labels(_ev(), N, PERIODS)
    assert (labels[0:50] == D).all() and (labels[50:60] == X).all() and (labels[60:] == D).all()


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2, 3])
def test_load_sample_game(k):
    if not (paths.sample_dir() / "data" / f"Sample_Game_{k}").exists():
        pytest.skip("sample data not fetched")
    t, ev = events.load_sample_game(k)
    assert t.pitch == (105.0, 68.0)
    assert list(ev.columns) == ["team", "type", "subtype", "start", "end", "period"]
    assert set(ev["team"]) == {"H", "A"}
    assert ev["type"].str.isupper().all()
    assert ev["subtype"].notna().all()
    assert ev["start"].is_monotonic_increasing
    assert ev["start"].dtype.kind == "i" and ev["period"].dtype.kind == "i"
    labels = derive_labels(ev, len(t.ball), t.periods)
    assert len(labels) == len(t.ball)
