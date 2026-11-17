"""Smart Tagging tests -- synthetic SportsCode XML only."""

import numpy as np
import pytest

from montreal.tagging import (apply_tag_prior, calibrate, calibration_stats, check_orientation,
                              confident_team, goal_dead, load_tags, tag_votes)

NAMES = ["ball_vis", "d_home", "d_away"]


def _instance(code, start, end, labels=()):
    lab = "".join(f"<label><group>{g}</group><text>{t}</text></label>" for g, t in labels)
    return (f"<instance><ID>1</ID><start>{start}</start><end>{end}</end>"
            f"<code>{code}</code>{lab}</instance>")


def _write(tmp_path, instances):
    path = tmp_path / "tags.xml"
    path.write_text("<file><ALL_INSTANCES>" + "".join(instances) + "</ALL_INSTANCES></file>")
    return path


def test_team_mapping(tmp_path):
    path = _write(tmp_path, [
        _instance(" build up ", 1.0, 3.0, [("Team", "Barcelona")]),
        _instance("BUILD UP", 4.0, 5.0, [("Team", "ATLETIC CLUB")]),
        _instance("BUILD UP", 6.0, 7.0, [("Team", "N/A")]),
        _instance("BUILD UP", 8.0, 9.0),
        _instance("SET PIECES", 10.0, 11.0, [("Team", "Athletic Club"), ("Type", "Corner Kick")]),
    ])
    tags = load_tags(path)
    assert [t["team"] for t in tags] == ["H", "A", None, None, "A"]
    assert tags[0]["code"] == "BUILD UP"
    assert (tags[0]["start"], tags[0]["stop"]) == (25, 75)
    assert tags[3]["type"] == "" and tags[4]["type"] == "corner kick"

    votes = tag_votes(tags, 400)
    assert votes["attack"].dtype == np.int8
    assert votes["attack"][30] == 1 and votes["attack"][110] == -1
    assert votes["attack"][160] == 0 and votes["attack"][210] == 0
    assert votes["setpiece"][260] == -1


def test_defend_implies_other_team(tmp_path):
    path = _write(tmp_path, [
        _instance("DEFENDING IN MIDDLE THIRD", 1.0, 2.0, [("Team", "Barcelona")]),
        _instance("DEFENSIVE TRANSITION", 3.0, 4.0, [("Team", "Atletic Club")]),
    ])
    votes = tag_votes(load_tags(path), 150)
    assert votes["defend"][30] == -1
    assert votes["defend"][80] == 1
    assert not votes["attack"].any()


def test_overlap_summed_then_sign(tmp_path):
    path = _write(tmp_path, [
        _instance("BUILD UP", 0.0, 4.0, [("Team", "Barcelona")]),
        _instance("PROGRESSION", 0.0, 4.0, [("Team", "Barcelona")]),
        _instance("LONG BALL", 2.0, 4.0, [("Team", "Atletic Club")]),
        _instance("CREATING CHANCES", 3.0, 4.0, [("Team", "Atletic Club")]),
    ])
    v = tag_votes(load_tags(path), 100)["attack"]
    assert v[10] == 1 and v[60] == 1 and v[99] == 0  # 2 H vs 1 A -> +1; 2 H vs 2 A -> 0


def test_goal_dead_until_kickoff(tmp_path):
    path = _write(tmp_path, [
        _instance("GOALS", 10.0, 12.0),
        _instance("SET PIECES", 5.0, 6.0, [("Type", "Kick Off (Start)")]),
        _instance("SET PIECES", 20.0, 21.0, [("Type", "Corner Kick")]),
        _instance("SET PIECES", 40.0, 41.0, [("Type", "Kick Off (After Goal)")]),
    ])
    mask = goal_dead(load_tags(path), 3000)
    assert mask.dtype == bool
    assert np.flatnonzero(mask).tolist() == list(range(250, 1000))


def test_goal_dead_cap_and_clip(tmp_path):
    path = _write(tmp_path, [_instance("GOALS", 10.0, 12.0)])
    mask = goal_dead(load_tags(path), 3000)
    assert np.flatnonzero(mask).tolist() == list(range(250, 250 + 2250))
    short = goal_dead(load_tags(path), 400)
    assert np.flatnonzero(short).tolist() == list(range(250, 400))


def test_calibrate_cap_and_minimum():
    conf = np.ones(1000, dtype=np.int8)
    assert calibrate({"attack": np.ones(1000, dtype=np.int8)}, conf) == {"attack": 1.0}
    assert calibrate({"attack": np.ones(100, dtype=np.int8)}, conf[:100])["attack"] == 0.0
    half = np.tile(np.array([1, -1], dtype=np.int8), 500)
    assert calibrate({"attack": half}, conf)["attack"] == 0.0
    w = calibrate({"a": np.ones(1000, dtype=np.int8), "b": half}, conf)
    assert set(w) == {"a", "b"}


def test_calibrate_below_chance_is_zero():
    conf = np.ones(1000, dtype=np.int8)
    vote = -np.ones(1000, dtype=np.int8)
    assert calibrate({"x": vote}, conf)["x"] == 0.0


def _F(n, rows):
    F = np.zeros((n, 3))
    F[:, 1:] = 10.0
    for a, b, vis, dh, da in rows:
        F[a:b] = [vis, dh, da]
    return F


def test_confident_run_length():
    F = _F(200, [(10, 34, 1, 0.5, 5.0), (50, 75, 1, 0.5, 5.0), (100, 140, 1, 5.0, 0.5)])
    c = confident_team(F, NAMES)
    assert c.dtype == np.int8
    assert not c[10:34].any()
    assert (c[50:75] == 1).all() and c[75] == 0 and c[49] == 0
    assert (c[100:140] == -1).all()


def test_confident_requires_ball_visible_and_separation():
    F = _F(100, [(0, 50, 0, 0.5, 5.0), (50, 100, 1, 0.5, 2.9)])
    assert not confident_team(F, NAMES).any()


def _rand_logp(n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, size=(n, 3))
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def _vote_args(n, seed=1):
    rng = np.random.default_rng(seed)
    votes = {k: rng.integers(-1, 2, size=n).astype(np.int8) for k in ("attack", "defend", "setpiece")}
    weights = {"attack": 1.0, "defend": 0.7, "setpiece": 0.566}
    goal = rng.random(n) < 0.2
    return votes, weights, goal


def test_apply_tag_prior_keeps_dead_score_and_alive_mass():
    n = 500
    logp = _rand_logp(n)
    votes, weights, goal = _vote_args(n)
    out = apply_tag_prior(logp, votes, weights, np.zeros(n, dtype=bool))
    assert out.shape == (n, 3) and out.dtype == np.float64
    np.testing.assert_array_equal(out[:, 2], logp[:, 2])  # D is untouched by any vote
    np.testing.assert_allclose(np.logaddexp(out[:, 0], out[:, 1]),
                               np.logaddexp(logp[:, 0], logp[:, 1]), atol=1e-12, rtol=0)
    # the input is not mutated
    assert not np.shares_memory(out, logp)


def test_apply_tag_prior_net_vote_shifts_h_minus_a_and_caps_at_one():
    logp = _rand_logp(6, seed=3)
    z0 = logp[:, 0] - logp[:, 1]
    votes = {"attack": np.array([1, 1, -1, 0, 1, -1], dtype=np.int8),
             "defend": np.array([0, 1, -1, 0, 1, 1], dtype=np.int8),
             "setpiece": np.array([0, 0, 0, 0, 1, -1], dtype=np.int8)}
    weights = {"attack": 0.5, "defend": 0.25, "setpiece": 0.4}
    out = apply_tag_prior(logp, votes, weights, np.zeros(6, dtype=bool))
    shift = (out[:, 0] - out[:, 1]) - z0
    # +0.5; 0.75; -0.75; 0; 0.5+0.25+0.4 = 1.15 -> capped 1.0; 0.5-... = -0.5+0.25-0.4 = -0.65
    np.testing.assert_allclose(shift, [0.5, 0.75, -0.75, 0.0, 1.0, -0.65], atol=1e-12)
    # +1 vote raises H - A by exactly min(w * vote, 1)
    one = apply_tag_prior(logp, {"attack": np.ones(6, dtype=np.int8)}, {"attack": 0.3},
                          np.zeros(6, dtype=bool))
    np.testing.assert_allclose((one[:, 0] - one[:, 1]) - z0, 0.3, atol=1e-12)
    cap = apply_tag_prior(logp, {"attack": np.ones(6, dtype=np.int8)}, {"attack": 2.5},
                          np.zeros(6, dtype=bool))
    np.testing.assert_allclose((cap[:, 0] - cap[:, 1]) - z0, 1.0, atol=1e-12)
    capped_down = apply_tag_prior(logp, {"attack": -np.ones(6, dtype=np.int8)}, {"attack": 2.5},
                                  np.zeros(6, dtype=bool), v_cap=0.5)
    np.testing.assert_allclose((capped_down[:, 0] - capped_down[:, 1]) - z0, -0.5, atol=1e-12)


def test_apply_tag_prior_goal_mask_adds_to_d_only():
    logp = _rand_logp(4, seed=5)
    goal = np.array([False, True, False, True])
    out = apply_tag_prior(logp, {}, {}, goal, w_goal=1.5)
    np.testing.assert_allclose(out[:, 2], logp[:, 2] + 1.5 * goal, atol=1e-12)
    np.testing.assert_allclose(out[:, :2], logp[:, :2], atol=1e-12)


def test_apply_tag_prior_zero_votes_is_identity_and_stable_for_extreme_logits():
    logp = _rand_logp(50, seed=7)
    votes, weights, goal = _vote_args(50)
    zero = {k: np.zeros(50, dtype=np.int8) for k in votes}
    np.testing.assert_allclose(apply_tag_prior(logp, zero, weights, np.zeros(50, dtype=bool)),
                               logp, atol=1e-12)
    ext = np.array([[-800.0, -1.0, -2.0], [-1.0, -900.0, -3.0], [-0.5, -0.6, -1000.0]])
    out = apply_tag_prior(ext, {"attack": np.array([1, -1, 1], dtype=np.int8)}, {"attack": 1.0},
                          np.zeros(3, dtype=bool))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.logaddexp(out[:, 0], out[:, 1]),
                               np.logaddexp(ext[:, 0], ext[:, 1]), atol=1e-9)


def test_calibration_stats_report_agreement_and_support():
    conf = np.ones(1000, dtype=np.int8)
    vote = np.ones(1000, dtype=np.int8)
    vote[:100] = -1  # 90% agreement on 1000 supported frames
    stats = calibration_stats({"attack": vote}, conf)
    stats.update(calibration_stats({"tiny": np.ones(100, dtype=np.int8)}, conf[:100]))
    assert stats["attack"]["n"] == 1000 and abs(stats["attack"]["p"] - 0.9) < 1e-12
    assert stats["tiny"]["n"] == 100 and stats["tiny"]["p"] == 1.0
    empty = calibration_stats({"x": np.zeros(10, dtype=np.int8)}, np.ones(10, dtype=np.int8))
    assert empty["x"]["n"] == 0 and np.isnan(empty["x"]["p"])
    # calibrate still gives the capped log-odds from the same agreement
    assert calibrate({"attack": vote}, conf)["attack"] == 1.0


def test_orientation_guard_fails_when_attack_agreement_below_half():
    ok = {"attack": {"p": 0.73, "n": 4000}, "defend": {"p": 0.3, "n": 5000}}
    check_orientation(ok)  # the guard looks at the attack family only
    check_orientation({"attack": {"p": 0.2, "n": 249}})  # too little support to judge
    check_orientation({"attack": {"p": 0.5, "n": 4000}})  # exactly 0.5 is not flipped
    with pytest.raises(RuntimeError, match="flipped"):
        check_orientation({"attack": {"p": 0.2, "n": 250}})
    check_orientation({})  # no attack family: nothing to check


def test_load_tags_skips_instances_without_start_or_end(tmp_path):
    path = tmp_path / "tags.xml"
    path.write_text(
        "<file><ALL_INSTANCES>"
        "<instance><ID>1</ID><end>3.0</end><code>BUILD UP</code></instance>"
        "<instance><ID>2</ID><start>1.0</start><code>BUILD UP</code></instance>"
        "<instance><ID>3</ID><start></start><end>2.0</end><code>BUILD UP</code></instance>"
        + _instance("BUILD UP", 1.0, 3.0, [("Team", "Barcelona")]) +
        "</ALL_INSTANCES></file>")
    tags = load_tags(path)
    assert len(tags) == 1 and (tags[0]["start"], tags[0]["stop"]) == (25, 75)