"""Smart Tagging tests -- synthetic SportsCode XML only."""

import numpy as np

from montreal.tagging import (calibrate, confident_team, goal_dead, load_tags, tag_prior,
                              tag_votes)

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


def test_tag_prior():
    votes = {"attack": np.array([1, -1, 0, 1], dtype=np.int8),
             "defend": np.array([1, 0, 0, -1], dtype=np.int8)}
    goal = np.array([False, False, True, True])
    p = tag_prior(votes, goal, {"attack": 0.5, "defend": 0.25}, w_goal=1.0)
    assert p.dtype == np.float64 and p.shape == (4, 3)
    np.testing.assert_allclose(p, [[0.75, 0, 0], [0, 0.5, 0], [0, 0, 1.0], [0.5, 0.25, 1.0]])
