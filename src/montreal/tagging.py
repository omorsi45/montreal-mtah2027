"""Smart Tagging (SportsCode XML) as a capped, calibrated team-only adjustment of the log-scores.

Smart Tagging is Metrica's automatic annotation of tactical phases and set pieces. It is *not*
ground truth. Each tag family is weighted by how often its team vote agrees with confident
tracking evidence (ball visible, one team's nearest player at the ball and the other far away) --
never by agreement with any label. Weights are capped, and the net vote is capped too, so the\ntags can only nudge H against A; they never move a frame between alive and dead (except the\nadditive goal-dead term).

Codes H=0, A=1, D=2.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

ATTACK = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL",
}
DEFEND = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION",
}
SETPIECE = {"SET PIECES"}
GOAL = {"GOALS"}

TEAM_H = {"barcelona"}
TEAM_A = {"atletic club", "athletic club"}

GOAL_CAP_SECONDS = 90
MIN_RUN = 25
MIN_CALIBRATION_FRAMES = 250
WEIGHT_CAP = 1.0
P_EPS = 1e-6  # Ruling R5: clip p before the log-odds


def _team(text: str | None) -> str | None:
    key = (text or "").strip().lower()
    if key in TEAM_H:
        return "H"
    if key in TEAM_A:
        return "A"
    return None


def load_tags(path, fps: int = 25) -> list[dict]:
    """Read the SportsCode XML. Frames are ``round(seconds * fps)``, half-open ``[start, stop)``."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        try:
            start_s = float(instance.findtext("start"))
            end_s = float(instance.findtext("end"))
        except (TypeError, ValueError):  # no <start> or <end> (or not a number): skip the instance
            continue
        team_text = None
        type_text = ""
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip().lower()
            if group == "team":
                team_text = label.findtext("text")
            elif group == "type":
                type_text = (label.findtext("text") or "").strip().lower()
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start": int(round(start_s * fps)),
            "stop": int(round(end_s * fps)),
            "team": _team(team_text),
            "type": type_text,
        })
    return out


def _span(tag: dict, n: int) -> tuple[int, int]:
    a = min(max(int(tag["start"]), 0), n)
    b = min(max(int(tag["stop"]), 0), n)
    return a, b


def tag_votes(tags: list[dict], n: int) -> dict[str, np.ndarray]:
    """Per-frame team vote of each family: +1 implies H, -1 implies A, 0 no evidence (int8 (n,)).

    A defend tag implies the other team. Overlapping same-family votes are summed, then the sign
    is taken. Tags without a known team contribute nothing.
    """
    families = {"attack": (ATTACK, 1), "defend": (DEFEND, -1), "setpiece": (SETPIECE, 1)}
    votes = {}
    for name, (codes, orient) in families.items():
        acc = np.zeros(n, dtype=np.int32)
        for tag in tags:
            if tag["code"] not in codes or tag["team"] is None:
                continue
            a, b = _span(tag, n)
            if b <= a:
                continue
            acc[a:b] += orient * (1 if tag["team"] == "H" else -1)
        votes[name] = np.sign(acc).astype(np.int8)
    return votes


def goal_dead(tags: list[dict], n: int, fps: int = 25) -> np.ndarray:
    """True from each GOALS start to the next kick-off set piece start, capped at 90 s."""
    cap = GOAL_CAP_SECONDS * fps
    kickoffs = sorted(t["start"] for t in tags if t["code"] in SETPIECE and "kick off" in t["type"])
    mask = np.zeros(n, dtype=bool)
    for tag in tags:
        if tag["code"] not in GOAL:
            continue
        g = tag["start"]
        end = g + cap
        nxt = next((k for k in kickoffs if k > g), None)
        if nxt is not None:
            end = min(nxt, g + cap)
        a = min(max(g, 0), n)
        b = min(max(end, 0), n)
        if b > a:
            mask[a:b] = True
    return mask


def _long_runs(cond: np.ndarray, min_len: int) -> np.ndarray:
    """Boolean mask of the rows belonging to a run of True at least ``min_len`` long."""
    n = len(cond)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    padded = np.concatenate(([False], cond, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    for a, b in zip(starts, stops):
        if b - a >= min_len:
            out[a:b] = True
    return out


def confident_team(F: np.ndarray, names: list[str]) -> np.ndarray:
    """+1/-1 where the tracking is unambiguous about who has the ball, else 0 (int8 (n,)).

    +1: ball visible, a home player within 1 m and every away player >= 3 m away, held for at
    least 25 consecutive rows; -1 is the mirror.
    """
    vis = F[:, names.index("ball_vis")] == 1
    dh = F[:, names.index("d_home")]
    da = F[:, names.index("d_away")]
    home = vis & (dh <= 1.0) & (da >= 3.0)
    away = vis & (da <= 1.0) & (dh >= 3.0)
    out = np.zeros(len(F), dtype=np.int8)
    out[_long_runs(home, MIN_RUN)] = 1
    out[_long_runs(away, MIN_RUN)] = -1
    return out


def calibration_stats(votes: dict[str, np.ndarray], confident: np.ndarray) -> dict[str, dict]:
    """Per family: ``n`` = frames with both a vote and a confident tracking team, ``p`` = the raw
    share of them where the two agree (NaN when ``n`` is 0)."""
    stats = {}
    for family, vote in votes.items():
        both = (vote != 0) & (confident != 0)
        n = int(both.sum())
        agree = int((np.sign(vote[both]) == confident[both]).sum())
        stats[family] = {"n": n, "p": agree / n if n else float("nan")}
    return stats


def check_orientation(stats: dict[str, dict]) -> None:
    """Fail loudly if the attack tags agree with the tracking less than chance (H and A flipped)."""
    s = stats.get("attack")
    if s and s["n"] >= MIN_CALIBRATION_FRAMES and s["p"] < 0.5:
        raise RuntimeError(
            f"Smart Tagging attack tags agree with the tracking in only {s['p']:.3f} of "
            f"{s['n']} confident frames: home and away look flipped between the tags and the "
            "tracking. Check the team mapping before running.")


def calibrate(votes: dict[str, np.ndarray], confident: np.ndarray) -> dict[str, float]:
    """Weight per family: capped log-odds of agreeing with the confident tracking evidence."""
    weights = {}
    for family, s in calibration_stats(votes, confident).items():
        if s["n"] < MIN_CALIBRATION_FRAMES:
            weights[family] = 0.0
            continue
        p = min(max(s["p"], P_EPS), 1.0 - P_EPS)
        weights[family] = float(np.clip(np.log(p / (1.0 - p)), 0.0, WEIGHT_CAP))
    return weights


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(1 / (1 + exp(-x))), stable for any x."""
    return -np.logaddexp(0.0, -x)


def apply_tag_prior(logp: np.ndarray, votes: dict[str, np.ndarray], weights: dict[str, float],
                    goal_mask: np.ndarray, w_goal: float = 1.0, v_cap: float = 1.0) -> np.ndarray:
    """Team-only Smart Tagging adjustment of the (n, 3) log-probabilities over H/A/D.

    The calibrated weights only measure H-versus-A agreement (on frames where play is alive), so
    the votes may only move H against A. Each frame keeps its alive mass ``logaddexp(H, A)`` and
    its D score. With ``z = H - A``, ``alive = logaddexp(H, A)`` and the net signed vote
    ``v = clip(sum_f w_f * vote_f, -v_cap, v_cap)``::

        H' = alive + log sigmoid(z + v)      A' = alive + log sigmoid(-(z + v))      D' = D

    so ``H' - A' = z + v`` and ``logaddexp(H', A') = alive``. Attack and defend tags describe the
    same phase from both sides, hence the cap on the *net* vote. Set-piece votes therefore only say
    which team holds the ball once play is alive; they cannot pull a stoppage towards alive. The goal
    mask adds ``w_goal`` to the D column (additive, as before). The input is not modified.
    """
    logp = np.asarray(logp, dtype=np.float64)
    n = len(logp)
    v = np.zeros(n, dtype=np.float64)
    for family, vote in votes.items():
        v += float(weights.get(family, 0.0)) * np.asarray(vote, dtype=np.float64)
    v = np.clip(v, -float(v_cap), float(v_cap))

    z = logp[:, 0] - logp[:, 1]
    alive = np.logaddexp(logp[:, 0], logp[:, 1])
    out = logp.copy()
    out[:, 0] = alive + _log_sigmoid(z + v)
    out[:, 1] = alive + _log_sigmoid(-(z + v))
    out[:, 2] = logp[:, 2] + float(w_goal) * np.asarray(goal_mask, dtype=bool)
    return out