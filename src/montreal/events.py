"""Metrica sample games: load tracking + events, and derive per-frame H/A/D/X possession labels.

The label rules follow football possession logic (dead ball, restart, change of team on the ball);
they are fixed by the spec and must not be tuned.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from montreal import paths
from montreal.tracking import Tracking, read_epts, read_metrica_csv

H, A, D, X = 0, 1, 2, 3
PITCH = (105.0, 68.0)
COLUMNS = ["team", "type", "subtype", "start", "end", "period"]

DEAD_AT_START = {"BALL OUT", "FAULT RECEIVED"}
# Types that restart play while dead and hand possession over while alive (rules 3 and 4 share
# the same list; CHALLENGE, BALL LOST, CARD and anything else never act on possession).
RESTART_TYPES = {"SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY"}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _frame(events: pd.DataFrame) -> pd.DataFrame:
    out = events[COLUMNS].copy()
    for col in ("start", "end", "period"):
        out[col] = out[col].astype(np.int64)
    out = out.sort_values("start", kind="stable").reset_index(drop=True)
    return out


def _load_csv_events(path) -> pd.DataFrame:
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    team = raw["Team"].astype(str).str.strip().str.upper().map({"HOME": "H", "AWAY": "A"})
    sub = raw["Subtype"].fillna("").astype(str).str.strip().str.upper()
    frame = pd.DataFrame(
        {
            "team": team,
            "type": raw["Type"].astype(str).str.strip().str.upper(),
            "subtype": sub,
            "start": raw["Start Frame"].astype(np.int64) - 1,
            "end": raw["End Frame"].fillna(0).astype(np.int64) - 1,  # End Frame 0 -> -1
            "period": raw["Period"].astype(np.int64),
        }
    )
    return _frame(frame.dropna(subset=["team"]))


def _first_team_id(meta_path) -> str:
    for el in ET.parse(str(meta_path)).getroot().iter():
        if el.tag.rsplit("}", 1)[-1] == "Team" and "id" in el.attrib:
            return el.attrib["id"]
    raise ValueError("no <Team id=...> in metadata")


def _subtype_text(sub) -> str:
    if sub is None:
        return ""
    if isinstance(sub, dict):
        return str(sub.get("name", "")).strip().upper()
    return "-".join(str(s.get("name", "")).strip().upper() for s in sub)


def _load_json_events(path, home_id: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)["data"]
    rows = []
    for e in data:
        start = e["start"]["frame"]
        end = e["end"]["frame"]
        if start is None:
            continue
        rows.append(
            {
                "team": "H" if e["team"]["id"] == home_id else "A",
                "type": str(e["type"]["name"]).strip().upper(),
                "subtype": _subtype_text(e.get("subtypes")),
                "start": int(start) - 1,
                "end": (int(end) if end is not None else int(start)) - 1,
                "period": int(e["period"]),
            }
        )
    return _frame(pd.DataFrame(rows, columns=COLUMNS))


def load_sample_game(k: int) -> tuple[Tracking, pd.DataFrame]:
    """Load Metrica public sample game k (1, 2 or 3): tracking and a normalised events frame."""
    if k not in (1, 2, 3):
        raise ValueError(f"sample game must be 1, 2 or 3, got {k!r}")
    folder = paths.sample_dir() / "data" / f"Sample_Game_{k}"
    if k in (1, 2):
        tracking = read_metrica_csv(
            folder / f"Sample_Game_{k}_RawTrackingData_Home_Team.csv",
            folder / f"Sample_Game_{k}_RawTrackingData_Away_Team.csv",
            PITCH,
        )
        return tracking, _load_csv_events(folder / f"Sample_Game_{k}_RawEventsData.csv")
    meta = folder / "Sample_Game_3_metadata.xml"
    tracking = read_epts(meta, folder / "Sample_Game_3_tracking.txt", PITCH)
    return tracking, _load_json_events(folder / "Sample_Game_3_events.json", _first_team_id(meta))


# --------------------------------------------------------------------------------------
# Label derivation
# --------------------------------------------------------------------------------------


def _is_dead_shot(typ: str, sub: str) -> bool:
    return typ == "SHOT" and (sub.endswith("GOAL") or sub.endswith("-OUT"))


def derive_labels(events: pd.DataFrame, n: int, periods: list[tuple[int, int]]) -> np.ndarray:
    """Per-frame possession codes (H=0, A=1, D=2, X=3) from an events frame, int8 of length n.

    Per period: starts dead; BALL OUT / FAULT RECEIVED (at start) and SHOT ...GOAL / ...-OUT (at
    max(start, end)) make the ball dead; the first SET PIECE/PASS/CARRY/SHOT/RECOVERY while dead
    restarts play for that team (strictly after the dead trigger's own frame); while alive, the
    first PASS/CARRY/SHOT/SET PIECE/RECOVERY by the
    other team hands possession over. Everything outside the periods is X.
    """
    labels = np.full(n, X, dtype=np.int8)
    if len(events):
        team = events["team"].to_numpy()
        typ = events["type"].to_numpy()
        sub = events["subtype"].to_numpy() if "subtype" in events else np.full(len(events), "")
        start = events["start"].to_numpy(dtype=np.int64)
        end = np.maximum(events["end"].to_numpy(dtype=np.int64), start)  # end < start -> start
        order = np.argsort(start, kind="stable")
    else:
        team = typ = sub = start = end = np.empty(0)
        order = np.empty(0, dtype=np.int64)

    for p_start, p_stop in periods:
        p_start = max(int(p_start), 0)
        p_stop = min(int(p_stop), n)
        if p_stop <= p_start:
            continue
        # actions in this period: (time, event position, kind, team); kind 0 = event, 1 = dead
        actions = []
        for pos, i in enumerate(order):
            s = int(start[i])
            if s < p_start or s >= p_stop:
                continue
            t, ty = team[i], str(typ[i])
            if ty in DEAD_AT_START:
                actions.append((s, pos, 1, None))
                continue
            if ty in RESTART_TYPES:
                actions.append((s, pos, 0, t))
            if _is_dead_shot(ty, str(sub[i])):
                actions.append((int(end[i]), pos, 1, None))
        actions.sort(key=lambda a: a[:3])

        state = D
        last = p_start
        trigger = -1  # frame of the dead trigger that started the current D run (-1: period start)
        for time, _, kind, t in actions:
            time = min(max(time, last), p_stop)
            if kind == 1:
                new = D
            elif state == D and time <= trigger:
                new = D  # an event at the trigger frame never restarts play
            else:
                code = H if t == "H" else A
                # dead -> alive for t; alive -> switch only if the other team acts
                new = code if state == D or code != state else state
            if new != state:
                if new == D:
                    trigger = time
                labels[last:time] = state
                last = time
                state = new
        labels[last:p_stop] = state
    return labels
