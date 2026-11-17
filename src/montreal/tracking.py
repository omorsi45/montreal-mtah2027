"""Tracking container and readers for FIFA/EPTS and Metrica CSV tracking data.

Frame index = array row. Coordinates are metres (normalised file values times the pitch size);
NaN marks a missing player or ball.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from montreal import paths

GAME = "BAR-ATH"
CACHE_NAME = "bar-ath_tactical.npz"


@dataclass(frozen=True)
class Tracking:
    ball: np.ndarray  # (n, 2) float64 metres, NaN = missing
    players: np.ndarray  # (n, S, 2) float64 metres, NaN = missing
    team: np.ndarray  # (S,) int8, 0 = home / first team, 1 = away
    periods: list[tuple[int, int]]  # half-open [start, stop) row spans of play
    pitch: tuple[float, float]  # (length, width) metres
    fps: int = 25


# --------------------------------------------------------------------------------------
# FIFA / EPTS
# --------------------------------------------------------------------------------------


def _local(tag: str) -> str:
    """Tag name without any XML namespace."""
    return tag.rsplit("}", 1)[-1]


@dataclass
class _Spec:
    start: int
    end: int
    fields: int  # fields per player group in the raw line
    slots: np.ndarray  # slot of each group, in line order
    x_pos: int
    y_pos: int


def _parse_meta(meta_path):
    root = ET.parse(str(meta_path)).getroot()
    teams: list[str] = []
    players: list[tuple[str, str]] = []  # (player id, team id)
    channel_player: dict[str, str] = {}
    channel_kind: dict[str, str] = {}
    spec_nodes = []
    params: dict[str, str] = {}
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "Team" and "id" in el.attrib:
            teams.append(el.attrib["id"])
        elif tag == "Player" and "id" in el.attrib:
            players.append((el.attrib["id"], el.attrib.get("teamId", "")))
        elif tag == "PlayerChannel":
            channel_player[el.attrib["id"]] = el.attrib["playerId"]
            channel_kind[el.attrib["id"]] = el.attrib.get("channelId", "")
        elif tag == "DataFormatSpecification":
            spec_nodes.append(el)
        elif tag == "ProviderParameter":
            name = value = None
            for child in el:
                if _local(child.tag) == "Name":
                    name = (child.text or "").strip()
                elif _local(child.tag) == "Value":
                    value = (child.text or "").strip()
            if name:
                params.setdefault(name, value or "")

    slot_of = {pid: i for i, (pid, _) in enumerate(players)}
    # Only the first two <Team> ids are the two teams; any other teamId (referee, unknown, ...)
    # gets code -1 and is ignored by every feature, like a slot with NaN positions.
    two = teams[:2]
    team = np.array([two.index(tid) if tid in two else -1 for _, tid in players], dtype=np.int8)

    specs: list[_Spec] = []
    for node in spec_nodes:
        groups: list[list[str]] = []  # channel ids, consecutive refs of one player form a group
        last_player = None
        for el in node.iter():
            if _local(el.tag) != "PlayerChannelRef":
                continue
            cid = el.attrib["playerChannelId"]
            pid = channel_player[cid]
            if pid != last_player:
                groups.append([])
                last_player = pid
            groups[-1].append(cid)
        if not groups:
            continue
        fields = len(groups[0])
        if any(len(g) != fields for g in groups):
            raise ValueError("inconsistent number of fields per player group")
        kinds = [channel_kind[c] for c in groups[0]]
        x_pos = kinds.index("x") if "x" in kinds else 0
        y_pos = kinds.index("y") if "y" in kinds else 1
        slots = np.array([slot_of[channel_player[g[0]]] for g in groups], dtype=np.int64)
        specs.append(
            _Spec(
                start=int(node.attrib["startFrame"]),
                end=int(node.attrib["endFrame"]),
                fields=fields,
                slots=slots,
                x_pos=x_pos,
                y_pos=y_pos,
            )
        )
    specs.sort(key=lambda s: s.start)
    return team, len(players), specs, params


def _periods_from_params(params: dict[str, str]) -> list[tuple[int, int]]:
    out = []
    for prefix in ("first_half", "second_half", "first_extra_half", "second_extra_half"):
        start, end = params.get(f"{prefix}_start", ""), params.get(f"{prefix}_end", "")
        if start and end:
            out.append((int(start) - 1, int(end)))  # frame numbers -> half-open rows
    return out


def _floats(tokens: list[str]) -> np.ndarray:
    try:
        return np.array(tokens, dtype=np.float64)
    except ValueError:
        out = np.full(len(tokens), np.nan)
        for i, tok in enumerate(tokens):
            try:
                out[i] = float(tok)
            except ValueError:
                pass
        return out


def read_epts(meta_path, raw_path, pitch, periods=None) -> Tracking:
    """Read FIFA/EPTS metadata + raw tracking into a Tracking (row = frame - 1)."""
    team, n_slots, specs, params = _parse_meta(meta_path)
    if not specs:
        raise ValueError("no DataFormatSpecification with player channels in metadata")
    n = max(s.end for s in specs)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, n_slots, 2), np.nan)

    current = None
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            try:
                frame = int(parts[0])
            except ValueError:
                continue
            row = frame - 1
            if row < 0 or row >= n:
                continue
            if current is None or not (current.start <= frame <= current.end):
                current = next((s for s in specs if s.start <= frame <= s.end), None)
                if current is None:
                    continue
            spec = current
            if len(parts) > 1 and parts[1]:
                vals = _floats(parts[1].replace(";", ",").split(","))
                if vals.size == spec.slots.size * spec.fields:
                    vals = vals.reshape(spec.slots.size, spec.fields)
                    players[row, spec.slots, 0] = vals[:, spec.x_pos]
                    players[row, spec.slots, 1] = vals[:, spec.y_pos]
            if len(parts) > 2 and parts[2]:
                b = _floats(parts[2].split(","))
                if b.size >= 2:
                    ball[row] = b[:2]

    scale = np.array([pitch[0], pitch[1]], dtype=np.float64)
    ball *= scale
    players *= scale
    if periods is None:
        periods = _periods_from_params(params) or [(0, n)]
    return Tracking(
        ball=ball,
        players=players,
        team=team,
        periods=[(int(a), int(b)) for a, b in periods],
        pitch=(float(pitch[0]), float(pitch[1])),
    )


# --------------------------------------------------------------------------------------
# Metrica CSV
# --------------------------------------------------------------------------------------


def _read_metrica_side(path):
    """Return (period, players (n, P, 2), ball (n, 2)) as normalised values."""
    head = pd.read_csv(path, header=None, nrows=3, dtype=str, keep_default_na=False)
    names = [str(v) for v in head.iloc[2].tolist()]
    n_players = sum(1 for v in names if v.startswith("Player"))
    ncols = 3 + 2 * (n_players + 1)
    df = pd.read_csv(path, header=None, skiprows=3, usecols=range(ncols), dtype=np.float64)
    values = df.to_numpy(dtype=np.float64)
    period = values[:, 0]
    xy = values[:, 3:ncols].reshape(len(values), n_players + 1, 2)
    return period, xy[:, :n_players, :], xy[:, n_players, :]


def read_metrica_csv(home_csv, away_csv, pitch) -> Tracking:
    """Read a Metrica sample game (separate home and away CSVs). Ball comes from the home file."""
    period_h, home, ball = _read_metrica_side(home_csv)
    period_a, away, _ = _read_metrica_side(away_csv)
    if len(period_h) != len(period_a):
        raise ValueError("home and away tracking files differ in length")
    scale = np.array([pitch[0], pitch[1]], dtype=np.float64)
    players = np.concatenate([home, away], axis=1) * scale
    team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)

    spans: list[tuple[int, int]] = []
    valid = ~np.isnan(period_h)
    start = None
    for i in range(len(period_h) + 1):
        cur = period_h[i] if i < len(period_h) and valid[i] else None
        prev = period_h[i - 1] if i > 0 and valid[i - 1] else None
        if cur != prev:
            if start is not None:
                spans.append((start, i))
            start = i if cur is not None else None
    return Tracking(
        ball=ball * scale,
        players=players,
        team=team,
        periods=spans,
        pitch=(float(pitch[0]), float(pitch[1])),
    )


# --------------------------------------------------------------------------------------
# Competition (cached)
# --------------------------------------------------------------------------------------


def _code_version() -> str:
    """Digest of this module's source: the cached arrays are only valid for the reader that made them."""
    return paths.source_digest(__file__)


def _cache_key(raw_path: str) -> np.ndarray:
    st = os.stat(raw_path)
    return np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)


def _load_cache(cache_file: Path, key: np.ndarray, raw_path: str, code: str):
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as z:
            if (not np.array_equal(z["key"], key) or str(z["raw_path"]) != raw_path
                    or str(z["code"]) != code):
                return None
            return Tracking(
                ball=z["ball"],
                players=z["players"],
                team=z["team"],
                periods=[(int(a), int(b)) for a, b in z["periods"]],
                pitch=(float(z["pitch"][0]), float(z["pitch"][1])),
                fps=int(z["fps"]),
            )
    except (OSError, ValueError, KeyError):
        return None


def load_competition() -> Tracking:
    """Tactical ATD tracking of the competition match, cached as .cache/bar-ath_tactical.npz."""
    games, _ = paths.upstream()
    assets = games.asset_paths(GAME, data_dir=str(paths.data_dir()))
    raw_path = str(assets["atd_raw"])
    key = _cache_key(raw_path)
    cache_file = Path(paths.cache_dir()) / CACHE_NAME

    code = _code_version()
    cached = _load_cache(cache_file, key, raw_path, code)
    if cached is not None:
        return cached

    periods = [(int(first), int(last) + 1) for _, first, last in games.half_video_blocks(GAME)]
    pitch = tuple(float(v) for v in games.GAMES[GAME]["pitch"])
    t = read_epts(assets["atd_meta"], raw_path, pitch, periods=periods)

    tmp = cache_file.with_name(cache_file.name + ".tmp.npz")
    np.savez(
        tmp,
        ball=t.ball,
        players=t.players,
        team=t.team,
        periods=np.array(t.periods, dtype=np.int64),
        pitch=np.array(t.pitch, dtype=np.float64),
        fps=np.array(t.fps, dtype=np.int64),
        key=key,
        raw_path=np.array(raw_path),
        code=np.array(code),
    )
    os.replace(tmp, cache_file)
    return t
