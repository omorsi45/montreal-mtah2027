"""Tracking readers: synthetic files only (no competition data)."""

from __future__ import annotations

import numpy as np
import pytest

from montreal import paths, tracking
from montreal.tracking import Tracking, read_epts, read_metrica_csv

NS = "FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS"


def _meta_xml(params: dict[str, str]) -> str:
    provider = "".join(
        f"<ProviderParameter><Name>{k}</Name><Value>{v}</Value></ProviderParameter>"
        for k, v in params.items()
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<main xmlns="{NS}">
  <Metadata>
    <GlobalConfig><ProviderGlobalParameters>{provider}</ProviderGlobalParameters></GlobalConfig>
    <Teams><Team id="TA"><Name>A</Name></Team><Team id="TB"><Name>B</Name></Team></Teams>
    <Players>
      <Player id="P1" teamId="TA"/><Player id="P2" teamId="TA"/><Player id="P3" teamId="TB"/>
    </Players>
    <PlayerChannels>
      <PlayerChannel channelId="x" id="player1_x" playerId="P1"/>
      <PlayerChannel channelId="y" id="player1_y" playerId="P1"/>
      <PlayerChannel channelId="s" id="player1_s" playerId="P1"/>
      <PlayerChannel channelId="x" id="player2_x" playerId="P2"/>
      <PlayerChannel channelId="y" id="player2_y" playerId="P2"/>
      <PlayerChannel channelId="s" id="player2_s" playerId="P2"/>
      <PlayerChannel channelId="x" id="player3_x" playerId="P3"/>
      <PlayerChannel channelId="y" id="player3_y" playerId="P3"/>
      <PlayerChannel channelId="s" id="player3_s" playerId="P3"/>
    </PlayerChannels>
    <DataFormatSpecifications>
      <DataFormatSpecification startFrame="1" endFrame="3" separator=":">
        <StringRegister name="frameCount"/>
        <SplitRegister separator=";">
          <SplitRegister separator=",">
            <PlayerChannelRef playerChannelId="player1_x"/><PlayerChannelRef playerChannelId="player1_y"/>
          </SplitRegister>
          <SplitRegister separator=",">
            <PlayerChannelRef playerChannelId="player2_x"/><PlayerChannelRef playerChannelId="player2_y"/>
          </SplitRegister>
          <SplitRegister separator=",">
            <PlayerChannelRef playerChannelId="player3_x"/><PlayerChannelRef playerChannelId="player3_y"/>
          </SplitRegister>
        </SplitRegister>
        <SplitRegister separator=",">
          <BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>
        </SplitRegister>
      </DataFormatSpecification>
      <DataFormatSpecification startFrame="4" endFrame="5" separator=":">
        <StringRegister name="frameCount"/>
        <SplitRegister separator=";">
          <SplitRegister separator=",">
            <PlayerChannelRef playerChannelId="player3_x"/><PlayerChannelRef playerChannelId="player3_y"/>
            <PlayerChannelRef playerChannelId="player3_s"/>
          </SplitRegister>
          <SplitRegister separator=",">
            <PlayerChannelRef playerChannelId="player1_x"/><PlayerChannelRef playerChannelId="player1_y"/>
            <PlayerChannelRef playerChannelId="player1_s"/>
          </SplitRegister>
          <SplitRegister separator=",">
            <PlayerChannelRef playerChannelId="player2_x"/><PlayerChannelRef playerChannelId="player2_y"/>
            <PlayerChannelRef playerChannelId="player2_s"/>
          </SplitRegister>
        </SplitRegister>
        <SplitRegister separator=",">
          <BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/>
        </SplitRegister>
      </DataFormatSpecification>
    </DataFormatSpecifications>
  </Metadata>
</main>
"""


PARAMS = {
    "first_half_start": "1",
    "first_half_end": "3",
    "second_half_start": "4",
    "second_half_end": "5",
    "first_extra_half_start": "",
    "first_extra_half_end": "",
}

RAW = "\n".join(
    [
        # spec 1: slots P1, P2, P3 (x,y)
        "1:0.1,0.2;0.3,0.4;0.5,0.6:0.5,0.5",
        "2:0.1,0.2;NaN,NaN;0.5,0.6:NaN,NaN",
        "3:0.11,0.21;0.31,0.41;0.51,0.61:1.02,1.285",
        # spec 2: order P3, P1, P2 with speed
        "4:0.5,0.6,9.0;0.1,0.2,8.0;0.3,0.4,7.0:0.25,0.75,3.0",
        "5:NaN,NaN,NaN;0.12,0.22,1.0;0.32,0.42,1.0:0.5,0.5,0.0",
    ]
)


def _write_epts(tmp_path, params=PARAMS):
    meta = tmp_path / "meta.xml"
    raw = tmp_path / "raw.txt"
    meta.write_text(_meta_xml(params), encoding="utf-8")
    raw.write_text(RAW + "\n", encoding="utf-8")
    return meta, raw


def test_epts_two_specs(tmp_path):
    meta, raw = _write_epts(tmp_path)
    t = read_epts(meta, raw, (105.0, 68.0))
    assert isinstance(t, Tracking)
    assert t.ball.shape == (5, 2)
    assert t.players.shape == (5, 3, 2)
    assert t.team.tolist() == [0, 0, 1]
    assert t.team.dtype == np.int8
    assert t.periods == [(0, 3), (3, 5)]
    assert t.pitch == (105.0, 68.0)
    assert t.fps == 25
    # spec 1 placement: slots in metadata order
    np.testing.assert_allclose(t.players[0, 0], [0.1 * 105, 0.2 * 68])
    np.testing.assert_allclose(t.players[0, 1], [0.3 * 105, 0.4 * 68])
    np.testing.assert_allclose(t.players[0, 2], [0.5 * 105, 0.6 * 68])
    np.testing.assert_allclose(t.ball[0], [0.5 * 105, 0.5 * 68])
    # NaN groups stay NaN
    assert np.isnan(t.players[1, 1]).all()
    assert not np.isnan(t.players[1, 0]).any()
    assert np.isnan(t.ball[1]).all()
    # spec 2: different order, 3 fields: still the right slot
    np.testing.assert_allclose(t.players[3, 0], [0.1 * 105, 0.2 * 68])
    np.testing.assert_allclose(t.players[3, 1], [0.3 * 105, 0.4 * 68])
    np.testing.assert_allclose(t.players[3, 2], [0.5 * 105, 0.6 * 68])
    np.testing.assert_allclose(t.ball[3], [0.25 * 105, 0.75 * 68])
    assert np.isnan(t.players[4, 2]).all()
    np.testing.assert_allclose(t.players[4, 0], [0.12 * 105, 0.22 * 68])


def test_epts_explicit_periods_override(tmp_path):
    meta, raw = _write_epts(tmp_path)
    t = read_epts(meta, raw, (105.0, 68.0), periods=[(1, 2)])
    assert t.periods == [(1, 2)]


def test_epts_empty_params_ignored(tmp_path):
    meta, raw = _write_epts(
        tmp_path, {"first_half_start": "1", "first_half_end": "3", "second_half_start": "", "second_half_end": ""}
    )
    t = read_epts(meta, raw, (105.0, 68.0))
    assert t.periods == [(0, 3)]


def test_epts_out_of_box_ball(tmp_path):
    meta, raw = _write_epts(tmp_path)
    t = read_epts(meta, raw, (105.0, 68.0))
    np.testing.assert_allclose(t.ball[2], [1.02 * 105, 1.285 * 68])
    assert t.ball[2, 0] == pytest.approx(107.1)
    assert t.ball[2, 1] == pytest.approx(87.38)


def _csv(team: str, jerseys: list[int], rows: list[list[str]], ball: bool = True) -> str:
    n = len(jerseys)
    h1 = ["", "", ""] + sum(([team, ""] for _ in range(n)), []) + ["", ""]
    h2 = ["", "", ""] + sum(([str(j), ""] for j in jerseys), []) + ["", ""]
    h3 = ["Period", "Frame", "Time [s]"] + sum(([f"Player{j}", ""] for j in jerseys), []) + ["Ball", ""]
    lines = [",".join(h1), ",".join(h2), ",".join(h3)]
    lines += [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


def test_metrica_csv_roundtrip(tmp_path):
    home_rows = [
        ["1", "1", "0.04", "0.1", "0.2", "0.3", "0.4", "0.5", "0.5"],
        ["1", "2", "0.08", "0.1", "0.2", "NaN", "NaN", "NaN", "NaN"],
        ["2", "3", "0.12", "0.2", "0.3", "0.4", "0.5", "0.25", "0.75"],
        ["2", "4", "0.16", "0.2", "0.3", "0.4", "0.5", "0.25", "0.75"],
    ]
    away_rows = [
        ["1", "1", "0.04", "0.6", "0.7"],
        ["1", "2", "0.08", "0.6", "0.7"],
        ["2", "3", "0.12", "0.8", "0.9"],
        ["2", "4", "0.16", "NaN", "NaN"],
    ]
    home = tmp_path / "home.csv"
    away = tmp_path / "away.csv"
    home.write_text(_csv("Home", [11, 1], home_rows), encoding="utf-8")
    away.write_text(_csv("Away", [25], [r + ["0.5", "0.5"] for r in away_rows]), encoding="utf-8")
    t = read_metrica_csv(home, away, (105.0, 68.0))
    assert t.periods == [(0, 2), (2, 4)]
    assert t.players.shape == (4, 3, 2)
    assert t.team.tolist() == [0, 0, 1]
    assert t.ball.shape == (4, 2)
    np.testing.assert_allclose(t.players[0, 0], [0.1 * 105, 0.2 * 68])
    np.testing.assert_allclose(t.players[0, 1], [0.3 * 105, 0.4 * 68])
    np.testing.assert_allclose(t.players[0, 2], [0.6 * 105, 0.7 * 68])
    np.testing.assert_allclose(t.players[2, 2], [0.8 * 105, 0.9 * 68])
    assert np.isnan(t.players[3, 2]).all()
    assert np.isnan(t.players[1, 1]).all()
    np.testing.assert_allclose(t.ball[0], [0.5 * 105, 0.5 * 68])
    assert np.isnan(t.ball[1]).all()  # missing ball -> NaN
    np.testing.assert_allclose(t.ball[2], [0.25 * 105, 0.75 * 68])


def test_competition_cache_reused(monkeypatch, tmp_path):
    games, _ = paths.upstream()
    data = tmp_path / "data"
    folder = data / "tactical"
    folder.mkdir(parents=True)
    ap = games.asset_paths(data_dir=str(data))
    (folder / "BAR-ATH_tactical_atd.xml").write_text(_meta_xml(PARAMS), encoding="utf-8")
    (folder / "BAR-ATH_tactical_atd.txt").write_text(RAW + "\n", encoding="utf-8")
    assert ap["atd_meta"].endswith("BAR-ATH_tactical_atd.xml")
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("MTAH_DATA_DIR", str(data))
    monkeypatch.setattr(paths, "cache_dir", lambda: cache)

    calls = {"n": 0}
    original = tracking.read_epts

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tracking, "read_epts", counting)
    first = tracking.load_competition()
    assert calls["n"] == 1
    assert (cache / "bar-ath_tactical.npz").exists()
    second = tracking.load_competition()
    assert calls["n"] == 1  # raw file not re-read
    np.testing.assert_array_equal(first.ball, second.ball)
    np.testing.assert_array_equal(first.players, second.players)
    np.testing.assert_array_equal(first.team, second.team)
    assert first.periods == second.periods == [(11185, 90652), (110808, 190198)]
    assert first.pitch == second.pitch == tuple(games.GAMES["BAR-ATH"]["pitch"])

    # touching the raw file invalidates the cache
    raw = folder / "BAR-ATH_tactical_atd.txt"
    raw.write_text(RAW + "\n\n", encoding="utf-8")
    tracking.load_competition()
    assert calls["n"] == 2
