# Possession Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic pipeline that turns the provided BAR-ATH tracking + Smart Tagging into
`submissions/Montreal.csv` (199,319 rows of H/A/D/X), with emission weights and switch costs
learned on Metrica's public sample games.

**Architecture:** Shared readers produce one `Tracking` container for our match and for three
public sample games. Team-level per-frame features feed an emission model (hand heuristic or
softmax regression) giving log P(H/A/D); Smart Tagging adds a calibrated, capped log-prior; a
3-state Viterbi decoder with switch costs turns frames into whole possessions per half.

**Tech Stack:** Python 3.12, numpy, pandas, pytest. Official scorer imported from the upstream
repository clone.

**Spec:** `docs/superpowers/specs/2026-11-17-possession-pipeline-design.md`

## Global Constraints

- Repo root: `C:\Users\Omar Morsi\Desktop\Projects\montreal-mtah2027`. Python 3.12 venv at
  `.venv` (`py -3.12 -m venv .venv`). Runtime deps: numpy, pandas only; dev: pytest. Every pin `==`.
- Upstream repo: env `MTAH_REPO_DIR`, default `<repo>/../more-than-a-hack-2027`. Data: env
  `MTAH_DATA_DIR`, default `<upstream>/data`. The official scorer (`games.py`,
  `evaluation/possession_metrics.py`) is **imported from upstream, never copied**. The public GT is
  never copied. *(Builder's call, replacing "vendored" in the spec: importing guarantees the exact
  official scorer and redistributes nothing.)*
- Sample data: env `MONTREAL_SAMPLE_DIR`, default `<repo>/external/metrica-sample-data`, from
  `https://github.com/metrica-sports/sample-data` at commit
  `e706dd506b360d69d9d123d5b8026e7294b13996`.
- Label codes everywhere: `H=0, A=1, D=2, X=3` (same as upstream `possession_metrics`).
- Frame index = array row. Competition: row = video frame = ATD `frame_id - 1`. Sample games:
  row = sample frame number − 1.
- Competition periods (half-open): from `games.half_video_blocks("BAR-ATH")` →
  `[(11185, 90652), (110808, 190198)]`. Everything outside is `X`. No other competition frame
  number may appear in code.
- Pitch metres: competition `games.GAMES["BAR-ATH"]["pitch"]` = (105.3, 68.0); sample games
  (105.0, 68.0). fps = 25.
- Submission bytes: header `frame,label`, 199,319 rows, `\n` endings, ASCII, no trailing blank line.
- Determinism: the only RNG is `numpy.random.default_rng(20261117)` (derived seeds via
  `default_rng([20261117, k])`); log-scores rounded with `np.round(x, 6)` before Viterbi; two runs
  give the same SHA-256.
- **Data privacy:** tests use synthetic data only. No script or test prints raw rows or values of
  the competition data — only counts, shares, scores. `.gitignore` covers `data/`, `.cache/`,
  `external/`, `submissions/*.csv`, `.venv/`, `.playwright-cli/`.
- **Rule compliance:** public 5-minute GT appears only in `run.py`'s final "sanity check only"
  line; it is never used to choose any parameter or version.

## Review Focus

1. **Ungraded / empty frames** — halftime and pre-match rows (no players, no ball) must give finite
   features and `X` in the output; no possession may span the halftime gap. (Tasks 4, 9.)
2. **Tracklet chaos** — slots NaN most of the time, more than 22 present, or one team with zero
   tracked players on a frame: features stay finite (team distance = 30 m). (Task 4.)
3. **Ball far outside the pitch** (normalised y up to ~1.3) — `ball_out=1`, `ball_edge=0`, no
   crash or warning-as-error. (Task 4.)
4. **Smart Tagging strings** — `Atletic Club` (sic), `N/A`, missing `Team`, mixed case in codes:
   matched case-insensitively; unknown team → no team evidence. (Task 7.)
5. **Sample events with odd frames** — `End Frame` 0 or < start, frames beyond the tracking, events
   outside a period: clipped or ignored; label array length always `n`. (Task 3.)

---

### Task 1: Scaffold, paths, upstream scorer access

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `LICENSE` (MIT, "Team Montreal"),
  `src/montreal/__init__.py`, `src/montreal/paths.py`, `src/montreal/metrics.py`,
  `scripts/fetch_sample_data.py`, `tests/test_paths_metrics.py`

**Interfaces:**
- Produces:
  - `paths.repo_dir() -> Path`, `paths.upstream_dir() -> Path`, `paths.data_dir() -> Path`,
    `paths.sample_dir() -> Path`, `paths.cache_dir() -> Path` (creates `<repo>/.cache`).
  - `paths.upstream() -> tuple[ModuleType, ModuleType]` → `(games, possession_metrics)`, imported by
    inserting `upstream_dir()` and `upstream_dir()/evaluation` on `sys.path` once.
  - `metrics.score(gt: np.ndarray, pred: np.ndarray) -> dict` with keys `spell_f1`, `macro_f1`,
    `n_pred_spells`, `n_gt_spells` from upstream `pm.evaluate(gt, pred)`. Inputs are int8 code
    arrays. Pull the possession counts from `evaluate`'s output, or use `pm.possession_spells`
    with `pm.valid_runs(gt)`.

- [ ] **Step 1:** Create the venv with `py -3.12 -m venv .venv`, then run
  `.venv\Scripts\python -m pip install numpy pandas pytest`. Write the exact installed versions
  into `requirements.txt` (numpy, pandas) and `requirements-dev.txt` (`-r requirements.txt`
  plus pytest). `pyproject.toml`:
  - setuptools `src` layout, package `montreal`, `requires-python = ">=3.12"`;
  - `[tool.pytest.ini_options] pythonpath = ["src"]`, `testpaths = ["tests"]`.

  Then run `.venv\Scripts\python -m pip install -e . --no-deps`.
- [ ] **Step 2: Write failing tests** in `tests/test_paths_metrics.py`:
  - `test_upstream_importable`: `games, pm = paths.upstream()`; `games.half_video_blocks("BAR-ATH") == [(1, 11185, 90651), (2, 110808, 190197)]`; `pm.H == 0 and pm.X == 3`.
  - `test_env_overrides(monkeypatch, tmp_path)`: setting `MTAH_DATA_DIR`/`MONTREAL_SAMPLE_DIR` changes `data_dir()`/`sample_dir()`.
  - `test_score_perfect_and_constant`: GT = X×10, H×100, A×100, D×50, X×10. `score(gt, gt)["spell_f1"] == 1.0`. A prediction of all D scores `spell_f1 == 0.0`. `n_gt_spells == 2`.
- [ ] **Step 3:** Run `.venv\Scripts\python -m pytest tests/test_paths_metrics.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4:** Implement `paths.py` and `metrics.py`. `scripts/fetch_sample_data.py`
  clones the sample repo into `sample_dir()` if absent, checks out the pinned commit, and
  verifies `git rev-parse HEAD` equals the pin. It exits non-zero on mismatch.
- [ ] **Step 5:** Run the tests. Expected: PASS. Run `.venv\Scripts\python scripts/fetch_sample_data.py`. Expected: prints the pinned SHA.
- [ ] **Step 6:** Commit: `git add` the files above. Message: `chore: scaffold package, paths and scorer access`.

### Task 2: Tracking container and readers (EPTS + Metrica CSV) with cache

**Files:**
- Create: `src/montreal/tracking.py`, `tests/test_tracking.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Tracking:
      ball: np.ndarray            # (n, 2) float64 metres, NaN = missing
      players: np.ndarray         # (n, S, 2) float64 metres, NaN = missing
      team: np.ndarray            # (S,) int8, 0 = home (first <Team> in XML / "Home" CSV), 1 = away
      periods: list[tuple[int, int]]  # half-open [start, stop) row spans of play
      pitch: tuple[float, float]  # (length, width) m
      fps: int = 25
  ```
  - `read_epts(meta_path, raw_path, pitch, periods=None) -> Tracking`. Handles any number of
    `DataFormatSpecification` blocks: each has `startFrame`/`endFrame` and an ordered
    `PlayerChannelRef` list naming `playerN_x`/`playerN_y`(/`_s`). It maps to `Player id` →
    `teamId` via `PlayerChannel`. Slots are the distinct players in metadata order, and S is
    the number of `<Player>` elements. Each raw line is `frame:groups:ball`. Groups are split
    on `;`, fields on `,`, and the field count per group comes from that spec's channel refs.
    The ball takes its first two fields. `NaN` means missing. Row = frame − 1.
    - **Periods:** when `periods` is None, read `first_half_start/end` and
      `second_half_start/end` from the `ProviderParameter`s, as frames converted to rows,
      half-open. Ignore empty values.
  - `read_metrica_csv(home_csv, away_csv, pitch) -> Tracking`: the Metrica sample CSV layout. It
    has 3 header rows, the third naming `Period, Frame, Time [s], PlayerN, , ..., Ball, `, and
    x,y column pairs. Periods are the row spans of each `Period` value. Slots are the home
    columns then the away columns. The ball comes from the home file.
  - `load_competition() -> Tracking`: tactical ATD via upstream `games.asset_paths()`, periods
    from `half_video_blocks` (stop = last + 1), pitch from `games.GAMES`. It is cached in
    `.cache/bar-ath_tactical.npz`, keyed by the raw file's size and mtime.

- [ ] **Step 1: Write failing tests** with synthetic files written to `tmp_path`:
  - `test_epts_two_specs`: a meta XML with 3 players (2 teams) and two specs, frames 1–3 and 4–5. The spec for frames 4–5 lists a different player order and 3 fields per group (`x,y,s`). Expect `ball.shape == (5, 2)` and players placed in the right slot in both specs. Coordinates are scaled: `0.5` → `0.5*105` m on x and `0.5*68` m on y. `NaN` groups become NaN, `team` is `[0, 0, 1]`, and periods come from the provider params.
  - `test_epts_out_of_box_ball`: a ball at normalised `(1.02, 1.285)` is kept unclipped, giving `ball[0] ≈ (107.1, 87.38)`.
  - `test_metrica_csv_roundtrip`: a 4-row synthetic home/away CSV pair. `periods == [(0, 2), (2, 4)]`, slots are home then away, and a missing ball gives NaN.
  - `test_competition_cache_reused(monkeypatch, tmp_path)`: point `MTAH_DATA_DIR` at a synthetic bundle with a 5-line ATD. The second `load_competition()` call does not re-open the raw file (monkeypatch `read_epts` to count calls).
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_tracking.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Stream the raw file line by line. Pre-allocate with `np.full(..., np.nan)`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Real-data check (aggregates only): `python -c "from montreal.tracking import load_competition as L; t=L(); import numpy as np; print(t.ball.shape, t.players.shape, t.periods, round(float(np.isnan(t.ball[:,0]).mean()),3))"`. Expected: `(199319, 2) (199319, 46, 2) [(11185, 90652), (110808, 190198)] 0.431`.
- [ ] **Step 5:** Commit with message `feat: tracking readers for EPTS and Metrica CSV`.

### Task 3: Sample-game events and label derivation

**Files:**
- Create: `src/montreal/events.py`, `tests/test_events.py`

**Interfaces:**
- Consumes: `Tracking` (Task 2), `paths.sample_dir()`.
- Produces:
  - `load_sample_game(k: int) -> tuple[Tracking, pd.DataFrame]` for k ∈ {1, 2, 3}.
    - Games 1–2: `read_metrica_csv` plus `Sample_Game_k_RawEventsData.csv`.
    - Game 3: `read_epts(Sample_Game_3_metadata.xml, Sample_Game_3_tracking.txt)` plus `Sample_Game_3_events.json`, reading `data[i]["team"]["name"]`, `type.name`, `subtypes` (a dict, a list joined with `-`, or None), `start.frame`, `end.frame`, `period`.
    - Pitch is (105.0, 68.0).
  - The events DataFrame has columns `team` ∈ {"H","A"}, `type` (upper), `subtype` (upper, `""` if none), `start` and `end` (int rows, frame − 1), and `period` (int). It is stable-sorted by `start`.
    - Home is "Home" in CSV games, and the first metadata team id in game 3, which maps to the event team whose id equals it.
  - `derive_labels(events: pd.DataFrame, n: int, periods: list[tuple[int,int]]) -> np.ndarray` (int8 codes). Rules, applied per period:
    1. Outside periods → X. Each period starts in D.
    2. **Dead triggers** set D from the trigger frame:
       - `BALL OUT` and `FAULT RECEIVED` at `start`;
       - `SHOT` whose subtype ends with `GOAL` or `-OUT` at `max(start, end)`.
    3. **While D**, the first event of type `SET PIECE`, `PASS`, `CARRY`, `SHOT` or `RECOVERY` makes play alive for that event's team from its `start`.
    4. **While alive with team T**, the first event by the other team U of type `PASS`, `CARRY`, `SHOT`, `SET PIECE` or `RECOVERY` switches to U at its `start`. `CHALLENGE`, `BALL LOST`, `CARD` and anything else never switch.
    5. Frames are clipped to the period; events whose `start` falls outside every period are ignored. `end < start` → use `start`.

- [ ] **Step 1: Write failing tests** (synthetic event frames, n=100, periods `[(0,50),(60,100)]`):
  - `test_kickoff_then_switch`: SET PIECE H@0, PASS H@5, RECOVERY A@20, PASS A@22. Expect `labels[0:20]==H`, `labels[20:50]==A`, `labels[50:60]==X`, and `labels[60:]` starts D.
  - `test_challenge_does_not_switch`: H alive, CHALLENGE A@10, PASS H@12 → all H.
  - `test_ball_out_until_restart`: H alive, BALL OUT H@30, SET PIECE A@40 → `[30:40]==D`, `[40:50]==A`.
  - `test_goal_shot_dead_from_end`: SHOT H start 10 end 14, subtype `ON TARGET-GOAL`, SET PIECE A@30 → `[14:30]==D`.
  - `test_odd_frames` (Review Focus 5): end=−1 (CSV `End Frame` 0), an event at 120 (beyond n), and an event at 55 (gap) → no exception, `len==100`, gap frames X.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_events.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS. Real check on the public sample games:
  `python -c "from montreal.events import *; ..."`, printing per game the frame count, the H/A/D shares and the possession count via `pm.possession_spells`. Expected: D share roughly 0.25–0.45 and possessions roughly 150–500 per game. If outside, re-read the rules rather than tuning them.
- [ ] **Step 5:** Commit with message `feat: sample-game events and possession label derivation`.

### Task 4: Per-frame features

**Files:**
- Create: `src/montreal/features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `Tracking`.
- Produces: `FEATURE_NAMES: list[str]` (exactly the names below, in this order) and
  `compute_features(trk: Tracking) -> np.ndarray` (n, len(FEATURE_NAMES)) float64. Every value is
  finite, all computed inside each period (no window or carry crosses a period boundary), and rows
  outside periods are 0.

  **Base features** (`speed` = centred difference over ±2 rows × fps / 4, NaN-safe):
  - `ball_vis` — 1 if the ball is visible.
  - `d_home`, `d_away` — min distance from the ball to that team's players, clipped to [0, 30]; 30 if the ball is missing or the team has no tracked player.
  - `d_diff` — `clip(d_away − d_home, −15, 15)`; 0 if the ball is missing.
  - `n_home_3m`, `n_away_3m` — count of team players within 3 m of the ball.
  - `ball_out` — visible and outside `[−0.5, L+0.5] × [−0.5, W+0.5]`.
  - `ball_edge` — distance to the nearest boundary line, clipped to [0, 10]; 0 if outside, 10 if missing.
  - `ball_speed` — clipped to [0, 40]; 0 if not computable.
  - `ply_speed` — mean tracked-player speed, clipped to [0, 12]; 0 if none.
  - `frac_slow` — share of tracked players under 1 m/s; 1 if none.
  - `block_prog` — centroid x of all tracked players along home's attacking direction, divided by L.
    Home's direction for a period is +1 if home's mean x over the period is less than away's, else −1.
  - `spread_x_diff`, `spread_y_diff` — home std minus away std, clipped to ±20.
  - `vel_home`, `vel_away` — the team centroid's velocity along that team's own attacking direction, as a centred difference over ±12 rows, clipped to ±8.

  **Carry features** (within the period; "visible" means `ball_vis`):
  - `last_diff`, `next_diff` — `d_diff` at the previous / next visible row; 0 if none.
  - `since_seen`, `until_seen` — seconds, clipped to [0, 30]; 0 when visible; 30 when no such row exists.
  - `last_out` — `ball_out` at the last visible row.
  - `last_edge`, `next_edge` — `ball_edge` at the previous / next visible row; 10 if none.

  **Window features:** for each of `d_diff`, `ball_out`, `ply_speed`, `frac_slow`, `spread_x_diff`,
  `spread_y_diff`, `vel_home`, `vel_away` and `ball_vis`, the centred mean over ±12 and ±50 rows,
  named `<name>_w12` and `<name>_w50`. Use cumsum, truncated at period edges.

- [ ] **Step 1: Write failing tests** on small synthetic `Tracking` objects (pitch 105×68, periods `[(0,40),(50,90)]`, n=100):
  - `test_names_and_shape`: `len(FEATURE_NAMES) == 41`, `compute_features(t).shape == (100, 41)`, `np.isfinite(...).all()`.
  - `test_ball_proximity`: home player at the ball, away 10 m away → `d_home==0`, `d_away==10`, `d_diff==10`, `n_home_3m==1`.
  - `test_review_focus_empty_frames` (RF1): rows 40–49 all NaN, and also all-NaN rows inside a period → finite, outside-period rows all 0.
  - `test_review_focus_tracklets` (RF2): 46 slots with only 3 non-NaN, and a frame with zero away players → `d_away==30`, finite.
  - `test_review_focus_ball_far_out` (RF3): ball at (107.1, 87.4) → `ball_out==1`, `ball_edge==0`.
  - `test_carry_does_not_cross_period`: ball visible only at row 35 (`d_diff` +10) → `last_diff[36:40]==10` and `last_diff[50:60]==0`.
  - `test_attack_direction_flip`: home mean x 30 m in period 1 and 80 m in period 2, with players moving +x at 1 m/s in both → `vel_home > 0` in period 1 and `< 0` in period 2.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_features.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement, vectorised over frames. Use `np.errstate` and nan-aware reductions. The carry uses the `np.maximum.accumulate` index trick per period.
- [ ] **Step 4:** Run the tests. Expected: PASS. Real check: compute on `load_competition()` and print only the runtime and `np.isfinite(F).all()`. Expected: True, under 90 s.
- [ ] **Step 5:** Commit with message `feat: team-level per-frame features`.

### Task 5: Viterbi decoder

**Files:**
- Create: `src/montreal/decode.py`, `tests/test_decode.py`

**Interfaces:**
- Produces:
  - `cost_matrix(swap: float, dead: float, live: float) -> np.ndarray` (3, 3): `[[0, swap, dead], [swap, 0, dead], [live, live, 0]]`. Rows are from H/A/D, columns are to H/A/D.
  - `viterbi(logp: np.ndarray, cost: np.ndarray) -> np.ndarray` int8 (n,). Maximises `Σ logp[t, s_t] − Σ cost[s_{t−1}, s_t]` after `logp = np.round(logp, 6)`. Ties are broken towards keeping the previous state, then towards the lower state index.
  - `decode_periods(logp: np.ndarray, periods, cost) -> np.ndarray` int8 (n,): Viterbi run independently inside each period, X (3) elsewhere.
    - Each period is decoded **as if the state just before it were D**, so starting alive pays `live` once.
    - Football fact behind it: every half starts from a dead ball (kick-off). It is generic, not tied to any frame of the match.

- [ ] **Step 1: Write failing tests:**
  - `test_high_cost_suppresses_flicker`: logp favours H, with 3 single-frame A blips. At swap=10 the output is all H; at swap=0 the blips survive.
  - `test_long_switch_survives`: 200 frames favouring H by 1 nat, then 200 favouring A by 1 nat, swap=20 → exactly one switch, at frame 200 ± 1.
  - `test_periods_independent`: periods `[(0,10),(20,30)]` → frames 10–19 are 3, and each period is decoded separately (the state at 20 is not influenced by frame 9).
  - `test_period_starts_from_dead`: frames 0–4 have logp `[0, −9, 0]` and frames 5–104 have `[0, −9, −3]`, with live=10. Frames 0–4 decode to D (a tie, so the previous state is kept) and frames 5–104 to H.
  - `test_deterministic_ties`: an all-equal logp → all 0 (H), identical on repeat.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_decode.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement: plain-Python scalar loop over frames with 3 states, plus an int8 backpointer array. It must run a 200k-frame sequence in under 2 s.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: three-state Viterbi decoder with switch costs`.

### Task 6: Emission models

**Files:**
- Create: `src/montreal/emission.py`, `tests/test_emission.py`

**Interfaces:**
- Consumes: `FEATURE_NAMES` (Task 4).
- Produces:
  - `heuristic_logp(F: np.ndarray, names: list[str]) -> np.ndarray` (n, 3): log-softmax of `[z_team/2, −z_team/2, z_dead]`, where:
    ```
    gap    = 1 − ball_vis
    z_team = 0.35*d_diff_w12 + 0.15*gap*(last_diff*exp(−since_seen/8) + next_diff*exp(−until_seen/8))
    z_dead = −1.5 + 3.0*ball_out_w12 + 2.5*gap*last_out + 2.0*(frac_slow_w50 − 0.6) + 1.0*(1 − ball_vis_w50)
    ```
  - `class SoftmaxRegression`: `__init__(l2: float = 1e-2, class_weight: str | None = None)`, `fit(X, y) -> self`, `log_proba(X) -> np.ndarray` (n, 3), `to_dict() -> dict`, `from_dict(d) -> SoftmaxRegression` (classmethod).
    - Features are standardised with the training mean and std (std 0 → 1).
    - The intercept is not penalised.
    - `class_weight="balanced"` weights rows by `n / (3 · n_class)`.
    - Fit by Newton/IRLS on the full batch: at most 25 iterations, stopping when `max|Δw| < 1e-6`.
    - Rows with label X are dropped before fitting.
    - `to_dict` stores the feature names, mean, std, weights, l2 and class_weight as plain lists (JSON-safe).

- [ ] **Step 1: Write failing tests:**
  - `test_heuristic_shapes_and_direction`: `d_diff_w12 = +10` with everything else neutral → `argmax == H`; `ball_out_w12 = 1` with `gap = 1` and `last_out = 1` → `argmax == D`; rows of `logp` log-sum-exp to 0.
  - `test_softmax_learns_separable`: a 3-blob synthetic problem gives training accuracy above 0.95.
  - `test_softmax_deterministic_and_roundtrip`: two fits give identical weights, and the `from_dict(to_dict())` round-trip reproduces `log_proba` exactly.
  - `test_ignores_X_rows`: adding X-labelled rows leaves the fitted weights unchanged.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_emission.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: heuristic and softmax-regression emissions`.

### Task 7: Smart Tagging evidence and calibration

**Files:**
- Create: `src/montreal/tagging.py`, `tests/test_tagging.py`

**Interfaces:**
- Consumes: `FEATURE_NAMES`, the feature matrix.
- Produces:
  - `load_tags(path, fps=25) -> list[dict]` (stdlib XML). Keys:
    - `code`: upper, stripped;
    - `start`, `stop`: `round(s*fps)` half-open;
    - `team`: `"H"` if the Team lowercased is `barcelona`, `"A"` if it is `atletic club` or `athletic club`, else None;
    - `type`: lower, `""` if absent.
  - Families:
    - `ATTACK = {BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION, BALL IN FINAL THIRD, BALL IN THE BOX, PLAYERS IN THE BOX, LONG BALL}`;
    - `DEFEND = {DEFENDING IN ATTACKING THIRD, DEFENDING IN MIDDLE THIRD, DEFENDING IN DEFENSIVE THIRD, DEFENSIVE TRANSITION}`;
    - `SETPIECE = {SET PIECES}`;
    - `GOAL = {GOALS}`.
  - `tag_votes(tags, n) -> dict[str, np.ndarray]` with keys `attack`, `defend`, `setpiece`. Each value is int8 (n,) of +1 (implies H), −1 (implies A) or 0. A defend tag implies the other team. Overlapping same-family votes are summed, then the sign is taken.
  - `goal_dead(tags, n) -> np.ndarray` bool: from each GOALS start to the start of the next SET PIECES tag whose type contains `kick off`, capped at 90 s.
  - `confident_team(F, names) -> np.ndarray` int8:
    - +1 where `ball_vis == 1`, `d_home ≤ 1.0` and `d_away ≥ 3.0` hold for a run of at least 25 consecutive rows;
    - −1 for the mirror condition;
    - 0 otherwise.
  - `calibrate(votes, confident) -> dict[str, float]`. For each family, `p` is the share of frames with `vote ≠ 0` and `confident ≠ 0` on which the two agree. The weight is `w = clip(log(p/(1−p)), 0, 1.0)`. If fewer than 250 such frames exist, `w = 0`.
  - `tag_prior(votes, goal_mask, weights, w_goal=1.0) -> np.ndarray` (n, 3) additive:
    - column H += `w_f` where the vote is +1; column A += `w_f` where it is −1;
    - column D += `w_goal` where `goal_mask` is set.

- [ ] **Step 1: Write failing tests** with a synthetic SportsCode XML in `tmp_path`:
  - `test_team_mapping` (RF4): the strings `Barcelona`, `ATLETIC CLUB`, `N/A` and a missing Team map to H, A, None, None. A code ` build up ` maps to the `attack` family.
  - `test_defend_implies_other_team`: a defend tag for team H gives votes of −1.
  - `test_goal_dead_until_kickoff`: GOALS at 10–12 s and a kick-off tag at 40 s → mask true over rows [250, 1000).
  - `test_calibrate_cap_and_minimum`: perfect agreement on 1000 frames gives weight 1.0 (capped); 100 frames gives 0.0; 50% agreement gives 0.0.
  - `test_confident_run_length`: a 24-row run gives 0 and a 25-row run gives ±1.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_tagging.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: Smart Tagging votes, goal stoppages and calibrated prior`.

### Task 8: Pipeline, writer and CLI — Milestone 1 submission

**Files:**
- Create: `src/montreal/pipeline.py`, `src/montreal/run.py`, `models/params.json`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces:
  - `params.json` schema is `{"emission": "heuristic" | <SoftmaxRegression.to_dict()>, "costs": {"swap": float, "dead": float, "live": float}, "use_tags": bool, "meta": {...}}`.
    - Milestone 1 values are `{"emission": "heuristic", "costs": {"swap": 20, "dead": 10, "live": 10}, "use_tags": true, "meta": {"source": "hand-set M1"}}`.
  - `emission_logp(F, params) -> np.ndarray` (n, 3), via the heuristic or `from_dict`.
  - `predict_codes(trk, F, params, prior=None) -> np.ndarray` int8: `decode_periods(emission_logp + prior, trk.periods, cost_matrix(**costs))`.
  - `predict_match(params) -> np.ndarray` int8 (199319,). It runs on the competition data, with the tag prior from `calibrate(tag_votes, confident_team)` when `use_tags` is set.
  - `write_submission(codes, path) -> str` returns the SHA-256 hex digest. The format is the Global Constraints bytes, with letters H/A/D/X.
  - CLI `python -m montreal.run [--params models/params.json] [--out submissions/Montreal.csv]`:
    1. writes the file;
    2. validates it with upstream `pm.read_label_csv` + `pm.validate_submission(frame, 199319)` and exits 1 on failure;
    3. prints label counts, the predicted possession count and mean duration, the calibrated tag weights and the SHA-256;
    4. finally prints one line `public-window sanity check only (not used for selection): spellF1 x.xxxx macroF1 x.xxxx`, scored against the upstream public GT.

- [ ] **Step 1: Write failing tests:**
  - `test_write_submission_bytes`: codes `[3, 0, 1, 2]` give the file bytes `b"frame,label\n0,X\n1,H\n2,A\n3,D\n"`, and the returned SHA matches `hashlib`.
  - `test_predict_codes_synthetic_end_to_end` (RF1): a synthetic `Tracking` with two periods and a gap. Home players are near the ball for 300 rows, then away for 300. The output has X in the gap, no possession spans the gap, and there is one H→A switch within ±25 rows.
  - `test_static_players_no_ball_is_dead`: a block of 500 rows with all players still and no ball, inside a period, decodes mostly to D (> 80%) with the heuristic.
  - `test_run_twice_identical` (marked `@pytest.mark.slow`, needs the real data): runs `predict_match` twice → `array_equal`.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_pipeline.py -v -m "not slow"`. Expected: FAIL.
- [ ] **Step 3:** Implement. Register the `slow` marker in `pyproject.toml`.
- [ ] **Step 4:** Run all tests, including slow. Expected: PASS. Then run `.venv\Scripts\python -m montreal.run`. Expected: validation OK; the possession count is printed. Record the sanity line in the commit message, for information only.
- [ ] **Step 5:** Commit with message `feat: end-to-end pipeline and CLI (milestone 1, heuristic)`.

### Task 9: Training and leave-one-game-out tuning — Milestone 2

**Files:**
- Create: `src/montreal/train.py`, `tests/test_train.py`
- Modify: `models/params.json` (written by the CLI)

**Interfaces:**
- Consumes: `load_sample_game`, `derive_labels`, `compute_features`, `SoftmaxRegression`, `heuristic_logp`, `decode_periods`, `cost_matrix`, `metrics.score`.
- Produces:
  - `simulate_dropout(trk: Tracking, rng) -> Tracking` (new arrays):
    - **Ball:** inside periods, mask the ball in blocks with lengths `int(exp(U(log 50, log 1500)))`, placed at uniform starts independent of labels. Keep adding blocks until the *newly masked* rows cover 25% of period rows.
    - *Why:* the sample games already miss the ball on ~40% of rows, ~75% of it during dead ball. Topping up to a total share would add almost nothing and teach "missing ⇒ dead". Label-independent extra masking is what breaks that shortcut.
    - **Players:** for each player slot, NaN-out random blocks of `U(25, 250)` rows until 5% of its period rows are masked.
  - `COST_GRID`: `swap ∈ {10, 20, 40, 80, 160}`, `dead ∈ {5, 10, 20, 40}`, `live ∈ {5, 10, 20, 40}`.
  - `CANDIDATES`: `"heuristic"`, plus `SoftmaxRegression(l2, cw)` for `l2 ∈ {1e-3, 1e-1}` × `cw ∈ {None, "balanced"}`.
  - `logo(games) -> pd.DataFrame`: one row per (candidate, swap, dead, live), with columns `spell_f1_g1/g2/g3`, `spell_f1_mean` and `macro_f1_mean`. For held-out game g:
    - fit on the other two games, with features from `simulate_dropout` copies (`rng = default_rng([20261117, g])`), training on every 5th row;
    - evaluate on the held-out game's dropout copy (`rng = default_rng([20261117, 100 + g])`) against its derived labels.
  - `select(table) -> row`: the highest `spell_f1_mean`; ties go to the higher `macro_f1_mean`, then to the earlier candidate/grid order.
  - CLI `python -m montreal.train`:
    1. prints the top-10 table and the heuristic's best row;
    2. refits the chosen candidate on all three games;
    3. writes `models/params.json` with `"use_tags": true` and `meta` holding the selection table summary (top-5 rows), the sample commit SHA and the seed.

    If the heuristic wins, it writes `"emission": "heuristic"` with the chosen costs.

- [ ] **Step 1: Write failing tests:**
  - `test_simulate_dropout_share_and_determinism`: start from a synthetic 10k-row `Tracking` with the ball visible everywhere. The masked share inside periods lands in [0.25, 0.30]. If 30% is already missing, the newly masked rows still cover 25% or more of period rows. The same rng seed gives the same mask, and the input is not mutated.
  - `test_select_tiebreak`: in a hand-built table, equal `spell_f1_mean` is decided by `macro_f1_mean`.
  - `test_logo_runs_on_tiny_synthetic_games`: three tiny synthetic games, with the grid monkeypatched to 2 points, give a table with the expected columns and finite values.
- [ ] **Step 2:** Run `.venv\Scripts\python -m pytest tests/test_train.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Cache per-game features in `.cache/`. Each fold is independent of the others.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then run `.venv\Scripts\python -m montreal.train` (expect it to take minutes). Report the table to the controller.
- [ ] **Step 5:** Run `.venv\Scripts\python -m montreal.run`. Expected: validation OK.
- [ ] **Step 6:** Commit `models/params.json` and the code with message `feat: leave-one-game-out training and tuning (milestone 2)`. The selection is by sample-game LOGO only.

### Task 10: Reproducibility, README, technical document

**Files:**
- Create: `README.md`, `docs/Montreal.md`, `scripts/check_reproducible.py`
- Output (not committed): `submissions/Montreal.csv`, `docs/Montreal.pdf`

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1:** Write `scripts/check_reproducible.py`. It runs `python -m montreal.run --out` into two temporary paths in fresh subprocesses, compares the SHA-256s, prints them, exits 1 if they differ, and deletes the temporary files.
- [ ] **Step 2:** Run it. Expected: two identical SHAs.
- [ ] **Step 3:** Write `README.md`. It covers:
  - prerequisites: clone the upstream repo, unpack the data into it (or set `MTAH_DATA_DIR`), Python 3.12;
  - setup commands;
  - `python scripts/fetch_sample_data.py`;
  - `python -m montreal.train` (optional, because `models/params.json` is committed);
  - `python -m montreal.run`, which produces `submissions/Montreal.csv`;
  - the expected SHA-256;
  - tests: `pytest`;
  - an explicit statement that no match data is in the repo.
- [ ] **Step 4:** Write `docs/Montreal.md` (about 3–5 pages), with the SUBMISSION.md sections and real numbers from Tasks 8–9:
  - approach and why;
  - assets used (tactical ATD, Smart Tagging) and dropped (videos, broadcast ATD and homography, tactical homography), with reasons;
  - training data: Metrica sample games 1–3, labelled by Metrica (human event data on other matches), with frame counts;
  - possession rules (Task 3 rules + decoder);
  - how missing and noisy tracking is handled;
  - limitations;
  - reproduction.
- [ ] **Step 5:** Render the PDF: convert the Markdown to HTML with a small stdlib script, print it to `docs/Montreal.pdf` with headless Edge `--print-to-pdf`, and check the PDF exists and is larger than 20 KB.
- [ ] **Step 6:** Run the full test suite and the upstream validator on `submissions/Montreal.csv`. Expected: all pass, `OK`.
- [ ] **Step 7:** Commit README, docs and the script with message `docs: README, technical document, reproducibility check`.
