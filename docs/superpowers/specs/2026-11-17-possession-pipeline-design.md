# Team Montreal — MTAH 2027 Phase 1 possession pipeline: design

## Intent

Win More than a Hack 2027 Phase 1. For each of the 199,319 frames of the tactical video of
FC Barcelona vs Athletic Club, output `H` / `A` / `D` / `X`. Ranked on **spellF1** (whole
possessions matched one-to-one by temporal IoU, thresholds 0.3–0.7), tie-break macroF1. Floor to
beat: 0.2433. The team (Montreal) submits; this project produces all three deliverables:
`Montreal.csv`, a public reproducible repository, and a short technical document.

Authoritative references (read them, do not restate them): the upstream repository cloned at
`C:\Users\Omar Morsi\Desktop\Projects\more-than-a-hack-2027` — `CHALLENGE.md`, `DATA.md`,
`EVALUATION.md`, `SUBMISSION.md`, `RULES.md`, and `evaluation/possession_metrics.py` (the metric
is defined by that file).

## Goals

1. A deterministic pipeline: provided data in → `submissions/Montreal.csv` out, byte-identical on
   every run, passing `validate_submission.py`.
2. Maximise full-match spellF1 by producing **whole, well-bounded possessions**: few spurious
   splits, every real stoppage detected, boundaries within ~0.5 s where evidence allows.
3. Every parameter chosen on data we are allowed to tune on (see Tuning protocol), and documented.
4. Public repo + technical document that let the organisers reproduce the exact file.

## Non-goals

- Rebuilding detection, tracking or calibration. No video processing in the core pipeline.
- Using the broadcast feed (not synchronised; mapping it is a project of its own).
- Dead-ball ownership (the answer is always `D`).

## Anti-goals (failure even if it "works")

- Anything that breaks RULES.md: hand labels or hand corrections on this match; any human-produced
  data about this match; constants tied to specific frames of this match (beyond the published
  half boundaries in `games.half_video_blocks`); choosing between versions using the public
  5-minute ground truth.
- A file the repository cannot regenerate byte-for-byte.
- Match data in the public repository, or raw match data printed into an LLM context (terms of
  use forbid sending data or extracts to hosted LLMs). Scripts print aggregates only.

## Constraints

- **Schedule (user's instruction):** ignore the deadline; build the whole thing, the user submits
  at the end. *Builder's call:* still ship in two milestones — (M1) pure B with hand-set weights,
  validated, reproducible, documented; then (M2) C on top. M1 is a complete fallback if C does
  not beat it. The repo must always reproduce the latest candidate file.
- Python 3.12 venv; numpy + pandas (+ pytest for tests). No kloppy, no sklearn, no GPU. Exact pins.
- Data lives in the upstream clone's `data/` (verified complete); the solution reads it via an
  `MTAH_DATA_DIR`-style setting and never copies it into the repo.
- Runtime target: full pipeline under ~10 minutes on this machine (24 threads, 31 GB RAM).
- Determinism across machines: fixed seeds, sorted iteration, and emission scores quantised
  (e.g. rounded to 1e-6) before decoding so BLAS float drift cannot flip a Viterbi tie.

## What the data actually looks like (verified, aggregates only)

- ATD raw line: `frame:slot;slot;...(46):ball`, each group `x,y,speed`, `NaN` when missing.
  Coordinates normalised 0–1 (105.3 × 68.0 m). Ball has **no owner or alive channel**.
- Ball missing on 40.7% of graded frames, in **150 long blocks** (median ≈ 11 s, p75 ≈ 21 s,
  p90 ≈ 34 s; 99.9% of missing frames are in gaps > 2 s). Interpolation is useless; long gaps
  need non-ball evidence.
- 4.2% of visible graded ball positions are outside the 0–1 box (direct out-of-play evidence).
- Smart Tagging: 638 phases, clock verified (`round(s*25)` = video frame). Team tag `N/A` on 36,
  including every throw-in.
- Public GT (correctness check only): possessions 0.8–42 s, dead-ball stretches 5–20 s.

## Approaches considered

- **A. Hand-written rule state machine.** Fast and explainable; hard thresholds, brittle in the
  long ball gaps. Rejected as the core; its rules become features.
- **B. Per-frame evidence scores for H/A/D + Viterbi decoding with switch costs.** Directly
  produces clean segments, which is what spellF1 rewards; degrades gracefully in ball gaps
  because team-shape and tagging evidence keep flowing. **Chosen.**
- **C. Learn B's emission weights and switch costs on other matches.** Principled, rule-compliant
  tuning on hundreds of possessions instead of 17. **Chosen as the add-on.**
- D. Own ball detection from video to fill gaps. Deferred (no GPU, large effort); stretch only.

The team chose **B with C**.

## Architecture

```
data/ (provided)          external/ (downloaded, gitignored)
   |                          |  Metrica sample games 1-3, pinned commit
   v                          v
 io.py  -> frames arrays    sample_io.py -> arrays + event-derived H/A/D labels
   |                          |
   +-----> features.py <------+          (same feature code for both sources)
              |
   tagging.py (our match only: phase priors, restart instants)
              |
           emission.py  (logistic model, weights from train.py, + tagging prior)
              |
           decode.py    (3-state Viterbi, switch costs, restart-team rule)
              |
           run.py  -> submissions/Montreal.csv  -> validate + public-window sanity print
```

### 1. Loading (`io.py`)
Stream the raw ATD once into numpy arrays — ball `(n,2)`, ball speed `(n,)`, players `(n,46,2)`,
speeds `(n,46)`, slot team from the metadata XML (first team = home). Cache as `.npz` outside the
repo. Video frame = `frame_id - 1`.

### 2. Features (`features.py`) — team-level, no player identities
Computed identically for our match and for the sample games:
- **Ball proximity:** distance (m) from ball to nearest H and nearest A player; counts of each
  team within 2 m / 5 m; which team is nearest, and for how many consecutive frames.
- **Control proxies:** same-team nearest for ≥ k frames; a pass completed between two players of
  one team (nearest-team stays constant across a ball flight).
- **Ball state:** visible; outside the pitch box (with a small margin); ball speed; ball
  stationary; ball on a restart spot (corner arcs, touchlines, goal areas, centre spot).
- **Collective motion:** mean player speed, share of near-stationary players, its change over
  ±2 s (play stops → everyone slows; restarts → acceleration).
- **Team shape (carries the ball gaps):** per-half attacking direction (derived from the data,
  e.g. team centroids at the half's start); each team's centroid along attack direction, its
  velocity, spread/compactness, defensive-line height. The attacking team is spread and
  advancing; the defending team is compact and retreating.
- Each feature also as short windowed means (e.g. ±0.5 s, ±2 s) so the model sees context.

### 3. Smart Tagging (`tagging.py`) — our match only
- **Attacking phases** (Build Up, Progression, Creating Chances, Attacking Transition, Ball in
  Final Third, Ball in the Box, Players in the Box) with team T → evidence for T in possession.
- **Defending phases** (Defending in Attacking/Middle/Defensive Third, Defensive Transition) with
  team T → evidence for the *other* team.
- **Set pieces:** restart instants → evidence for a D→alive transition near the tag; tagged team
  is the restarting team (N/A → decided by nearest player at the restart). Goals → D until the
  next kickoff.
- Used as an additive log-prior. It is never copied across as a label, because tags are phases,
  not possessions.
- **How its weight is set without labels and without circularity (builder's call):** estimate the
  tags' reliability on *confident* frames only — ball visible and one team clearly in control by
  tracking evidence (e.g. a player within ~1 m for ≥ 1 s) — then apply it as a log-likelihood
  ratio, `w = log(p / (1 − p))`, per tag family (attacking, defending, set-piece team). The
  estimate comes from independent evidence on different frames, so it is not tuned to agree with
  itself. Switch costs are tuned on sample games where no tagging exists; to keep the scale
  honest, the tagging term is capped so that it can tip uncertain frames (mainly inside ball gaps)
  but cannot by itself pay for a possession change against confident tracking evidence.
- Tag-agreement and possession-count checks are **diagnostics only**: they can flag a bug, they
  never override what the primary protocol picked.

### 4. Emission (`emission.py`, `train.py`)
Multinomial logistic regression (numpy, L2, fixed seed) from features → P(H), P(A), P(D).
- **Training data (C):** Metrica public sample games 1–3 (github.com/metrica-sports/sample-data,
  pinned commit; anonymised, human-produced events on *other* matches — allowed, documented).
  Frame labels derived from events (builder's call, documented in the technical document):
  - **Possession flips** at a `RECOVERY` by the other team, or at the other team's first
    controlled action (`PASS`, `CARRY`, `SHOT`, `SET PIECE`). `CHALLENGE`, deflections and a lone
    touch do not flip it — matching CHALLENGE.md's "contested or loose stays with the last team".
  - **`D`** starts at `BALL OUT` (the frame the ball leaves), at a foul (`FAULT RECEIVED`), or at a
    goal; it ends at the next `SET PIECE` start frame (the restart).
  - Sides mapped to H/A per game.
  - **Using the public 5 minutes for conventions:** CHALLENGE.md says the public GT is the
    reference for *how cases are labelled*, and RULES.md allows it for "labels meaning what you
    think". So it may be inspected **qualitatively** — whether brief interceptions count as
    possessions, whether `D` starts as the ball leaves or at the whistle — to choose the label
    *rules* above. No number (threshold, window, cost, weight) is ever fitted or compared on it.
- **Domain gap handling:** our ATD loses the ball in long blocks and has fragmented, duplicated
  tracklets. On training games, mask the ball in blocks sampled from our gap-length distribution
  (an allowed statistic of provided data, not labels), and randomly drop/duplicate player slots.
- Learned weights are committed as a small JSON file, so regenerating the submission needs only
  the provided data; `train.py` regenerates the weights from the public sample data.
- **Fallback (pure B):** if C does not beat hand-set weights in leave-one-game-out spellF1, ship
  hand-set weights and say so in the technical document.

### 5. Decoding (`decode.py`)
Viterbi over {H, A, D} per half on summed log-scores, with switch costs `c_HA` (possession
change), `c_alive→D`, `c_D→alive`. High `c_HA` is what produces whole possessions. Optional
minimum-duration states if the plain HMM still fragments. After a D run, the alive state that
follows is chosen by restart evidence (set-piece team, nearest player at the restart).
- **Graded range:** labels on the published half blocks only; `X` everywhere else.
- **No `X` on graded frames:** under spellF1 an `X` equals repeating the previous label, which the
  decoder can do explicitly; `X` would only cost macroF1.
- The second-half pre-kickoff stretch must come out `D` from evidence (players static, ball on
  centre spot, kickoff tag), not from a hard-coded frame.

## Tuning protocol (the rule-compliant part)

1. **Primary:** leave-one-game-out spellF1 on the three sample games (with simulated dropout),
   computed with the official `possession_metrics.py`. Chooses emission features/regularisation
   and switch costs.
2. **Secondary, our match, no labels:** agreement of the predicted team with Smart Tagging phase
   teams; predicted D→alive transitions vs set-piece instants; possession count and mean duration
   plausibility against football norms (not against the public GT's published counts).
3. **Public 5 minutes:** run only as a pipeline correctness check (alignment, label meaning, file
   validity) at smoke-test milestones and at the end. **Never used to pick between versions.** The
   result is reported, not optimised.

## Deliverables and repository

- Repo `montreal-mtah2027` (public, GitHub `omorsi45`, push only after the user confirms):
  `src/montreal/…`, `tests/`, `scripts/fetch_sample_data.py`, `models/emission.json`,
  `requirements.txt` (exact pins), `README.md` with the one reproduce command, `LICENSE` (MIT),
  `.gitignore` (data, cache, external, `.playwright-cli`, submissions except the final file).
  The upstream scorer (`games.py`, `evaluation/`) is vendored with MIT attribution so `run.py`
  can validate its own output.
- `submissions/Montreal.csv` — validated with the official validator.
- `docs/Montreal.md` → `Montreal.pdf` (rendered locally, e.g. browser print), covering what
  SUBMISSION.md asks: approach, assets used/dropped (homography and broadcast dropped, why),
  training data (Metrica sample games: source, labelled by Metrica, amount), possession rules,
  limitations, reproduction.
- **Reproducibility test:** two clean runs → identical SHA-256.

## Engineering practice

TDD per Superpowers: unit tests on synthetic arrays for parsing, every feature, the label
derivation, Viterbi on toy sequences, and the writer/validator; an end-to-end test that runs the
pipeline and validates the file. Tests and scripts print aggregates only — never raw match rows.

## Left to the builder

Exact feature list and window sizes, logistic-regression details, restart-spot geometry, cache
location, module boundaries beyond the sketch above, and PDF rendering method.
