# Montreal: possession pipeline for More than a Hack 2027, Phase 1

Team Montreal's entry. For each of the 199,319 frames of the tactical video of FC Barcelona vs
Athletic Club, the pipeline writes one label: `H` (home in possession), `A` (away in possession),
`D` (dead ball) or `X` (not graded). The output is `submissions/Montreal.csv`. The technical
document is [`docs/Montreal.md`](docs/Montreal.md) (the submitted PDF is `Montreal.pdf`, rendered from it).

**No match data is in this repository.** The competition data is licensed for the competition only
and is not redistributable. You must obtain `data/` from the organisers (see below). The repository
contains only code, tests, the small trained-parameter file `models/params.json`, and documents.

## How it works

The pipeline uses no video. It reads the tactical automatic tracking (ATD) and computes 41
per-frame, team-level features: how far the nearest home and away players are from the ball,
whether the ball is outside the pitch, how many players are nearly stationary, the shape and
advance of each team, and short-window averages of these. A softmax regression turns the features
into per-frame log-probabilities for H, A and D. Where the ball is missing, the team-shape and
carried-over features keep working. Smart Tagging adds a small, calibrated log-prior. A three-state
Viterbi decoder with switch costs then turns the noisy per-frame scores into whole possessions,
inside each half. Everything outside the published halves is `X`. The model and the switch costs
are chosen by leave-one-game-out validation on the three public Metrica sample games, never on the
competition match. See the technical document for the details.

## Prerequisites

- Python 3.12 and git.
- The organisers' repository, cloned next to this one, with the data bundle unpacked into its
  `data/` folder:

  ```
  Projects/
    more-than-a-hack-2027/      <- the organisers' challenge repository
      data/                      <- the match data bundle, from the organisers
    montreal-mtah2027/           <- this repository
  ```

  The code imports the official scorer (`games.py`, `evaluation/possession_metrics.py`) from that
  clone; it is never copied here. To use other locations set `MTAH_REPO_DIR` (the upstream clone)
  and/or `MTAH_DATA_DIR` (the data folder; default `<MTAH_REPO_DIR>/data`).

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/omorsi45/montreal-mtah2027
cd montreal-mtah2027
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

On Linux or macOS use `python3.12 -m venv .venv` and `.venv/bin/python`. Dependencies are pinned
in `requirements.txt` (numpy, pandas and their dependencies) and `requirements-dev.txt` (adds
pytest). The last command installs the `montreal` package itself (from `src/`) into the venv; it
is what makes `python -m montreal.run` work.

## Fetch the public training data (only needed for training and the slow tests)

```powershell
.venv\Scripts\python scripts\fetch_sample_data.py
```

This clones the Metrica Sports public sample data
(<https://github.com/metrica-sports/sample-data>) into `external/` at the pinned commit
`e706dd506b360d69d9d123d5b8026e7294b13996`. Set `MONTREAL_SAMPLE_DIR` to use another location.

## Train (optional)

```powershell
.venv\Scripts\python -m montreal.train
```

This repeats the leave-one-game-out search on the three sample games (about 5 minutes, one core) and
rewrites `models/params.json`. The file is committed, so you can skip this step: the submission
needs only the match data and `models/params.json`.

## Produce the submission

```powershell
.venv\Scripts\python -m montreal.run
```

This writes `submissions/Montreal.csv`, validates it with the official checks and prints
aggregate numbers only (label counts, number and mean length of possessions, the SHA-256 of the file).
It takes a few seconds once the tracking has been cached in `.cache/` (a first run parses the raw
tracking file, which takes a little longer). The run also prints one line labelled "sanity check
only": the score on the organisers' public 5-minute window, shown to confirm that frames and labels
are aligned. It is never used to choose any parameter.

Expected SHA-256 of `submissions/Montreal.csv`:

```
7e7c10bae5019dd290d8cf5cbe768de0f6741046d5088b6cff24deaca1ee1d6e
```

To check that the file regenerates byte for byte (two fresh runs, compared by SHA-256):

```powershell
.venv\Scripts\python scripts\check_reproducible.py
```

To validate with the organisers' script:

```powershell
.venv\Scripts\python ..\more-than-a-hack-2027\evaluation\validate_submission.py submissions\Montreal.csv
```

## Tests

```powershell
.venv\Scripts\python -m pytest -q -m "not slow"   # fast tests, synthetic data only
.venv\Scripts\python -m pytest -q                 # also the slow tests (need data/ and external/)
```

All unit tests use synthetic data only.

## Repository layout

```
src/montreal/
  paths.py       locations (env overrides) and import of the official upstream scorer
  tracking.py    Tracking container; readers for the FIFA/EPTS tracking and Metrica CSV; cache
  events.py      Metrica sample games: events -> per-frame H/A/D/X labels (the possession rules)
  features.py    41 per-frame team-level features
  emission.py    heuristic emission and numpy softmax regression
  tagging.py     Smart Tagging -> calibrated, capped log-prior
  decode.py      three-state Viterbi with switch costs, per half
  pipeline.py    features -> emission (+ prior) -> decode -> submission file
  run.py         CLI: python -m montreal.run
  train.py       CLI: leave-one-game-out training and switch-cost tuning
  metrics.py     thin wrapper around the official spellF1 / macroF1
scripts/
  fetch_sample_data.py   clone the public sample data at the pinned commit
  check_reproducible.py  two fresh runs, compare SHA-256
  render_pdf.py          technical document: Markdown -> HTML (stdlib) -> PDF (headless Edge)
models/params.json       trained emission weights and switch costs (committed)
docs/Montreal.md         technical document (source of the PDF)
tests/                   pytest suite
```

## Licence

MIT (see `LICENSE`). The licence covers this repository only and says nothing about the match data
or the Metrica sample data, which keep their own terms.
