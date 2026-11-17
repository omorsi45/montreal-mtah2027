# Team Montreal: technical document

More than a Hack 2027, Phase 1 (possession labelling, FC Barcelona vs Athletic Club).
Code: <https://github.com/omorsi45/montreal-mtah2027>. Deliverables: `Montreal.csv`, `Montreal.pdf` (this document), the repository.

## Approach

For each of the 199,319 frames we output `H`, `A`, `D` or `X`. The score that ranks teams (spellF1) rewards whole possessions that start and end at about the right time. A per-frame classifier on its own produces flickering labels and many short, wrong possessions. So we split the problem in two.

1. **Per-frame evidence.** From the tactical tracking we compute 41 team-level features per frame. A multinomial logistic regression (softmax regression) turns them into log-probabilities for H, A and D.
2. **Segment decoding.** A three-state Viterbi decoder adds a cost for every change of state. Within each half it chooses the label sequence with the best total score, which gives clean, whole possessions.

We chose this over a hand-written rule state machine, because hard thresholds break in the long stretches where the ball is not tracked. We chose it over end-to-end video models, because we had no GPU and no frame-level labels for this match, and because the tracking already carries most of the signal. The model and its switch costs are tuned on other matches (below), never on this one.

The pipeline is deterministic: the only random generator is `numpy.random.default_rng(20261117)` and is used only when training, log-scores are rounded to 6 decimals before decoding, and two fresh runs give the same SHA-256.

## Assets used and dropped

| Asset | Decision | Reason |
|---|---|---|
| Tactical ATD (automatic tracking) | **Used** | Source of all per-frame features (players and ball in pitch coordinates). |
| Tactical Smart Tagging | **Used** | Weak team prior, and goal periods are dead. Weight calibrated without labels (see below). |
| Tactical video | Dropped | The core pipeline needs no pixels, and we had no GPU for our own ball detection. |
| Broadcast video, broadcast ATD, broadcast homography | Dropped | Not synchronised to the tactical timeline; mapping it is a project of its own. |
| Tactical homography | Dropped | The tracking is already in pitch coordinates. |

Only the published half boundaries (`games.half_video_blocks`) are used from the competition metadata. Everything outside the two halves is `X`.

## Training data

| Game | Source | Frames |
|---|---|---|
| Metrica sample game 1 | public sample data | 145,006 |
| Metrica sample game 2 | public sample data | 141,156 |
| Metrica sample game 3 | public sample data | 143,761 |

The data is Metrica Sports' public sample data (<https://github.com/metrica-sports/sample-data>, commit `e706dd506b360d69d9d123d5b8026e7294b13996`): anonymised tracking and event data of three **other** matches. The events were produced by Metrica; they are the only human-produced data we use. None of it is about the competition match. The repository fetches it with `scripts/fetch_sample_data.py` and does not contain it.

**Frame labels are derived from the events by fixed rules** (next section, implemented in `src/montreal/events.py`). **No hand labels were made by the team on any game.** The rules were fixed before training and not tuned. We trained on every fifth frame (about 86,000 frames over the three games) and fitted on the frames labelled H, A or D.

## Possession rules implemented

The rules turn Metrica's events into per-frame labels. They define what the model learns, so this is where the line on loose balls, challenges and restarts is drawn. Each half is processed on its own and starts dead (`D`).

- **The ball becomes dead** at a `BALL OUT` or `FAULT RECEIVED` event, and at the end of a `SHOT` whose result is a goal or out of play.
- **Play restarts** at the first `SET PIECE`, `PASS`, `CARRY`, `SHOT` or `RECOVERY` after the ball became dead. The team that performs it is in possession. The restart must come **strictly after** the frame of the dead-ball event. In game 3 a `CARRY` is logged at the very frame of 50 of 64 `BALL OUT` events. Without this rule the dead ball would last zero frames in those cases. With it, 70 of 76 first restarts are set pieces, a median 13 s later (aggregate check on the open data).
- **While play is live, possession changes** only when the *other* team performs a `PASS`, `CARRY`, `SHOT`, `SET PIECE` or `RECOVERY`, that is, a controlled action or a recovery.
- **Loose and contested balls** never change possession. `CHALLENGE`, `BALL LOST`, cards and every other event type are ignored by the rules, so a contested or loose ball stays with the last team, and a lone touch by the other team does not count.
- **Dead-ball ownership** is not modelled: the answer is always `D`.
- Frames outside the two halves are `X`.

Resulting labels on the three sample games, inside the halves: D share 0.387, 0.337 and 0.290; 323, 294 and 343 possessions; mean possession 11.0, 12.7 and 11.9 s.

**Decoder.** States are H, A, D. Costs per transition: H to A or A to H = `swap` = 10; alive to D = `dead` = 10; D to alive = `live` = 10. A half starts as if the previous state was D. Ties keep the previous state. These costs were chosen from a grid by the validation below. The decoder never outputs `X` on a graded frame, because under spellF1 an `X` equals repeating the previous label, so abstaining gains nothing.

**Smart Tagging prior.** Attacking phases of team T add evidence for T. Defending phases add evidence for the other team. Set pieces add evidence for the tagged team. Goal tags add evidence for `D` until the next kick-off (at most 90 s). Tags with team `N/A` or an unknown team add nothing. The tags are phases, not possessions, so they are never copied as labels, only added to the log-scores. Each family is weighted by the log-odds that its team vote agrees with **confident tracking evidence**: ball visible, one team's nearest player within 1 m and every opponent at least 3 m away, held for at least 1 s. This uses different frames of the same match and no labels. The weight is capped at 1.0 so the tags can tip uncertain frames but cannot overrule clear tracking. Calibrated weights: attack 1.000, defend 1.000, set piece 0.566. (Two weights sit at the cap; see the limitations below.)

## Handling missing and noisy tracking

In the competition tracking the ball is missing on about 41% of graded frames, in long blocks (median about 11 s). Players are tracked as slots that come and go. Our handling:

- **Always finite.** Every feature is finite on every frame, including halftime and the empty frames before kick-off. If nobody of a team is tracked, the distance to the ball is 30 m. Coordinates are only used when both x and y are present.
- **Ball gaps.** `ball_vis` tells the model when the ball is missing. Carry-over features give the last and next visible ball-to-player evidence and the time since and until it was seen (capped at 30 s). Window means over 0.5 s and 2 s of each signal smooth the rest.
- **Team shape.** Each team's centroid advance and velocity, the spread of both teams, the mean speed of all players and the share of nearly stationary players carry the stretches without a ball. The attacking direction of each half is derived from where the teams stand, not from any frame number.
- **Ball outside the pitch.** A ball more than 0.5 m outside the pitch box sets `ball_out`. This is direct evidence of a dead ball.
- **Simulated domain gap.** The sample games have clean tracking. When fitting and validating we hide the ball in random blocks (newly hidden rows reach 25% of each half, block length 2 to 60 s, log-uniform) and hide each player slot in random blocks (5% of each half, 1 to 10 s). The blocks are placed without looking at any label. Both the training copies and the held-out game get this treatment.

## Results

**Model selection** is leave-one-game-out (LOGO) on the three sample games: fit on two games, decode the third (with simulated dropout), score with the organisers' own scorer, repeat for all three. Five emission candidates (a hand-set heuristic and four softmax regressions) and a grid of 175 cost triples were compared, 2,625 evaluations in total.

| Model | spellF1 game 1 | game 2 | game 3 | spellF1 mean | macroF1 mean |
|---|---|---|---|---|---|
| Heuristic, no training (best costs: 40 / 10 / 10) | 0.3054 | 0.3112 | 0.3067 | 0.3078 | 0.5975 |
| **Softmax regression, l2 = 0.1, no class weights; costs 10 / 10 / 10 (submitted)** | **0.4500** | **0.4336** | **0.4676** | **0.4504** | **0.7735** |

The selected swap cost is not at the edge of the grid (2.5 and 5 score lower). The final model is refitted on all three games. These numbers measure agreement with labels derived by our own rules on other matches, so they are not comparable with a leaderboard on this match.

**The submission** has 208 possessions of mean length 23.0 s. Label counts: H 74,187, A 45,670, D 39,000, X 40,462. Smart Tagging weights: attack 1.000, defend 1.000, set piece 0.566. The file passes the organisers' validator. Its SHA-256 is `7e7c10bae5019dd290d8cf5cbe768de0f6741046d5088b6cff24deaca1ee1d6e`.

**Public 5-minute window, sanity check only.** Running the pipeline prints one line scored on the public window: spellF1 0.3571, macroF1 0.6010. We used it only to confirm that frames, file format and label meaning line up. It never selected any parameter, version or rule. As the organisers warn, 17 possessions cannot rank anything, and we did not read anything into the number.

## Limitations and future work

- **Domain gap.** The sample tracking is clean, the competition tracking is automatic. We only simulate the difference (label-independent ball and player dropout), so the real gap, such as fragmented tracklets, is not fully covered.
- **Long ball gaps.** With the ball missing for about 11 s at a time, the team-shape features carry those stretches and are much less precise than ball proximity. Boundaries inside such gaps are the least reliable part of the output.
- **Merged possessions.** The mean possession is 23 s against about 12 s in the labelled sample games. Some short possessions are probably merged, which costs spellF1 on both sides of a merge.
- **Weakly identified costs.** Dead and live costs trade off almost exactly (for example 5 / 20 and 20 / 5 score the same), so the split between them is not identified. Every dead-ball run pays both, so in practice mostly their sum matters.
- **Label convention.** Our labels follow the rules above, applied to Metrica's event vocabulary. The organisers' convention may differ in details (brief interceptions, when a dead ball starts). We have not aligned the two.
- **Smart Tagging weights hit the cap** for attack and defend. The calibration only measures agreement on confident frames, which are easier than the frames where the tags matter.
- **No video.** The video could give ball position in the gaps and the exact restart instant.

With more time we would: detect the ball ourselves in the tactical video to fill the gaps; learn a restart detector (players static, then ball on a restart spot, then movement); align the label convention with the organisers' guidelines; and add more training games.

## Reproduction

Prerequisites: Python 3.12, the organisers' repository cloned next to ours with the data bundle in its `data/` folder (or set `MTAH_REPO_DIR` and `MTAH_DATA_DIR`). The match data is not in our repository.

```
git clone https://github.com/omorsi45/montreal-mtah2027
cd montreal-mtah2027
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m montreal.run
```

This writes `submissions/Montreal.csv` (a few seconds once the tracking is cached) and prints its SHA-256, which should be the one above. `models/params.json` holds the trained model, so retraining is optional: `scripts/fetch_sample_data.py` fetches the public sample games and `python -m montreal.train` reruns the selection (about 5 minutes). `scripts/check_reproducible.py` runs the pipeline twice in fresh processes and compares the two SHA-256 digests; in our run both were equal and equal to the submitted file. Tests: `python -m pytest -q`.

## AI assistance and data handling

The code was written with AI coding assistants (Kiro CLI with Claude models). **No match data, videos, extracts or labels of the competition match were sent to any hosted language model.** Our scripts print aggregates only (counts, shares, scores and digests), and only those aggregates were ever shown to an assistant. The Metrica sample games are open data. The unit tests use synthetic data only.
