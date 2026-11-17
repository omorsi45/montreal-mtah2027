# Team Montreal: technical document

More than a Hack 2027, Phase 1 (possession labelling, FC Barcelona vs Athletic Club).
Code: <https://github.com/omorsi45/montreal-mtah2027>. Deliverables: `Montreal.csv`, `Montreal.pdf` (this document), the repository.

## Approach

For each of the 199,319 frames we output `H`, `A`, `D` or `X`. The score that ranks teams (spellF1) rewards whole possessions that start and end at about the right time. A per-frame classifier on its own produces flickering labels and many short, wrong possessions. So we split the problem in two.

1. **Per-frame evidence.** From the tactical tracking we compute 51 team-level features per frame (41 base features plus 10 turnover and restart features, see below). A multinomial logistic regression (softmax regression) turns them into log-probabilities for H, A and D.
2. **Segment decoding.** A three-state Viterbi decoder adds a cost for every change of state. Within each half it chooses the label sequence with the best total score, which gives clean, whole possessions.

We chose this over a hand-written rule state machine, because hard thresholds break in the long stretches where the ball is not tracked. We chose it over end-to-end video models, because we had no GPU and no frame-level labels for this match, and because the tracking already carries most of the signal. The model and its switch costs are tuned on other matches (below), never on this one.

The pipeline is deterministic: the only random generator is `numpy.random.default_rng(20261117)` and is used only when training, log-scores are rounded to 6 decimals before decoding, and two fresh runs give the same SHA-256.

## Assets used and dropped

| Asset | Decision | Reason |
|---|---|---|
| Tactical ATD (automatic tracking) | **Used** | Source of all per-frame features (players and ball in pitch coordinates). |
| Tactical Smart Tagging | **Used** | Weak prior on H against A only (never on alive against dead), and goal periods are dead. Fixed design weights, not fitted to any data (see below). |
| Tactical video | Dropped | The core pipeline needs no pixels, and we had no GPU for our own ball detection. |
| Broadcast video, broadcast ATD, broadcast homography | Dropped | Not synchronised to the tactical timeline; mapping it is a project of its own. |
| Tactical homography | Dropped | The tracking is already in pitch coordinates. |

Only the organisers' `games.py` is used from the competition metadata: the half boundaries (`games.half_video_blocks`) and the pitch size. Everything outside the two halves is `X`.

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

**Decoder.** States are H, A, D. Costs per transition: H to A or A to H = `swap` = 5; alive to D = `dead` = 40; D to alive = `live` = 40. A half starts as if the previous state was D. Ties keep the previous state. These costs were chosen from a grid by the validation below, on the public sample games only. The decoder never outputs `X` on a graded frame, because under spellF1 an `X` equals repeating the previous label, so abstaining gains nothing.

**Smart Tagging prior.** The tags only say which team has the ball once play is alive, so they move `H` against `A` and nothing else. Attacking phases of team T vote for T. Defending phases vote for the other team. Set pieces vote for the tagged team. Tags with team `N/A` or an unknown team vote for nobody. The tags are phases, not possessions, so they are never copied as labels. The three families (attack, defend, set piece) carry **fixed design weights** of 1.0 each (`TAG_WEIGHTS` in `src/montreal/tagging.py`). They were set a priori and are fitted to no data, because no external game carries Smart Tagging to tune them on. For each frame the weighted votes are summed into one net vote, which is capped at 1.0 in total (attack and defend tags describe the same phase from both sides, so they must not add up). The net vote shifts the frame's log-odds of H against A by that amount. The frame's total alive probability and its `D` score are left exactly as the model gave them, so a tag can never turn a stoppage into live play or the reverse. So a tag has no say on whether the ball is dead, and, because the net vote is at most 1 nat, it can only decide H against A where the tracking is within about one nat of undecided; a clearer preference for a team stays. The one exception is the goal tag, which adds 1.0 to the `D` score from the goal to the next kick-off (at most 90 s). Weights (fixed): attack 1.000, defend 1.000, set piece 1.000.

**Orientation check.** As an automatic check only, the run measures how often each tag family's team vote agrees with confident tracking evidence (ball visible, one team's nearest player within 1 m and every opponent at least 3 m away, held for at least 1 s). The run aborts if the attack tags agree less often than chance, which would mean home and away are flipped between the tags and the tracking. The figures are printed (attack 0.763 on 8,352 frames, defend 0.864 on 6,198, set piece 0.638 on 2,214), but they feed no weight and cannot change the output: the file is byte-identical whatever they are, unless the check aborts. An earlier version calibrated the weights on the match's own tracking (no labels); it was replaced to comply with the rule that all tuning uses external data.

## Handling missing and noisy tracking

In the competition tracking the ball is missing on about 41% of graded frames, in long blocks (median about 11 s). Players are tracked as slots that come and go. Our handling:

- **Always finite.** Every feature is finite on every frame, including halftime and the empty frames before kick-off. If nobody of a team is tracked, the distance to the ball is 30 m. Coordinates are only used when both x and y are present.
- **Ball gaps.** `ball_vis` tells the model when the ball is missing. Carry-over features give the last and next visible ball-to-player evidence and the time since and until it was seen (capped at 30 s). Window means over a centred window of 0.5 s either side (+/-12 rows) and 2 s either side (+/-50 rows) of each signal smooth the rest.
- **Turnover and restart evidence (features v2).** Ten features were added after the first model, and kept because they raised the validation score (below). `near_run_home` and `near_run_away` are how long (0 to 10 s) each team has been the nearest to a visible ball, which separates a lasting possession from a passing touch. `gap_flip` is 1 when the ball is missing and the nearest team differs just before and just after the gap (both by at least 2 m), so the ball changed hands unseen. `ball_still` (ball slower than 0.5 m/s) and `restart_spot` (ball on a corner arc, within 1.5 m of a touchline or goal line, on the centre spot, or inside a goal area) point to a stoppage or a restart. `ply_accel` is the mean player speed over the next 2 s minus the previous 2 s, which changes when play stops or resumes. Four window means (`ball_still`, `restart_spot` over 2 s either side; the two run lengths over 0.5 s either side) complete the set.
- **Team shape.** Each team's centroid advance and velocity, the spread of both teams, the mean speed of all players and the share of nearly stationary players carry the stretches without a ball. The attacking direction of each half is derived from where the teams stand, not from any frame number.
- **Ball outside the pitch.** A ball more than 0.5 m outside the pitch box sets `ball_out`. This is direct evidence of a dead ball.
- **Simulated domain gap.** The sample games have clean tracking. When fitting and validating we hide the ball in random blocks (newly hidden rows reach 25% of each half) and hide each player slot in random blocks (5% of each half, 1 to 10 s). The ball-gap lengths are **resampled from the public sample games' own missing-ball runs** of at least 50 rows inside the halves (299 gaps pooled over the three games: 10th percentile 102 rows, median 385, 90th percentile 1,261, maximum 3,292; `external_gap_lengths` in `src/montreal/train.py`), never from the competition tracking. The shares (25%, 5%) and the player block range are fixed design constants set a priori. The blocks are placed without looking at any label. Both the training copies and the held-out game get this treatment.

## Results

**Model selection** is leave-one-game-out (LOGO) on the three sample games: fit on two games, decode the third (with simulated dropout), score with the organisers' own scorer, repeat for all three. Eleven emission candidates (a hand-set heuristic and ten softmax regressions: l2 in 0.001, 0.01, 0.1, 1 and 10, each with and without balanced class weights) and a grid of 175 cost triples were compared, once with the 41 base features (v1) and once with all 51 (v1 + v2): 5,775 evaluations per feature set. v2 was to be kept only if its best mean spellF1 beat the best of v1.

| Model | spellF1 game 1 | game 2 | game 3 | spellF1 mean | macroF1 mean |
|---|---|---|---|---|---|
| Heuristic, no training (best costs: 20 / 5 / 10) | 0.3108 | 0.3071 | 0.3444 | 0.3208 | 0.6090 |
| Best with the 41 base features (v1): softmax, l2 = 0.001, no class weights; costs 5 / 40 / 40 | 0.4484 | 0.4825 | 0.4984 | 0.4764 | 0.7836 |
| **Best with 51 features (v1 + v2), submitted: softmax, l2 = 0.001, no class weights; costs 5 / 40 / 40** | **0.4877** | **0.4981** | **0.4971** | **0.4943** | **0.7889** |

The best v2 score (0.4943) is higher than the best v1 score (0.4764), so v2 is submitted. The gain is small and the three folds are few: the next four v2 configurations (the same l2 with balanced class weights, or costs such as 5 / 5 / 20) score 0.4921 to 0.4938, within 0.003 of the winner, so the exact configuration among them is not significant. The v2 features raise the score across the top configurations, though, not only in the winner. The swap cost (5) is not at the edge of its grid. The dead and live costs (40 each) sit at the top of their grid, and l2 (0.001) at the bottom of its grid, so a wider search might move them. The final model is refitted on all three games. These numbers measure agreement with labels derived by our own rules on other matches, so they are not comparable with a leaderboard on this match.

**Smart Tagging, with and without (label-free check).** Because the tags only move H against A, they hardly change which frames are dead (the goal term is the only exception). With the submitted model, the share of graded frames labelled D is 0.324 without tags and 0.336 with them. The number of possessions goes from 298 (mean 14.4 s) to 240 (mean 17.6 s): the tags mostly stop short flips between H and A. The earlier additive version of the prior had pushed D down to 0.246 of the graded frames, which is why it was replaced. We did not compare versions with the public window.

**The submission** has 240 possessions of mean length 17.6 s. Label counts: H 62,720, A 42,774, D 53,363, X 40,462. Smart Tagging weights (fixed): attack 1.000, defend 1.000, set piece 1.000. The file passes the organisers' validator. Its SHA-256 is `7e47ef64d6d6de764e68539b94a4f122d9c4bc73db1aa3f8418f27265bbf2451`.

**Public 5-minute window, sanity check only.** Running the pipeline prints one line scored on the public window: spellF1 0.6000, macroF1 0.7001. We used it only to confirm that frames, file format and label meaning line up. It never selected any parameter, version or rule. As the organisers warn, 17 possessions cannot rank anything, and we did not read anything into the number. During design we also looked at the public window's possession and dead-ball durations, only to understand how cases are labelled, as the rules (RULES.md) allow. No parameter was derived from it.

## Compliance with the competition rules

How each rule of the organisers is respected, and where in the code.

- **No checking against the video.** The pipeline reads no video, and no output was ever checked against the video. The videos were not downloaded. The pipeline reads tracking, Smart Tagging and the published half boundaries only (`src/montreal/pipeline.py`).
- **No hand labels or manual data on the provided game.** None were made. The training labels come from fixed rules applied to Metrica's public events on other matches (`src/montreal/events.py`).
- **No upload of the provided data to AI services.** See "AI assistance and data handling" below, including the two small exceptions.
- **No training, fitting or calibrating on the ground truth or the video.** Every learned or tuned number comes from the three public Metrica sample games, by leave-one-game-out validation (`src/montreal/train.py`): the model weights, l2, class weights, the feature set (v1 or v2) and the three switch costs. An earlier version calibrated the Smart Tagging weights on the match's own tracking (no labels); that calibration was removed.
- **Fixed design constants** (not tuned on any data): the Smart Tagging weights (1.0 each) and net-vote cap (1.0), the goal term (1.0, at most 90 s), the dropout shares (ball 25%, players 5%) and the player block range (25 to 250 rows) in `src/montreal/tagging.py` and `src/montreal/train.py`, and the feature thresholds (for example 1 m / 3 m for confident possession, 0.5 m/s for a still ball, 0.5 m outside the pitch) in `src/montreal/features.py` and `src/montreal/tagging.py`. The simulated ball-gap lengths are resampled from the sample games' own gaps. Where these constants came from, precisely: the Smart Tagging weight and cap of 1.0 were written into the plan before any data was opened; the 25% extra ball masking and the 2 s minimum gap were set at design time with the organisers' published figure in mind (DATA.md: the ball is missing on 41% of graded frames) and the sample games' own missing-ball share; none of them was compared across values on any data.
- **Design changes made after looking at our own output on the match (no labels).** Two decisions were informed by label-free aggregates of our own output on the match, never by the ground truth: (1) the Smart Tagging prior was changed from additive to team-only after a code review argued it was structurally wrong (it pushed frames away from dead ball, which its weight never measured), and the D share of our output (0.246 against 0.33 to 0.40 on the sample games) was consistent with that; (2) Smart Tagging is used at all (`use_tags`), as planned in the design. Neither choice was tuned: there was no comparison of values and no score.
- **The public ground truth is read in one place**: the sanity line in `src/montreal/run.py`. It is printed, labelled sanity-only, and feeds no parameter.
- **What the provided data is used for:** input only (tracking, Smart Tagging, the published half boundaries and the pitch size), plus automatic checks: the organisers' validator, the orientation check (it can abort a run, never change its output) and the sanity line.
- **What we looked at during design.** The team inspected aggregate statistics of the provided tracking (ball-missing share and gap lengths) and the public window's segment durations, to understand the data and the labelling convention. No parameter of the submitted algorithm is derived from them.

## Limitations and future work

- **Domain gap.** The sample tracking is clean, the competition tracking is automatic. We only simulate the difference (label-independent ball and player dropout), so the real gap, such as fragmented tracklets, is not fully covered.
- **Long ball gaps.** With the ball missing for about 11 s at a time, the team-shape features carry those stretches and are much less precise than ball proximity. Boundaries inside such gaps are the least reliable part of the output.
- **Merged possessions.** The mean possession is 17.6 s against about 12 s in the labelled sample games. Some short possessions are probably merged, which costs spellF1 on both sides of a merge.
- **Weakly identified costs, and a search at its edges.** Dead and live costs trade off almost exactly (for example 5 / 20 and 20 / 5 score the same), so the split between them is not identified. Every dead-ball run pays both, so in practice mostly their sum matters. The selected 40 / 40 is at the top of the grid and l2 = 0.001 at the bottom of its grid, and the top configurations are within 0.003 of each other, so the exact choice is not significant with three folds.
- **Label convention.** Our labels follow the rules above, applied to Metrica's event vocabulary. The organisers' convention may differ in details (brief interceptions, when a dead ball starts). We have not aligned the two.
- **Smart Tagging weights are not tuned.** They are fixed at 1.0 (net vote capped at 1 nat) because no external game carries tags to tune them on. They may over- or under-weight a family. The agreement printed by the orientation check (set pieces 0.638, attack 0.763) hints that the families differ in reliability.
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

This writes `submissions/Montreal.csv` (a few seconds once the tracking is cached) and prints its SHA-256, which should be the one above. `models/params.json` holds the trained model, so retraining is optional: `scripts/fetch_sample_data.py` fetches the public sample games and `python -m montreal.train` reruns the selection (about 10 minutes on three cores with `--workers 3`). `scripts/check_reproducible.py` runs the pipeline twice in fresh processes and compares the two SHA-256 digests (with `--cold` each run also starts from an empty cache folder, `MONTREAL_CACHE_DIR`); in our runs both were equal and equal to the submitted file. We also reproduced the file from a fresh clone with a new virtual environment, following the commands above. Tests: `python -m pytest -q`.

## AI assistance and data handling

The code was written with hosted AI coding assistants. **No match files, videos, or tracking or tagging rows were uploaded to any AI service.** The assistants saw only aggregate statistics computed locally (counts, shares, percentiles, scores and digests) and file-format structure (tag names and field counts). Two small exceptions, stated plainly:

- the organisers' starter check `starter/load_smart_tagging.py`, run once to verify the clock, printed the two tagged kick-off intervals (four frame numbers);
- during design, the public 5-minute ground-truth file from the organisers' public GitHub repository was displayed as its list of 21 segments, to understand the labelling convention.

Implementation agents were instructed never to open the data files. The Metrica sample games are open data. The unit tests use synthetic data only.
