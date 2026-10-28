# Stage 5: measured chess strength — 14 October 2026

## Status

The combined candidate passed fresh value and paired-game promotion gates.
All 120 scheduled Stockfish ladder games completed, with no execution failures.
The one-second conditional engine-anchor estimate is **1037 Elo [863, 1134]**.
This is not a calibrated human rating. No unfinished game was scored as a draw.

## What changed

- Optional Numba compilation executes the existing generic geometry features,
  learned categorical regions, fitted tree traversal and safety aggregation.
  Constant-false fitted regions and their learned guards can be skipped before
  constructing full feature matrices. The NumPy reference remains available.
- Position values are evaluated in batches and cached within each search.
  Cached entries are board-value predictions, never repetition/history-dependent
  search scores. Iterative deepening retains its last completed result.
- Three value candidates were taught from new Stockfish depth-8 labels: an MLP,
  the project's existing recurrent HRM, and a faster regularized linear model.
  The selected linear value uses the original 391 features; its weights are
  learned, not authored piece values or square tables.
- A small 133→64→32→1 action prior learns Stockfish's preferred move using the
  taught attack field, token identities and generic relative coordinates.
  It orders learned-legal actions; it cannot override legality or terminal status.
- The selected configuration combines the linear value, learned prior,
  quiescence depth 2 and prior-PV ordering under a one-second move budget.
- `pai/chess/agent.py` consumes public `StateBelief` objects. It has no engine
  or chess-library import. Teacher/opponent execution lives in `ladder.py`.

Given: standard symbolic board tokens, colours, coordinates, counters/history,
action syntax, network architecture and classical search algorithms. Learned:
legal rules, effects/state updates, endings/claims, value weights and policy prior.
This is classical search over learned models, not an EFE chess objective.

The rules/state/endings checkpoints and original value checkpoint were not
retrained or overwritten. `runs/chess_strength_data/promoted_agent.json` records
the approved configuration and model hashes; the original value remains available
for comparison. Checkpoints and engine binaries are ignored by Git.

## Before/after, fresh validation

| Check | Incumbent | Combined candidate |
|---|---:|---:|
| Fresh clipped value MAE, 1,969 boards | 333.13 cp | 248.47 cp |
| Advantage-sign accuracy, 1,457 applicable boards | 84.35% | 91.97% |
| Equal-time paired match, 20 games | 1 win / 2 draws / 17 losses | 17 wins / 2 draws / 1 loss |

The final position corpus uses 25 teacher-game clusters. Game-averaged MAE gain
is **97.04 cp**, cluster-bootstrap 95% interval **[71.63, 123.21] cp**;
one-sided paired cluster sign test **p = 0.00000972**. It generated 2,000 boards;
31 duplicate/mirror/teaching-overlap boards were excluded. Individual positions
within a game are not treated as independent samples.

Promotion uses ten unseen openings and both colours, with **1.0 s per move for
both agents**, the same 1,000,000-node safety cap, and no execution failures or
unfinished games. Candidate score **0.90**, pair-bootstrap 95% score interval
**[0.80, 1.00]**; paired sign test **p = 0.001953**. The measured relative Elo
difference is **+381.70**, with bootstrap interval **[+240.82, +∞)**. This is a
difference against this incumbent/configuration, not an absolute human rating.
It does not isolate the effect of value, prior and quiescence individually.

Reports: `results/chess_strength_final_positions.json`,
`results/chess_strength_final_promotion.json`,
`results/chess_strength_final_promotion.jsonl`,
`results/chess_strength_approved_config.json`.

## Rejected and development-only candidates

| Candidate | Development value MAE | Development match evidence |
|---|---:|---|
| Original linear incumbent | 302.22 cp | reference |
| MLP, teacher data | 255.05 cp | not promoted |
| MLP, teacher + self-play data | 254.77 cp | 4/4 draws at equal 0.1 s |
| Existing HRM, 1,200 sampled teacher boards | 227.08 cp | 4/4 draws at equal 0.1 s |
| New linear value, no action prior | 214.63 cp | 6/6 draws at equal 0.2 s; 6/6 losses vs SF 1320 |
| Linear + action prior + quiescence | 214.63 cp | 4/4 wins vs incumbent at equal 1.0 s; only two opening pairs |

These development matches cannot substitute for the fresh promotion match.
The first HRM development value score is not an independently fresh selection
test: its small sample may overlap common validation boards. Its fixed two
segments do not train the HRM's chess halting head. It remains a separate candidate.
The prior's development teacher-top-one agreement was **26.6%**, not a strength
measurement. Historical rejected HRM/context/self-play reports remain intact.

Eight agent-only self-play games provided **510** teaching positions. The engine
connection for their labels opened only after all self-play decisions finished.
Truncated games supplied positions, not invented terminal targets.

## Seed registry and fixed books

- Teacher training: **10400–10403**, depth 8, 16,000 rows.
- Teacher validation: **20400–20401**, depth 8, 2,500 rows.
- Self-play training: **36400–36407**, 0.03 s / 128 nodes, at most 64 plies each.
- HRM sample/optimizer: **41400**; policy optimizer/subsample: **42400**.
- Development opening generators: **54400, 54410, 54420**.
- Final value corpus: **30400–30401**, depth 10; never used for tuning.
- Final promotion opening generator: **64400**, ten fixed roots, both colours.
- Final ladder opening generator: **74400**, ten fixed roots, both colours at each level/time.
- Bootstrap analyses use fixed seed **0**; fresh position analysis uses **51400**.

Root FENs are stored in immutable `.meta.json` books. Exact and mirrored teaching
roots are rejected. Game roots are declared FENs: pre-root repetition history is
unknown and excluded for both sides. Final positions/openings were not reused
for development. Normal play can naturally enter a previously taught position.
- Legality microbenchmark only: **84400**; not used to tune the candidate.

## Ladder and thinking-time curve

The completed full run is **120 games**: UCI Elo **1320, 1500, 1700**, each with
twenty games at **0.2 s** and twenty at **1.0 s**. Both sides receive the same
move-time target; Stockfish has no extra depth/node cap. Threads 1 / Hash 16 MB.
The same ten opening roots are used across conditions.

| Both-side move target | Stockfish configured Elo | Games | Candidate W / D / L | Score | Conditional anchor Elo | 95% interval |
|---|---:|---:|---|---:|---:|---|
| 0.2 s | 1320 | 20 | 1 / 2 / 17 | 0.100 | 938 | −∞ to 1105 |
| 0.2 s | 1500 | 20 | 0 / 1 / 19 | 0.025 | 864 | −∞ to 1064 |
| 0.2 s | 1700 | 20 | 0 / 0 / 20 | 0 | no finite point | −∞ to 1651 |
| 1.0 s | 1320 | 20 | 4 / 0 / 16 | 0.200 | 1079 | 808 to 1212 |
| 1.0 s | 1500 | 20 | 0 / 0 / 20 | 0 | no finite point | −∞ to 1451 |
| 1.0 s | 1700 | 20 | 1 / 0 / 19 | 0.050 | 1188 | −∞ to 1399 |

The all-loss rows have no finite MLE. Their nonzero uncertainty bounds use a
conservative opening-pair Hoeffding interval because an empirical all-zero
bootstrap is degenerate. Other per-rung intervals are pair bootstraps. The
game-independent Wilson interval is retained only as a labelled reference.

| Move target | Games across levels | Overall conditional Elo / 95% CI | Mean completed main depth | Mean inspected nodes | Mean / maximum search time |
|---|---:|---|---:|---:|---|
| 0.2 s | 60 | 904 / [615, 1037] | 0.456 | 33.84 | 0.206 / 0.598 s |
| 1.0 s | 60 | 1037 / [863, 1134] | 1.554 | 247.86 | 1.007 / 2.530 s |

The depth excludes bounded quiescence extensions. The point estimate rises by
133 Elo, but the intervals overlap: a statistically monotonic time/strength
relationship is not established. At one second the agent scored five wins and
55 losses across levels; it remains weaker than these opponents overall.

A separate warm legality microbenchmark on ten identical boards measured
**10.766 ms/board** for the NumPy reference and **1.279 ms/board** for compiled
inference: **8.42×** faster, with exactly equal legal sets. This isolates legality
computation, not complete search throughput or a universal laptop speedup.

Reports: `results/chess_strength_final_ladder.json` and `.jsonl`,
`results/chess_strength_protocol.json`,
`results/chess_strength_legality_benchmark.json`. The independent raw-game audit
replayed **140/140** final PGNs, verified legal moves and terminal/draw-claim
evidence, and recovered exactly 18 promotion points and 7.5 ladder points.

The overall point estimate solves the logistic expected-score equation against
configured engine anchors. Draws count as half a point; unfinished games do not
count. The bootstrap resamples an opening jointly across all levels, preserving
colour and cross-level dependencies. A null endpoint denotes an unbounded end;
all-zero/all-one data never produce a pseudo-count rating.

The output is **conditional engine-anchor Elo under these limits**. Stockfish's
configured numbers are not independently calibrated human ratings in this
short-time protocol. Both sides' budgets change across the curve, so this is an
equal-time benchmark family, not an isolated causal effect of agent thinking time.

Search budgets are soft deadlines checked between model operations. Fixed
compilation/setup warm-up uses the standard initial board outside move timing;
actual move elapsed times and inspected nodes/depths are retained in raw records.
The measured overruns above are not hidden; these are soft move-time targets,
not hard real-time clock guarantees. Root setup and an in-progress model
operation can finish after the target. A hard preemptive tournament clock is
not implemented, so the rating applies to this declared software protocol.
CPU contention, engine move randomization and the small number of independent
opening clusters limit precision. No game is adjudicated from the agent's score.

## Verification

All **146** `tests/test_chess_*.py` tests passed in the final full regression run
(330.51 s). The earlier candidate freeze also passed all 143 then-collected tests.
The original rule/state/endings gate tests were not edited. New tests cover
compiled/reference features, value I/O and colour symmetry, policy I/O,
scalar/batch equality, incomplete/duplicate pairs, endpoint statistics, and an
agent move selected while chess-library access and engine startup are blocked.
The final run includes the pure player interface, promotion-manifest integrity,
degenerate-bootstrap uncertainty and distillation trajectory/orbit checks.

## Reproduce

Build missing frozen checkpoints with the existing commands in `CHESS_RULES.md`,
`CHESS_STATE.md` and `CHESS_EVALUATION.md`. Install Numba for the optional fast
backend (`.venv\Scripts\python.exe -m pip install numba`). Download Stockfish 19
as described in the task brief; it is not committed. Run from the checkout root.

```powershell
.venv\Scripts\python.exe scripts/chess_distill.py generate --directory runs/chess_strength_data --positions 16000 --workers 4 --depth 8 --seed-offset 400 --split train
.venv\Scripts\python.exe scripts/chess_distill.py generate --directory runs/chess_strength_data --positions 2500 --workers 2 --depth 8 --seed-offset 400 --split val
.venv\Scripts\python.exe scripts/chess_distill.py train --directory runs/chess_strength_data --epochs 45 --cpu --output-model runs/chess_strength_data/value.npz --report results/chess_strength_distill_development.json
.venv\Scripts\python.exe scripts/chess_strength_hrm.py
.venv\Scripts\python.exe scripts/chess_strength_selfplay.py --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe
.venv\Scripts\python.exe scripts/chess_distill.py train --directory runs/chess_strength_data --train-rows runs/chess_strength_data/train_mixed.jsonl.gz --epochs 45 --cpu --output-model runs/chess_strength_data/value_mixed.npz --report results/chess_strength_distill_mixed_development.json
.venv\Scripts\python.exe scripts/chess_strength_linear.py
.venv\Scripts\python.exe scripts/chess_strength_policy.py
.venv\Scripts\python.exe scripts/chess_distill.py generate --directory runs/chess_strength_final --positions 2000 --workers 2 --depth 10 --seed-offset 400 --split test
.venv\Scripts\python.exe scripts/chess_strength_positions.py --candidate runs/chess_strength_data/linear_candidate.json
.venv\Scripts\python.exe scripts/chess_ladder.py promote --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --agent-value runs/chess_strength_data/linear_candidate.json --policy runs/chess_strength_data/policy.npz --times 1.0 --quiescence 2 --pairs 10 --opening-seed 64400 --workers 4 --reuse-pv --output results/chess_strength_final_promotion.jsonl
.venv\Scripts\python.exe scripts/chess_ladder.py ladder --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --agent-value runs/chess_strength_data/linear_candidate.json --policy runs/chess_strength_data/policy.npz --times 0.2 1.0 --rungs 1320 1500 1700 --quiescence 2 --pairs 10 --opening-seed 74400 --workers 4 --reuse-pv --output results/chess_strength_final_ladder.jsonl
.venv\Scripts\python.exe scripts/chess_strength_promote.py
.venv\Scripts\python.exe scripts/chess_strength_audit.py --games results/chess_strength_final_promotion.jsonl results/chess_strength_final_ladder.jsonl
.venv\Scripts\python.exe scripts/chess_strength_protocol.py --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe
.venv\Scripts\python.exe scripts/chess_strength_benchmark.py --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe
$env:OMP_NUM_THREADS='1'
.venv\Scripts\python.exe -m pytest (Get-ChildItem tests/test_chess_*.py).FullName -q
```

Match metadata records model hashes, source fingerprint, commit, seeds, controls
and command. Raw JSONL appends/flushes each finished game. Resumption rejects
changed code/checkpoints/books rather than mixing runs. Engine randomization and
time-based search make replayed game outcomes nondeterministic; statistics can
be recomputed exactly from the retained raw games.

Load the validated configuration without an engine:

```python
from pai.chess.agent import AgentPlayer
agent = AgentPlayer.from_manifest("runs/chess_strength_data/promoted_agent.json")
command, diagnostics = agent.command_from_belief(public_belief)
```

An unapproved manifest or changed checkpoint is rejected. If retraining produces
different hashes, rerun the gates and publish a new manifest rather than reusing
this acceptance record. The first final-promotion summary hit a NumPy-boolean
serialization error after all twenty games were saved; the serializer was fixed
and the same raw games were summarized, without replaying or replacing them.

## Adaptive HRM in search with learned halting (2026-10-28)

This is a separate opt-in candidate. The previous promoted configuration and checkpoints remain intact. The paired match tests playing strength; lower teacher error alone is not a promotion gate. This study uses **0.2 seconds per move**, equally for both sides. Its ratings are conditional on that protocol and must not be substituted for the historical 1-second 1037 Elo estimate.

### Design: learned versus given

`pai/chess/adaptive_hrm.py` reuses the existing HRM encoder, coupled recurrent L/H modules, readout and two-logit halt head. Rank patches of eight board squares reduce the sequence to eight evidence tokens plus the readout. A 32-dimensional residual head refines the promoted linear value. Colour-mirrored views are evaluated independently and antisymmetrized, preserving colour symmetry. Alpha-beta calls this value at nonterminal search nodes through the existing batched, cached value interface; the same learned move prior orders moves for both players. The halting histogram in each move record counts actual executed recurrent segments, excluding cached values and fixed initial-board warm-up.

Given: public symbolic board tokens/context, rank patching, architecture, six-segment safety cap, 500-cp residual scale, search algorithm and budgets, a 2-cp cost per additional segment, and Stockfish labels. Learned: residual encoder/recurrent/readout weights and the halt classifier. The latter is supervised from when **future teacher-error reduction exceeds added recurrence cost**. Labels are detached; the backbone is frozen while fitting the halt head. No teacher error, legal oracle or engine output enters inference. This is supervised stopping, not a claim of chess reinforcement Q-learning. Existing cause HRM behavior is unchanged.

Backbone selection uses validation full-depth error; halt-head selection uses validation error plus 2 cp × mean segments. Training samples come only from the old depth-8 training corpus; train/validation board/mirror orbits are disjoint. Validation is development data previously used in chess work. Final data were not used for fitting or selection.

The fit retained 3981 unique teaching orbits and 800 development validation positions, with eight value epochs and 25 halt-head epochs.

### Fresh halting analysis

Seed 93400 generated 400 depth-8 teacher-labelled positions; 395 survived teaching-orbit/duplicate exclusion. This positional test is separate from the opening books.

| Measure | Frozen candidate |
|---|---:|
| Mean segments (cap six) | 4.297 |
| Recurrent segments saved versus fixed six | 28.4% |
| Easiest baseline-error quartile | 3.990 |
| Hardest baseline-error quartile | 4.672 |
| Baseline error / steps Pearson correlation | 0.341 |
| Baseline / adaptive / fixed-six MAE | 210.49 / 206.71 / 206.98 cp |

Executed-view segment histogram: `{'1': 0, '2': 2, '3': 182, '4': 186, '5': 419, '6': 1}`. Each position contributes two views. The hardest quartile is defined by baseline absolute teacher error, only for scoring. These are descriptive position statistics: trajectories are correlated, and the segment saving is not an end-to-end search speedup.

During the actual matches/ladder, 329750 uncached HRM views used mean 4.140 segments; histogram `{'1': 20, '2': 6591, '3': 78003, '4': 108195, '5': 136507, '6': 434}`.

### Equal-time paired match

Opening seed **97400**, ten roots, both colours, 20 completed games. Adaptive candidate W/D/L: **1/8/11**.

Score 25.0%, reported opening-pair 95% interval [12.5, 37.5]%; one-sided paired sign test p=1.

The candidate fails the paired promotion gate and is **not promoted**. This negative strength result is retained despite positive halting and positional-error results.

### Enlarged Stockfish ladder

Opening seed **98400**, twenty new roots reused across both players and all three anchors. Each player has **40 games per rung**, twice the previous twenty at this move target. The engine has Threads 1 / Hash 16 MB / UCI_LimitStrength; no extra engine node/depth cap. Both agents retain quiescence two, PV reuse, the same prior, and the million-node safety cap.

| Player | UCI anchor | Games | W/D/L | Score / reported pair 95% interval | Conditional anchor Elo / 95% interval |
|---|---:|---:|---|---|---|
| baseline | 1320 | 40 | 4/0/36 | 10.0% / [0.0, 22.5]% | 938 [-∞, 1105] |
| baseline | 1500 | 40 | 1/0/39 | 2.5% / [0.0, 7.5]% | 864 [-∞, 1064] |
| baseline | 1700 | 40 | 0/0/40 | 0.0% / [0.0, 30.4]% | no finite point [-∞, 1556] |
| adaptive | 1320 | 40 | 2/1/37 | 6.2% / [0.0, 13.8]% | 850 [-∞, 1001] |
| adaptive | 1500 | 40 | 0/1/39 | 1.2% / [0.0, 3.8]% | 741 [-∞, 936] |
| adaptive | 1700 | 40 | 1/2/37 | 5.0% / [0.0, 11.2]% | 1188 [-∞, 1341] |

| Player | Overall conditional engine-anchor Elo / 95% interval |
|---|---|
| baseline | 904 [615, 1037] |
| adaptive | 904 [738, 1005] |

Both point estimates are equal because the players earned the same total ladder points under the same anchor exposures. The distribution over openings differs, producing different bootstrap intervals. This noisy ladder result does not overturn the negative direct paired match.

These use the existing anchored-score equation and opening-cluster bootstrap, preserving both colours and cross-rung dependencies. Degenerate per-rung bootstraps use the existing conservative pair Hoeffding bounds; null Elo endpoints mean unbounded. Configured anchors are not human-calibrated ratings.

| Player / anchor | Mean nodes | Mean main depth | Mean search seconds | Maximum search seconds |
|---|---:|---:|---:|---:|
| baseline / 1320 | 89.3 | 1.033 | 0.203 | 0.675 |
| baseline / 1500 | 95.4 | 0.993 | 0.203 | 0.521 |
| baseline / 1700 | 93.3 | 0.954 | 0.204 | 0.547 |
| adaptive / 1320 | 32.2 | 0.436 | 0.205 | 0.486 |
| adaptive / 1500 | 32.0 | 0.330 | 0.206 | 0.599 |
| adaptive / 1700 | 33.7 | 0.350 | 0.204 | 0.400 |

### Failures, scope, and verification

All 260 scheduled PGNs were replayed independently for legal moves, results, and draw-claim witnesses. Unresolved games by group: `{'paired': 0, 'baseline_elo1320': 0, 'baseline_elo1500': 0, 'baseline_elo1700': 0, 'adaptive_elo1320': 0, 'adaptive_elo1500': 0, 'adaptive_elo1700': 0}`.

The models are frozen; a time-based game cannot be replayed bit-for-bit. Soft deadlines can overrun during model operations. CPU contention and limited independent opening clusters remain limitations. A 1024-ply cap is an execution ceiling, never draw adjudication. No failure or unfinished game is silently converted to a score. Positive halt/error results do not establish a playing-strength gain. No promoted manifest was changed, regardless of this candidate’s result.

Main depth excludes quiescence. Halt targets optimize positional teacher error plus recurrent cost, not search win rate. Selective reasoning or a learned move-ordering head would need a new development study; this completed final set must not become its tuning set. No 1-second HRM ladder was run.

Restricted Windows named pipes prevented python-chess asynchronous UCI and multiprocessing queues. The new scripts explicitly opt into `uci_sync.py` (ordinary pipes, bounded UCI limits, history preserved) and four separate worker processes with atomic gzip file checkpoints. The learned player never imports that transport. Per-worker locks prevent duplicate concurrent writers. Restarting adopts completed games; frozen code/checkpoint mismatches are rejected. No user/cache absolute paths are written to results.

Regression: 173 tests passed in the full required cohort; six focused adaptive tests also passed (the latter include one new UCI protocol test added after full collection). Tests exercise true halting compute, mirror symmetry, save/load, benefit labels, and an actual agent move with teacher imports and engine startup blocked.

The test wrapper uses a fresh directory under `data` and inherited Windows permissions instead of inaccessible `mkdir(mode=0700)` directories. Existing temporary directories are never removed; use a new basename for repeat checks. Result and training commands are intended for clean artifact paths: existing plans/models are deliberately protected. `results/chess_hrm_final_tests.json.gz` retains the final six-test focused run.

### Seeds and exact commands

Training/optimizer 91400; source teacher data 10400–10403 and development validation 20400–20401. Capped smoke 95400. Final halting positions 93400; paired openings 97400; ladder openings 98400. Old published opening books and teaching orbits are excluded during book construction.

```powershell
$env:PYTHONPATH='.'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe scripts/chess_hrm_train.py
.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py tests
.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py smoke
.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py halting
.venv/Scripts/python.exe scripts/chess_hrm_study.py register --seconds .2 --pairs 20
.venv/Scripts/python.exe scripts/chess_hrm_study.py run --workers 4
.venv/Scripts/python.exe scripts/chess_hrm_study.py analyse
.venv/Scripts/python.exe scripts/chess_hrm_evaluate.py audit
.venv/Scripts/python.exe scripts/chess_hrm_report.py
```

Results: `results/chess_hrm_training.json.gz`, `chess_hrm_final_halting.json.gz`, `chess_hrm_search_plan.json.gz`, `chess_hrm_search_games.json.gz`, `chess_hrm_search_summary.json.gz`, `chess_hrm_final_audit.json.gz`, `chess_hrm_report.json.gz`. The model is `results/chess_hrm_adaptive_model.pt`. All per-game PGNs, player configs, move timings and executed halting histograms are retained in the compressed game archive.

`results/chess_hrm_completion_audit.json.gz` checks frozen identities, all game slots, teaching overlap, path privacy, file-size policy and the 174 distinct passing tests across the full and final focused runs.
