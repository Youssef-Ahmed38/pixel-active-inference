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
