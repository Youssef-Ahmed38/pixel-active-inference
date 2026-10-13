# Chess HRM and contextual value updates

The incumbent rule/state/ending/value checkpoints are preserved. These are
separate candidate value heads integrated into alpha-beta and MCTS through the
same `predict` interface. Batched child scoring supports neural values without
changing legality, counters, history or optional draw handling.

## Chess HRM

`hrm_value.py` uses the existing `pai.thinker.hrm.HRM` with a new scalar value
head, 64 public board tokens and public turn/castling context. It predicts a
residual over the established linear value. Color-mirrored evaluation enforces
opposite value signs. It uses two fixed segments; the cause model's halting head
is not trained or reused as a chess halting policy.

The teacher labeled 968 positions at 2,000 Stockfish nodes each, seed 700. An
orbit hash keeps mirrored positions together in the internal train/validation
partition: 772 training, 196 validation. Validation boards may share game
trajectories. Epoch 11 was selected from 16 epochs by internal validation MAE.

Fresh seed 701, 240 positions: baseline MAE **379.9 cp**, one-segment HRM
**382.5 cp**, two-segment HRM **385.8 cp**. The candidate was rejected for value
improvement. The experiment does not demonstrate that more HRM computation
improves chess values. See `results/chess_hrm.json`.

Its separate search suite seed 702 passed legal choices, predicted variations,
node budgets and draw-claim execution. At 128 nodes it solved all four mates and
avoided all four queen traps. This verifies integration, not match strength.

## Contextual residual

`threat_value.py` aggregates the already learned attack field into 24 generic
category features: attacked occupancy, unprotected attacked occupancy, friendly
attack counts and aggregate attack activity. Coefficients are learned from
teacher scores. No piece's movement, material value or tactical preference is
assigned by this head.

Ridge strength is selected from four choices on the same internal teaching
partition, then tested on fresh seed 703. On 240 fresh positions, MAE improved
**337.5 → 315.4 cp** and RMSE **463.4 → 434.0 cp**. This admits the candidate to
trial; it does not promote it to the incumbent.

The paired tactical suite seed 706 shows why that distinction matters. At 128
nodes, baseline alpha-beta avoided **4/4** queen traps; contextual value avoided
**2/4**, and completed slightly less search depth under the same node cap. Both
solved four mates and passed legality/variation/budget/claim checks. Lower value
error has not established better move selection. Reports:
`results/chess_context_value.json`, `results/chess_context_search.json`,
`results/chess_search_seed706.json`.

The paired Stockfish 19 trial used configured UCI Elo 1320, a 1,000-node opponent,
128-node learner, and both colors from the same declared opening, seed 804.
Baseline lost both games. The contextual candidate lost as Black and drew as
White after resuming a 256-ply cap to completion at 280 plies. No capped game was
counted as a draw. Its two-game score was 0.25; the conditional relative Elo point
estimate is about -191 against this engine configuration, with an uninformative
score interval spanning 0–1. This is not a calibrated human rating or reliable
strength improvement. Engine move randomization is not paired between processes.

The value candidate remains separate because the tactical regressions and tiny
match sample do not satisfy the promotion gate. Reports:
`results/chess_context_baseline_matches.json`,
`results/chess_context_candidate_matches.json` (original cap),
`results/chess_context_candidate_completed.json` (completed continuation).

## Optional quiescence branch

Alpha-beta now supports bounded quiescence at its leaves. The learned successor
model identifies actions that reduce occupied cells. The taught safety model
selects full evasions when the state is unsafe. All extensions count against the
existing node budget. Incomplete iterations still preserve the prior completed
iteration; quiescence is disabled by default.

Fresh seed 707: six non-teaching tactical boards. At 128 nodes it solved two mates
and avoided three of four queen traps; at 32 nodes it avoided none. It is an
experimental branch, not a promoted strength improvement. Independent teacher
reference tests check its score on both colors, legal PVs and budget behavior.

## Search ordering and interruption recovery

The leaf value and move-ordering value can now be chosen separately with
`--ordering-checkpoint`. An optional `--reuse-pv` prioritizes the previous
completed iteration's principal variation, keyed by the exact action prefix.
These are ordering hints, not cached scores; both options preserve the node cap
and are experimental. Both switches and `--quiescence` work in match runs.
Checkpoint hashes and search settings must match when resuming.

Using baseline ordering with contextual leaves did not fix seed 706: only 1/4
traps were avoided at 128 nodes. On seed 708, baseline avoided 8/8 traps and
contextual value 7/8; PV reuse changed neither result. A subsequent shared
comparison on seeds 710–711 excluded teaching overlap and repeated boards:
15 boards, seven mates and eight traps. All four variants passed all correctness
checks. At 128 nodes all solved seven mates and avoided eight traps; at 32 nodes
baseline avoided none and contextual value one. This is limited tactical
evidence, not a promotion or playing-strength claim.

Long match runs save each completed game's PGN and atomically update the partial
JSON and CLI report. New reports store their full schedule: resume continues
truncated PGNs and runs previously unplayed slots without repeating saved games.
Legacy reports without a schedule still resume only saved truncated games.
An interruption test verifies recovery between the two colors. A real-engine
one-ply smoke trial and continuation verify the CLI; these capped games are
excluded from strength estimates. Changed search settings correctly reject resume.

Reports: `results/chess_search_comparison.json`,
`results/chess_context_baseline_ordering_seed706.json`,
`results/chess_schedule_smoke_resumed.json`.

The broader focused run exposed a Windows clock-resolution bug in very short
search deadlines (143 passed, one failed). Both alpha-beta and MCTS now use
`perf_counter`, the high-resolution platform clock, instead of the potentially
15 ms `monotonic` clock. The previously failing deadline check now passes;
four ordering/deadline checks also pass, including frozen coarse ticks for both
planners. The full 144-test selection was not rerun after this clock-only fix.

```powershell
.venv\Scripts\python.exe scripts/chess_search_compare.py
```

## Reproduce

```powershell
.venv\Scripts\python.exe scripts/chess_hrm.py --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe
.venv\Scripts\python.exe scripts/chess_context_value.py --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe
.venv\Scripts\python.exe scripts/chess_search.py --value-checkpoint runs/chess_hrm/candidate.pt --eval-seed 702 --output results/chess_hrm_search.json
.venv\Scripts\python.exe scripts/chess_search.py --value-checkpoint runs/chess_context/candidate.json --eval-seed 706 --output results/chess_context_search.json
.venv\Scripts\python.exe scripts/chess_search.py --quiescence 3 --eval-seed 707 --output results/chess_quiescence.json
```

The new value checkpoints support match evaluation via `--value-checkpoint`.
The original linear self-play fitter remains limited to the linear JSON value.
HRM is CPU-trained independently of Jobs A/C; its inference needs no teacher or
engine process. Context checkpoints include their learned attack table so they
can be reloaded without a teaching environment.
