# Learned complete chess transitions

The next Stage 5 foundation is implemented: the agent can advance the full public
game state using learned rules and metadata effects. Runtime inference uses NumPy
and saved JSON trees, with no python-chess or scikit-learn calls.

## Architecture

`GameState` wraps the existing `BoardObservation` with the halfmove clock,
fullmove number, and public position history. The observation format and original
rule checkpoint fingerprints stay compatible.

`LearnedChessStateModel` composes the existing learned legality/piece model with
eight teacher-fitted metadata heads: next turn, four castling flags, relative en
passant target, halfmove reset, and fullmove increment. Generic arithmetic applies
the predicted counter increments and target offset. No piece-specific castling,
pawn, or clock rule is authored into runtime inference.

The heads receive token identities, square identities, displacement and observed
permissions. Feature restrictions prevent geometric effects from depending on
unrelated absolute squares. Teacher-side fitting uses every legal transition in
the original training curriculum: 81,916 transitions from 3,815 nonterminal
teaching positions. All eight heads fit those labels exactly.

`predict(state, action)` preserves the entire state for rejected moves.
`successors(state)` enumerates model-approved actions and predicts their states
in a batch. Neither API accepts a teacher legality mask.

`GameState.to_belief()` and `from_belief()` connect counters and history to the
shared cognitive contracts. `ChessTeacher.observe()` exposes these public fields
and records them in the existing cognitive episode sequence. History advances
only when execution succeeds.

## Verified results (2026-10-10)

Fresh suites exclude positions in either the rule or metadata teaching corpus.
Each remaining board evaluates all teacher-legal moves against the complete next
state. Evaluation varies halfmove clocks over 0–199 and fullmove numbers over
1–499 to check counter extrapolation rather than memorized absolute values.

| Measure | Seed 210 | Seed 211 |
|---|---:|---:|
| Evaluated boards | 561 | 559 |
| Excluded teaching boards | 50 | 46 |
| Exact complete transitions | 13,652/13,652 | 13,509/13,509 |
| Errors in pieces, turn, rights, en passant, counters, history | 0 | 0 |
| Completed model rollouts | 10/10 | 10/10 |
| Exact rollout states | 400/400 | 398/398 |
| Exact complete legal sets during rollouts | 400/400 | 399/399 |

Rollouts start from unseen random-game roots and advance the model's own
predicted states. Their intermediate states are checked, not filtered out by
training fingerprints. One rollout reached a terminal position early, producing
one additional legal-set check without an additional move.

47 focused tests passed across chess rules, full state, the original baseline,
and shared cognition. Tests cover both colors, en passant creation/expiry/capture,
castling and rights loss, rook capture, promotion, counter updates, repeated
positions, rejected moves, portable checkpoint inference with teacher imports
blocked, and cognitive state round trips. The simulator-wide suite was not rerun.

The development seed 200 report is retained separately. It was checked before the
two final fresh suites; evaluation examples are never added to training.

## Reproduce

First build the rule checkpoint using [CHESS_RULES.md](CHESS_RULES.md), then:

```powershell
.venv\Scripts\python.exe scripts/chess_state.py --eval-seed 210
.venv\Scripts\python.exe scripts/chess_state.py --evaluate-only --eval-seed 211 --output results/chess_state_seed211.json
.venv\Scripts\python.exe -m pytest tests/test_chess_rules.py tests/test_chess_stage5.py tests/test_cognition.py tests/test_cognitive_extensions.py -q
```

Metadata checkpoint: `runs/chess_rules/state.json`. It is loaded alongside
`runs/chess_rules/rules.json`. Reports: `results/chess_state.json` and
`results/chess_state_seed211.json`.

## Remaining Stage 5 work

These finite suites establish measured correctness, not correctness on every
possible position. Input is a public symbolic standard-chess board, not pixels
or Chess960. History starts at the supplied root; a FEN cannot recover earlier
repetitions. Raw en passant targets are retained in history. Repetition
adjudication must account for whether an en passant capture is legally available,
rather than comparing raw observations directly.

[Learned ending evaluation and a position-value baseline](CHESS_EVALUATION.md)
are now implemented, along with [bounded search and draw-claim execution](CHESS_SEARCH.md).
Self-play and measured opponent strength remain unfinished. Jobs A and C are not required.
