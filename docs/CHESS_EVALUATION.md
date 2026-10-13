# Learned endings and position value

Stage 5 now has learned terminal/draw evaluation and a small learned position-value
baseline. Both compose with [learned rules](CHESS_RULES.md), the
[complete state model](CHESS_STATE.md), and the shared cognitive contracts.

## Ending evaluation

`LearnedChessEvaluator.status(state, claim_draw=False)` predicts the termination
reason and winner. It recognizes checkmate, stalemate, insufficient material,
the automatic seventy-five-move and fivefold-repetition draws, and ongoing games.
With `claim_draw=True`, it also evaluates fifty-move and threefold-repetition
claims, including a claim achievable with the next legal move.

The runtime has no chess-library or scikit-learn dependency. Its decisions use:

- Learned legal actions and king safety from the existing rule model.
- A teacher-fitted material tree using token counts and square parity, pooled
  across colors to avoid incidental color correlations.
- A teacher-fitted en passant action detector, combined with learned legality,
  to normalize repetition keys. A pinned or unavailable en passant capture does
  not distinguish two otherwise identical positions.
- Counts of normalized public positions in the available history.
- Learned outcome, winner and lookahead-eligibility trees. Claim checks use
  the learned state model for prospective states and move counters.

The material and outcome rules, thresholds, claim eligibility and reason
precedence are supervised labels, not authored chess rules in runtime code.
Generic token identity, geometry, counting, and history comparison are supplied
representations. Cached legal actions and successor templates avoid repeating
the same board inference when only clocks or claim mode differ.

The distinction between automatic and claimable endings follows the
[python-chess teacher API](https://python-chess.readthedocs.io/en/stable/core.html#chess.Board.outcome).
`claim_draw=True` means to evaluate the outcome if an eligible draw is claimed;
the [search and teacher skill adapter](CHESS_SEARCH.md) now execute valid
`claim_draw` commands, including an intended move when required.

Teaching uses 22,410 generated game states and 44,820 outcome examples across
both claim modes, 1,381 distinct material patterns, and 15,476 legal-action
examples for en passant detection. The resulting trees have 21 material nodes,
9 en passant nodes, 25 outcome nodes, 7 winner nodes and 5 lookahead nodes.

## Learned value

`LearnedPositionValue` is a linear baseline distilled from **Stockfish 19** on
2,789 teaching positions. The teacher receives a requested 2,000-node budget,
one thread and 64 MB hash. Labels are White-perspective centipawn scores clipped
to +/-2,000, including capped mate scores. The engine is used only for teaching
and measurement. The saved agent contains 391 learned coefficients.

Features describe signed token counts, token-square occupancy canonicalized
across colors, and the side to move. Ridge regression fits the coefficients
without an intercept, preserving exact color-mirror antisymmetry. No manual
piece values or piece-square table is used in inference.

`choose()` ranks the learned successors one move ahead, handling terminal
wins/losses and draws before position value. `command()` converts the selected
action into the project's existing `move_piece` skill using `StateBelief`.
Tests show that terminal evaluation selects immediate checkmates for both colors
even when the scalar value weights are zero.

## Verified fresh suites (2026-10-10)

| Measure | Seed 310 | Seed 311 |
|---|---:|---:|
| Exact ending predictions | 13,572/13,572 | 13,734/13,734 |
| Excluded exact teaching game states | 1,161 | 1,143 |
| All generated unique repetition keys matched | 489/489 | 484/484 |
| Fresh engine value positions | 328 | 342 |
| Value mean absolute error, centipawns | 360.5 | 368.7 |
| Predict-even baseline error, centipawns | 640.1 | 662.1 |
| Advantage direction accuracy, magnitude >= 100 cp | 84.2% | 87.7% |
| Legal one-move policy choices | 24/24 | 24/24 |
| Policy improved / tied / worse versus first legal action | 18 / 0 / 6 | 19 / 1 / 4 |

The ending suites include all eight labels, both claim modes, counter boundaries,
material combinations, and histories before/after claimable and automatic
repetition. All **27,306** ending predictions passed. Exact teaching game states
are excluded; a board can recur with different counters or history. Repetition
key checks cover all generated unique boards, including excluded states.
Related clock variants and adjacent game/history positions are not independent
samples. Development seeds 300-304 are retained separately and informed fixes.

The value model beats the predict-even baseline in both fresh suites. Its errors
remain several pawns, and its move policy made worse choices in 10/48 comparisons
against a weak first-legal-action baseline. The policy comparison uses fresh
Stockfish child-position scores at the same requested 2,000-node budget; it is
not a match result or rating. Neither a value score nor these checks establishes
strong chess play.

**93 focused tests passed**, including illegal-state preservation, complete
state transitions, outcome precedence, claims before moves, automatic repetition
precedence over optional claims, mixed material, legal versus pinned en passant,
JSON inference with oracle/trainer imports blocked, mirror symmetry, immediate
mate selection and shared cognitive integration. The simulator-wide suite was
not rerun for this chess build.

Two development issues were fixed: color/parity correlations in a sparse material
classifier, and inconsistent synthetic counters when the teacher replayed history.
Training now pools material evidence across colors and creates coherent history
stacks for clock variants; outcome queries operate on copies.

## Reproduce

Build the rule and state checkpoints first using their linked instructions.
Download the teacher from the [official Stockfish release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19).
The local teacher used here is under `runs/chess_teacher/stockfish/` with its
distribution license. It is not part of the agent's runtime checkpoint.

```powershell
.venv\Scripts\python.exe scripts/chess_evaluation.py --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --eval-seed 310
.venv\Scripts\python.exe scripts/chess_evaluation.py --evaluate-only --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --eval-seed 311 --output results/chess_evaluation_seed311.json
.venv\Scripts\python.exe -m pytest tests/test_chess_rules.py tests/test_chess_stage5.py tests/test_cognition.py tests/test_cognitive_extensions.py -q
```

Endings checkpoint: `runs/chess_rules/endings.json`.
Value checkpoint: `runs/chess_rules/value.json`.
Reports: `results/chess_evaluation.json` and `results/chess_evaluation_seed311.json`.
Endings can be trained/evaluated without passing `--engine`.

## Remaining work

These are measured finite suites for symbolic standard chess. A FEN alone omits
prior repetition history. The teacher's material adjudication does not prove all
possible dead positions, such as fortresses; this build matches its supported
material rules rather than solving arbitrary reachability.

Bounded multi-move search and draw-claim execution are now implemented in
[CHESS_SEARCH.md](CHESS_SEARCH.md). Next are self-play improvement and actual
Stockfish matches. HRM search integration,
opponent-level evaluation, thinking-budget comparisons, and measured Elo remain
unfinished. Jobs A and C are not required by this chess build.
