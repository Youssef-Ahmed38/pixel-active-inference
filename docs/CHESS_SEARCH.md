# Bounded learned chess search and draw claims

The agent now searches multiple moves ahead over its learned state model and
can execute a valid draw claim through the shared cognitive skill interface.
This composes the existing [rules](CHESS_RULES.md), [state transitions](CHESS_STATE.md),
and [ending/value evaluation](CHESS_EVALUATION.md).

## Search

`LearnedChessSearch.search(GameState, max_depth=3, max_nodes=256, time_limit=None)`
uses iterative deepening, negamax and full-width alpha-beta pruning. Learned
position value orders moves and evaluates nonterminal leaves; learned endings
score terminal wins, losses and draws. Mate scores prefer shorter wins and
longer forced losses. There is no teacher legality mask or chess-library call
in inference.

Successors use cached learned piece/metadata effects and advance counters and
history for the particular branch. History is kept separately on each branch;
the implementation has no board-only transposition score cache that could mix
different repetition histories or clocks.

Search returns a `SearchResult` with action, claim flag, score, completed depth,
visited nodes, exhaustion flag, elapsed time and principal variation. An aborted
iteration keeps the last fully completed iteration. If no iteration finishes,
a model-approved fallback remains available and its unevaluated score is `None`
(or zero for an available draw claim).

Node budgets count inspected search states, including repeated roots across
iterations. Root preparation, move ordering and model operations have additional
cost. The optional time limit is checked between nodes and can overrun by model
work already underway; it is not a hard real-time guarantee.

## Draw execution

Draw claims are optional zero-valued actions within search. Playable moves remain
available, so the agent can prefer a forced win to claiming a draw. Search uses
a separately teacher-fitted current-claim head to distinguish eligibility now
from eligibility requiring an intended move. That head has seven nodes and is
trained on the original ending curriculum, not the tactical evaluation boards.

`SearchResult.command()` returns either `move_piece(uci)` or `claim_draw(uci?)`.
For a claim before a move, search supplies a qualifying intended move. The
teacher validates the claim against its rule library without playing the
intended move. Board pieces, counters, and history remain unchanged when a
claim succeeds; the episode records the termination and a public `game_over`
relation. Subsequent move attempts are rejected. Automatic endings also prevent
further execution.

`LearnedChessSearch.command(StateBelief, **budgets)` provides the policy bridge.
Search, the original rule policy, and the one-move value policy abstain when the
belief reports that the episode is over. Direct `GameState` search represents
board/history state; callers should use the belief interface to honor an
already accepted external draw claim.

## Measured tactical checks (2026-10-10)

Fresh seeded suites generate two limited families: back-rank mates and a rook
capture that allows the opponent's queen to recapture the agent's queen. Exact
rule/state/value teaching boards are excluded. Seed 410 excluded one teaching
board; seed 411 excluded none.

| Measure | Seed 410 | Seed 411 |
|---|---:|---:|
| Evaluated boards | 7 | 8 |
| Legal choices / exact root move sets / valid predicted lines, each budget | 7/7 | 8/8 |
| Stayed within node budget, each budget | 7/7 | 8/8 |
| Mean completed depth at 32 / 128 nodes | 1 / 2 | 1 / 2 |
| Immediate mates solved at 32 and 128 nodes | 3/3 | 4/4 |
| Queen traps avoided at 32 nodes | 0/4 | 0/4 |
| Queen traps avoided at 128 nodes | 4/4 | 4/4 |

Both correctness gates passed at both budgets. Across suite instances, 128-node
search solved all 15 tactical checks. The 32-node runs solved the seven mates
but selected all eight poisoned rook captures. This measures a useful effect of
additional lookahead in these families; it does not establish that more thinking
improves every position or that the player has a particular rating.

All three draw-execution cases passed: a prospective fifty-move claim with an
intended move, a currently available fifty-move claim without one, and a
prospective threefold claim with the knight's intended return move. These same
three cases are repeated in both reports.

Tests compare completed depth-two scores with an independent teacher minimax
using the same learned scalar value. Additional checks cover node/time
exhaustion, incomplete-iteration retention, state immutability, winning despite
an available claim, invalid claims, automatic game termination, and search with
teacher/trainer imports blocked. **103 focused tests passed**. A further fresh
ending suite (seed 412) passed **13,806/13,806** outcome predictions,
**6,903/6,903** current-claim predictions and **499/499** generated repetition-key
checks. Exact teaching game states were excluded; related history/clock variants
are not independent samples. The simulator-wide suite was not rerun.

## Reproduce

Build the existing rule/state/value checkpoints using the linked instructions.
Older ending checkpoints need the newly supervised `current_claim` head:

```powershell
.venv\Scripts\python.exe scripts/chess_evaluation.py --eval-seed 412 --output results/chess_endings_search.json
.venv\Scripts\python.exe scripts/chess_search.py --eval-seed 410
.venv\Scripts\python.exe scripts/chess_search.py --eval-seed 411 --output results/chess_search_seed411.json
.venv\Scripts\python.exe -m pytest tests/test_chess_rules.py tests/test_chess_stage5.py tests/test_cognition.py tests/test_cognitive_extensions.py -q
```

Search loads the four JSON checkpoints under `runs/chess_rules/`. It needs no
Stockfish process or Jobs A/C at runtime. Reports are `results/chess_search.json`,
`results/chess_search_seed411.json`, and `results/chess_endings_search.json`.
Development seeds 400 and 405 retain the earlier checks and are not final gates.

## Remaining Stage 5 work

This is a bounded alpha-beta baseline. [Self-play and actual opponent matches](CHESS_PLAY.md)
are implemented and measured. A separate [MCTS baseline](CHESS_MCTS.md) is evaluated.
The [chess HRM and contextual value branches](CHESS_VALUE_UPDATES.md) are integrated
and evaluated separately. Demonstrated playing-strength improvement remains
unfinished. Finite generated suites cannot prove correctness
or strength over all chess positions.
