# Reliable rules from teacher examples

The original neural baseline classified sampled actions well but selected a legal move
on only 3/12 boards when it ranked the entire action space. The current rule learner
addresses that distribution mismatch directly. It uses relational features and induced
decision trees, with no chess-library calls at inference.

## What the agent learns

1. **Attacks:** the teacher labels 106,496 isolated-piece examples over all piece tokens,
   source squares, destination squares, and intervening-occupancy states. A fitted
   decision tree is compiled into a lookup table for fast inference. Its labels include
   blocked paths and pawn attacks. The runtime table is fitted data, not a hand-authored
   movement table. Separate tests compare it with the teacher on cluttered boards.
2. **Move proposals:** trees conditioned on each categorical source token and its learned
   attack relation learn which actions are locally admissible from
   public piece tokens, coordinates, occupancy, turn, permissions, en passant state,
   and the learned attack relation. Teacher-generated random games, sparse positions,
   diverse special-rule families, and systematic pawn captures across ranks/files and
   captured piece types cover both colors. Teacher feedback separates
   local move feasibility from the final self-check decision.
3. **Piece effects:** trees learn the moved token and additional token edits. Rejected
   actions predict an unchanged board. The generic
   operation transfers a token from source to destination. Learned edits account for
   promotion, rook relocation, and en passant removal. Restricting edit inputs to relevant
   token/displacement/event features reduces correlations with unrelated pieces.
4. **Safety:** a final tree learns from teacher corrections which post-action attack
   conditions make a move illegal. Generic features describe attacks on every token type;
   no `if king attacked: reject` rule is authored in the runtime.
5. **Clearance constraints:** generic spatial features include occupancy between action
   endpoints and along the direction to the board boundary. Where successful teacher
   examples in a category consistently share a clearance or ownership value, training
   induces that constant constraint. Neither the required value nor a chess-piece rule
   is authored in the runtime. This removes rare shortcuts such as jumping over a
   blocker or ignoring clutter along a castling path.
6. **Corrections:** training enumerates all 20,160 syntactic actions on each training
   board, collects wrong predictions, and refits. This continues until a pass finds no
   training errors or the configured correction budget is exhausted. It does not mine
   evaluation boards.

The inference module, `pai/chess/rules.py`, depends on NumPy, public observations, and
the saved learned trees/table. The chess library and scikit-learn appear only in teaching,
training, and answer-key evaluation. Checkpoints are JSON, not executable pickle files.
Tests block teacher and trainer imports during checkpoint loading and inference.

This approach uses a stronger spatial representation than the initial MLP. Geometry,
token ownership, and generic token transfer are supplied as representational priors;
chess attack labels, movement decisions, special effects, and safety decisions are learned
from supervision. It does not claim that all abstractions emerged from pixels unaided.

## Project integration

`LearnedChessRules.command(StateBelief)` returns an existing `move_piece` skill command.
`ChessTeacher.execute()` handles the public action and records its outcome and correction
in `CognitiveEpisode`. Episodes remain compatible with `CognitiveMemory`. A position with
no model-approved action produces no command, rather than selecting an arbitrary move.
The teacher validates execution as an environment; it never supplies an inference mask.

## Reproduce

```powershell
.venv\Scripts\python.exe -m pip install -e ".[chess]"
.venv\Scripts\python.exe scripts/chess_rules.py --games 60 --plies 80 --sparse 400 --special 800 --corrections 12 --eval-seed 108 --eval-games 12 --eval-sparse 100 --eval-special 120
.venv\Scripts\python.exe scripts/chess_rules.py --evaluate-only --eval-seed 109 --eval-games 12 --eval-sparse 100 --eval-special 120 --output results/chess_rules_seed109.json
.venv\Scripts\python.exe -m pytest tests/test_chess_rules.py tests/test_chess_stage5.py tests/test_cognition.py tests/test_cognitive_extensions.py -q
```

Model: `runs/chess_rules/rules.json`. The original neural checkpoint remains separate.
Reports: `results/chess_rules.json`, with fresh evaluation reports alongside it.
The development results for seeds 100–107 retain earlier model results; those seeds
informed feature changes and are not the final independent gate.

## Verified results (2026-10-10)

Training: 3,987 valid positions, 106,496 isolated attack examples, complete-action
correction mining, and destination-occupancy constraints induced from the same original
positive training examples. The latter prevent numeric target-token shortcuts such as
mistaking a pawn's quiet jump for a capture. The inference checkpoint is saved after this
refinement; evaluation examples never enter its teaching or correction data.

| Measure | Fresh seed 108 | Fresh seed 109 |
|---|---:|---:|
| Evaluated positions | 566 | 562 |
| Complete legal-move sets matched | 566/566 | 562/562 |
| All syntactic actions scored | 11,410,560 | 11,329,920 |
| False positives / missed legal moves | 0 / 0 | 0 / 0 |
| Legal precision / recall | 100% / 100% | 100% / 100% |
| Unfiltered choices that are legal | 543/543 | 540/540 |
| Correct terminal abstentions | 23/23 | 22/22 |
| Exact legal piece-board transitions | 13,249/13,249 | 13,496/13,496 |
| Rule gate / strict suite | pass / pass | pass / pass |

Combined: 1,128 evaluated boards, 22,740,480 action labels, 1,083 legal choices,
45 terminal abstentions, and 26,745 exact legal piece transitions across the two suites.
These totals count evaluation instances across suites, not a claim that all positions
are statistically independent. Exact training-position overlap is zero in each suite.
All seven topic groups in each suite passed: games, sparse boards, castling, en passant,
promotion, checkmate, and stalemate.

37 tests passed across the rule learner, original neural baseline, and shared cognitive
contract/extension suites. They include illegal attempts preserving state, blocked pawn
jumps, initial-rank pawn captures, castling through attack, discovered check on en passant,
promotion effects, JSON replay, shared skill execution, and inference with teacher/trainer
imports blocked. The simulator-wide test suite was not rerun for this chess-only build.

## Reliability measures

Each held-out board is evaluated against **every** syntactic action, including empty
source squares, wrong-color moves, and unsupported promotion requests. All exact public
training positions are excluded and reported; separate generator seeds produce fresh
games, clutter, castling, promotion, en passant, checkmate, and stalemate cases.

The report contains false positives and false negatives, legal precision/recall,
exact equality of complete legal-move sets, unfiltered move choices, correct terminal
abstentions, and exact piece-board effects for every legal move. Accuracy alone is not
used as evidence of reliability because most syntactic moves are illegal. The rule gate
requires legal precision, legal recall, and illegal recall each above 99%. The stricter
suite flag additionally requires zero move errors and zero piece-effect errors.

Finite independent suites provide measured reliability, not proof over every possible
chess position. Related random-game positions are not statistically independent, and
special-rule families remain synthetic. This model supports valid standard chess boards
with public state, not Chess960 or variants. The rule model itself predicts piece
occupancy; the separate [complete state model](CHESS_STATE.md) now adds learned
turn, castling/en passant updates, counters, and public history. A separate
[learned evaluator](CHESS_EVALUATION.md) now predicts endings/draw claims and
provides an engine-distilled position-value baseline. [Bounded search and draw-claim
execution](CHESS_SEARCH.md) are also implemented. Self-play, HRM/MCTS integration,
and measured Elo remain Stage 5 work.

Implementation reference: [scikit-learn decision-tree learning](https://scikit-learn.org/1.5/modules/tree.html).
