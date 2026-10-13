# Stage 5: learning chess from a teacher

The next roadmap stage has started. This build implements the teaching environment,
curriculum, neural legality and piece-transition baseline, and reproducible evaluation.
It does not require the arm's Job A or Job C checkpoints.

The neural model below is the original baseline. A subsequent
[teacher-induced relational rule learner](CHESS_RULES.md) adds complete-action correction
training and stronger independent gates. Use `scripts/chess_rules.py` for the current
rule-learning build.

## Architecture and alignment

- `pai/chess/teacher.py`: the environment owns `python-chess` rules. It demonstrates
  legal moves, rejects illegal attempts without changing the board, and gives corrections.
  Its `observe`, `available_skills`, and `execute` methods implement the existing domain
  adapter contract. Attempts, observations, outcomes, and feedback enter `CognitiveEpisode`
  and can be stored in the same `CognitiveMemory` used by arm and maze tasks.
- `pai/chess/representation.py`: public square tokens, side to move, castling rights,
  en passant state, and syntactic move commands. Neither this module nor the learner
  imports the chess rule library. Candidate generation includes illegal actions.
- `pai/chess/learner.py`: an action-conditioned neural board model learns legality and
  subsequent piece tokens from demonstrations and rejected actions. A generic token
  persistence prior encourages unchanged squares to remain unchanged. It contains no
  piece-movement rules. Move selection ranks the complete action space without a teacher
  legality mask or fallback. Confidence is a model score, not a calibrated probability.
- `scripts/chess_stage5.py`: training, checkpoint saving/loading, held-out scoring,
  and evaluation of unfiltered move choices.

This uses the project's observation → action → prediction → outcome structure and shared
episode vocabulary. Chess has its own board model because its state/action dimensions
differ from a robot. The existing HRM thinker is not yet connected to chess search.
Input is a symbolic public board, not camera perception. The effect head predicts piece
occupancy only; predicting next castling rights, en passant state, terminal outcomes,
and game history remains necessary before learned multistep search.

## Teacher coverage

Lessons cover each piece, captures, check, a mate-in-one setup, checkmate, stalemate,
castling, promotion, and en passant. Random legal games add diverse positions. Every
position supplies all legal demonstrations and sampled illegal actions from occupied
and arbitrary squares. Corrections are teacher feedback, never an inference-time mask.

Public board states are deduplicated. A stable hash assigns each whole position to
training or evaluation; all moves from one position stay together. Loading a checkpoint
also excludes its training-position hashes from fresh evaluation. This prevents exact
position leakage; it does not establish generalization to every chess position or
independence between related positions from one game. Special lesson topics are listed
in the report only when their position falls in the held-out split.

## Run

```powershell
.venv\Scripts\python.exe -m pip install -e ".[chess]"
.venv\Scripts\python.exe scripts/chess_stage5.py --games 40 --plies 48 --epochs 25
.venv\Scripts\python.exe scripts/chess_stage5.py --evaluate-only --seed 1 --games 12 --plies 48 --output results/chess_stage5_seed1.json
.venv\Scripts\python.exe -m pytest tests/test_chess_stage5.py tests/test_cognition.py tests/test_cognitive_extensions.py -q
```

The checkpoint is `runs/chess_stage5/model.pt`. Reports are
`results/chess_stage5.json` and `results/chess_stage5_seed1.json`.
Seeds, corpus size, training losses, zero exact-position overlap, confusion counts,
per-topic scores, changed-square accuracy, and exact board-effect accuracy are recorded.
The unmasked policy score measures a small sample of complete action-space choices;
it is separate from binary classification of sampled actions.

The rule gate requires accuracy, legal recall, illegal recall, and legal precision all
above 99%, with both classes present. Rejecting every move cannot pass. A passing sampled
classification score would still not establish strong chess play.

## Original neural baseline and remaining Stage 5 work

### Measured starting baseline (2026-10-10)

Training: seed 0, 40 random games, up to 48 plies, 25 epochs, 1,541 training positions.

| Measure | Seed 0 held-out | Fresh seed 1 held-out |
|---|---:|---:|
| Positions | 326 | 108 |
| Labeled actions | 19,918 | 6,753 |
| Legality accuracy | 95.90% | 94.54% |
| Legal recall | 97.77% | 95.87% |
| Legal precision | 94.23% | 93.35% |
| Exact next piece-board accuracy on legal actions | 44.94% | 4.84% |
| Changed-square accuracy | 99.11% | 94.99% |
| Unfiltered choices that are legal | 3/12 | 3/12 |
| Rule gate | unmet | unmet |

The high changed-square score does not imply accurate complete transitions: unintended
changes elsewhere make an entire predicted board wrong. The low unfiltered policy score
also exposes false positives when ranking thousands of actions. These results establish
a runnable starting baseline, not a chess-playing milestone. Tests: 23 passed across
the new chess suite and existing cognitive contract/extension suites. The full simulator
suite was not rerun for this build.

### Next implementation steps

1. The original MLP needs stronger rule learning. The subsequent relational learner
   has passed the roadmap rule gate and both final strict held-out suites; see
   [its measured results](CHESS_RULES.md). Maintain and extend independent rare-rule suites.
2. Complete state-transition prediction, including special state and terminal events.
3. Learned value, bounded search through the learned model, teacher-guided practice,
   then self-play. Measure whether additional thinking improves results.
4. Stockfish matches at several levels, with recorded wins/draws/losses and rating
   uncertainty. No Elo rating is claimed by this build.
5. Optional MuJoCo chess-piece manipulation after the software game gates pass.

Teacher API reference: [python-chess core documentation](https://python-chess.readthedocs.io/en/stable/core.html).
