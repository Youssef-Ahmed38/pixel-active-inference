# Brief: make the chess agent stronger (Stage 5), measured honestly

You are helping build **Pixel Active Inference**, a research project (PhD application) at
<https://github.com/Youssef-Ahmed38/pixel-active-inference>. One agent learns several skills in
simulation: robot-arm manipulation, a maze, doors and keys, chess, language and riding a bike.
This task is **chess playing strength**. Work on the branch **`chess-strength`** (it starts from
`main` plus an unfinished first attempt, see "Starting point").

## 1. What exists (read these first)

| File | What it is |
|---|---|
| `docs/CHESS_STAGE5.md` | Stage overview: teacher environment, curriculum, learned rules |
| `docs/CHESS_RULES.md`, `docs/CHESS_STATE.md` | The agent *learned* the legal rules and state transitions from a teacher (it does not import a chess library at decision time). These gates pass and must keep passing. |
| `docs/CHESS_SEARCH.md`, `docs/CHESS_MCTS.md` | Alpha-beta (the incumbent) and MCTS search over the learned model, quiescence |
| `docs/CHESS_EVALUATION.md`, `docs/CHESS_VALUE_UPDATES.md` | Value functions: original (incumbent), HRM and contextual candidates (both rejected) |
| `docs/CHESS_PLAY.md` | Self-play updates and Stockfish matches: **0 wins, 0 draws, 12 losses**, no finite Elo |
| `docs/AUDIT_OUTSIDE_BUILDS.md` §6 and item 7 | An independent audit of the chess stage |
| `pai/chess/` | `rules.py`, `state.py`, `search.py`, `mcts.py`, `evaluation*.py`, `hrm_value.py`, `play.py`, `teacher.py`, ... |
| `pai/thinker/hrm.py` | The project's HRM ("Hierarchical Reasoning Model") thinker, shared across stages |
| `pai/cognition/contracts.py` | The shared observation / skill / outcome / episode vocabulary |

Why it is weak (from the reports): the learner searched only **32 or 128 nodes** per move
against Stockfish at **1000 nodes**, with a weak hand-fitted value. Its learned rule model is
slow, so very few nodes fit in a move.

## 2. Starting point

The branch `chess-strength` adds an unfinished, unreviewed first attempt:
`pai/chess/distill.py` (teacher distillation), `pai/chess/ladder.py` (rating ladder),
`scripts/chess_distill.py`, `scripts/chess_ladder.py`, `tests/test_chess_strength.py`, and a
small change to `pai/chess/rules.py`. Read them, keep what is sound, and rewrite the rest.

## 3. The task

Make the agent play clearly stronger, and measure its strength with a real, finite rating.

1. **A stronger learned value (and policy prior).** Use Stockfish as a *teacher during training
   only*: label positions (from self-play and teacher games) at shallow depth and distil them into
   the agent's value model (and optionally a move-ordering/policy prior). Try the HRM branch
   (`pai/chess/hrm_value.py`, `pai/thinker/hrm.py`) as one candidate, so the shared thinker is used
   in chess. Choose features and model size that fit a laptop (CPU + 8 GB GPU).
2. **Search that uses its time.** Make search fast enough to use more nodes per move: cache the
   learned rule model, batch evaluations, order moves with the learned prior, iterative deepening
   under a **time** budget, not only a node budget.
3. **Accept only proven improvements.** A candidate replaces the incumbent only after it wins a
   paired comparison on **fresh** positions and games (same openings, both colours), with a
   statistical test. Keep the rejected candidates and their numbers.
4. **A real rating.** Build a ladder against Stockfish at several strengths (`UCI_LimitStrength` +
   `UCI_Elo` from 1320 upward, and/or limited depth/nodes), with **fixed time controls for both
   sides**, many games per level (≥ 20, both colours, varied openings from a fixed book), and
   report a finite Elo estimate with a 95% interval. Also report the thinking-time curve
   (strength vs time per move).

## 4. Rules you must follow

- **No engine at play time.** Stockfish is a teacher for training labels and the opponent in
  evaluation. The agent's own moves must come only from its learned models and its search: no
  Stockfish call, no `python-chess` legality or evaluation inside the agent's decision.
- **The learned-rule gates stay frozen and passing:** `tests/test_chess_rules.py`,
  `tests/test_chess_stage5.py` and the rest of `tests/test_chess_*.py`.
- **Fresh evaluation.** Develop on development seeds and openings; the final numbers come from
  seeds/openings never used during development. State them in a seed list in the doc.
- **Honest reporting.** Report only what you ran, with exact commands. Keep failures and rejected
  candidates. Never present a pseudo-count or extrapolated number as a measured Elo.
- **Every result file records** the git commit hash, the seeds, and the time controls.
- **Stay in your files:** `pai/chess/*`, `scripts/chess_*.py`, `tests/test_chess_*.py`,
  `docs/CHESS_*.md`, new `results/chess_*` files (keep the old ones). Do not change other stages.

## 5. Environment

- Windows 11, Python 3.12, `python-chess` 1.11.2, PyTorch, a CUDA GPU (8 GB). Commands in the
  docs use `.venv\Scripts\python.exe`.
- **Stockfish 19** is not in the repository: download the Windows build from
  <https://github.com/official-stockfish/Stockfish/releases/tag/sf_19> and pass its path with
  `--engine` (locally it lives at `runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe`).
- **Checkpoints** under `runs/chess_rules/` are not in the repository (`runs/` is git-ignored).
  Rebuild them with the commands in `docs/CHESS_RULES.md` and `docs/CHESS_STATE.md` ("Reproduce"),
  or ask the user to upload them.
- Commits: the author is the repository owner (configured in the repo); do **not** add AI
  co-author lines. Do not commit `runs/`, engine binaries or large data.

## 6. Deliverables

1. Code and fast unit tests (value model I/O, distillation data, search with a time budget, the
   ladder's Elo computation on synthetic results, and that the agent's move path never calls the
   engine).
2. `docs/CHESS_STRENGTH.md`: what changed, what is learned vs given, the before/after table
   (incumbent vs new), the ladder (games, score and Elo with 95% interval per level, overall Elo),
   the thinking-time curve, rejected candidates, seeds, and reproduce commands.
3. `results/chess_strength_*.json` with the raw games (PGN or move lists) and metadata.
4. A pull request from `chess-strength` to `main` with a short summary.

## 7. How the result will be checked

The project owner's other assistant will verify:

- that the agent's move selection never calls Stockfish or `python-chess`;
- that the final games used fresh openings/seeds, with the stated time controls for both sides;
- the Elo computation, by recomputing it from the raw games;
- that all `tests/test_chess_*.py` pass, and that the rule gates did not change.

A smaller honest gain is worth more than a large unverified one.
