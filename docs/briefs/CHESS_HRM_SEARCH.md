# Brief: the HRM thinker inside chess search, with learned halting (Stage 5)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (learned models, search, the HRM
"thinker" as a small recurrent reasoning module, learned rather than scripted, honest evaluation).
This task is Stage 5, chess.

## 1. Read first

| File | What it is |
|---|---|
| `docs/CHESS_STRENGTH.md` | The current player: learned rules, a learned value and move prior from a Stockfish teacher, search; 1037 conditional engine-anchor Elo [863, 1134] at 1 s/move over a 120-game ladder; the HRM was tried as a value (227 cp error, 4/4 draws at 0.1 s) and its halting head was not trained |
| `docs/CHESS_SEARCH.md`, `docs/CHESS_VALUE_UPDATES.md` | Search and the value branches |
| `pai/chess/search.py`, `mcts.py`, `hrm_value.py`, `hrm_training.py`, `ladder.py`, `pai/thinker/*` | Code |
| `docs/STATUS.md` (Stage 5 row) | Open item: "connect HRM to search with learned halting; more games per ladder rung" |

## 2. The task

1. **HRM in search with learned halting.** Use the HRM where it can help search, e.g. as a value or
   move prior at search nodes, and train its halting head so it thinks longer on hard positions and
   stops early on easy ones (halting learned from the data, e.g. from when extra recurrent steps
   stop changing the output or the teacher's error, not a fixed step count). Report the
   distribution of halting steps and whether harder positions get more steps.
2. **Measure strength honestly.** Compare the current player and the HRM-in-search player at equal
   time per move: a paired match on fresh openings, and the Stockfish ladder with **more games per
   rung** than before (the 120-game ladder had wide intervals), using the existing conditional
   engine-anchor Elo method and its intervals. Fresh opening seeds never used before (see the seeds
   listed in `docs/CHESS_STRENGTH.md`).
3. **No engine in the agent's move path.** Stockfish only as teacher (training labels) and as rated
   opponent, as before.

## 3. Rules

- Work in the main checkout on `main`. **Do not create branches, stash, reset or commit**; Claude
  reviews and commits.
- Touch only: `pai/chess/*`, `pai/thinker/*` (no change to behaviour used by other stages; add, do not
  break), new `scripts/chess_hrm_*.py`, `tests/test_chess*.py`, `tests/test_thinker*.py`, a new
  section in `docs/CHESS_STRENGTH.md`, new `results/chess_hrm_*` files. Results files larger than
  5 MB: gzip-compressed (`.json.gz`). **Never record absolute local paths** (user folders, caches) in
  any result file: use project-relative paths or a placeholder.
- **Do not touch** walking, maze, arm-cause, discovery or language files: other work is in progress.
  The laptop GPU is shared with humanoid training; keep batch sizes modest.
- Keep passing: `tests/test_chess*.py`, `tests/test_thinker*.py`, `tests/test_cognition.py`.

## 4. Deliverables

Code and fast unit tests; the new documentation section (design, learned vs given, halting
analysis, match and ladder tables with intervals, failures, seeds, exact commands);
`results/chess_hrm_*` with per-game records (PGN or move lists). Finish with a short report: files,
commands, seeds, numbers, what is not done.
