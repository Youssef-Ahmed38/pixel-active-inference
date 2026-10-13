# Stage 5: self-play and opponent measurement

`pai/chess/play.py` collects games through the shared chess teacher, records
search targets, fits a regularized value candidate, and plays Stockfish opponents.
The agent still uses its learned rules/state/endings/value and bounded alpha-beta
search. This is not HRM or MCTS.

## Measured results

The initial four self-play games generated 192 search-bootstrap targets. All
reached the 48-ply cap. No terminal reward or draw was invented for them.
The candidate fit improved replay error but degraded fresh Stockfish validation:
392.9 cp baseline MAE versus 397.5 cp candidate MAE on the same 246 positions.
The incumbent remained unchanged. See `results/chess_selfplay.json`.

Stockfish 19 matches used configured UCI Elo settings 1320, 1800 and 2400;
learner budgets 32 and 128 nodes; opponent budget 1000 nodes; both colors on a
paired declared opening. Three capped games were resumed from their saved PGNs,
preserving repetition history and move counters. **12/12 games completed, all
learner losses, with no execution failures.** See
`results/chess_matches_completed.json`; the original capped results are retained
in `results/chess_matches.json`.

A finite Elo point estimate is unavailable at zero score. The separately labeled
pseudocount estimate must not be presented as measured human Elo. These configured
strength settings are not calibrated for our node-limited matches. See the
[official Stockfish strength documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html).
One opening pair and correlated colors cannot establish a reliable rating curve.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts/chess_play.py --selfplay-only --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --output results/chess_selfplay.json
.venv\Scripts\python.exe scripts/chess_play.py --matches-only --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --output results/chess_matches.json
.venv\Scripts\python.exe scripts/chess_play.py --resume-report results/chess_matches.json --engine runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe --max-plies 256 --output results/chess_matches_completed.json
```

New reports hash all four model checkpoints and enforce identity on resume,
including the opponent engine name. Resume currently loads the incumbent value
checkpoint by default; resume an accepted candidate run using `--value-checkpoint`
with the saved candidate path. Legacy reports without hashes are labeled unverified on resume.
The original measured run predates this guard. The original opening is a declared root: history before
that root is excluded in both players. Resumption preserves all history after it.

## Remaining

The [chess HRM and contextual value candidates](CHESS_VALUE_UPDATES.md) are integrated
and evaluated separately. Stronger learned value/search, larger independent match
samples, an accepted self-play improvement, and a demonstrated thinking-time
strength curve remain research work. Optional physical chess is also unbuilt.
