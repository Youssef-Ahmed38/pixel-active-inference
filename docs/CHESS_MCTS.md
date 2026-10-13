# Learned-state MCTS baseline

`pai/chess/mcts.py` adds deterministic UCT search over the same learned legality,
state transitions, endings and scalar value as alpha-beta. Each path retains its
own counters and repetition history. There is no transposition merge and no
teacher or `python-chess` dependency during inference. It is not an HRM planner.

Leaf values use a bounded transform of the learned centipawn score. Values change
sign across player turns. Optional draw claims are zero-valued actions with valid
intended-move witnesses where needed; claims do not remove winning moves.
The most visited root action is selected, with mean value as a tie-breaker.

The node budget counts every inspected node, including revisits. Root model
preparation and generated child templates are additional work, so this is not a
hard timing bound. MCTS `completed_depth` stays zero because a sampled tree is
not a completed full-width search iteration. `deepest_ply`, simulations and
selected visits provide the actual sampling diagnostics. The reported value is
a search mean, not calibrated playing strength.

## Paired fresh comparison, seed 420

| Planner | Budget | Legal selections | Mates solved | Queen traps avoided |
|---|---:|---:|---:|---:|
| Alpha-beta | 32 | 8/8 | 4/4 | 0/4 |
| MCTS | 32 | 8/8 | 4/4 | 0/4 |
| Alpha-beta | 128 | 8/8 | 4/4 | 4/4 |
| MCTS | 128 | 8/8 | 4/4 | 1/4 |

Both passed root legality, variation/counter agreement, node-budget checks and
all three claim-execution cases. These are generated tactical families, not
independent match-strength evidence. **MCTS performed worse on these traps and
does not replace the alpha-beta incumbent.** Reports are `results/chess_mcts.json`
and `results/chess_search_seed420.json`.

Regression tests include single-node fallback, claims, automatic endings,
winning despite an available claim, node limits, immutability and teacher-free
runtime. Test fixtures train small models without needing saved `runs/` artifacts.

```powershell
.venv\Scripts\python.exe scripts/chess_search.py --planner mcts --eval-seed 420 --output results/chess_mcts.json
.venv\Scripts\python.exe -m pytest tests/test_chess_mcts.py -q
```

The [chess HRM and contextual value branches](CHESS_VALUE_UPDATES.md) now exist and
have fresh evaluations. Stronger search guidance and larger match samples remain.
The existence of MCTS does not satisfy the
roadmap's chess-strength or thinking-time rating gate.
