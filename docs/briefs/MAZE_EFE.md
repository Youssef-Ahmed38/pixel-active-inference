# Brief: make the maze explorer a real active-inference agent (Stage 3)

You are helping build **Pixel Active Inference**, a research project (PhD application) at
<https://github.com/Youssef-Ahmed38/pixel-active-inference>. One agent learns several skills in
simulation (arm, maze, doors and keys, chess, language, bike) with the same principles:
**active inference** (a generative model of the world, beliefs updated from observations, and
actions chosen by **expected free energy** = pragmatic value + epistemic value / information gain),
surprise, hidden-cause inference, memory, learned from experience rather than scripted.
This task is the **maze explorer**. Work on the branch **`maze-efe`** (cut from `main`).

## 1. What exists (read these first)

| File | What it is |
|---|---|
| `docs/MAZE.md`, `docs/EXTENSIONS.md`, `docs/NEXT_BUILDS.md` | The maze stage as built so far |
| `docs/AUDIT_OUTSIDE_BUILDS.md` §2 (items 2, 3, 5) and §4 | **An independent audit of this code. Read it fully; it is the reason for this task.** |
| `docs/ROADMAP.md`, `docs/BUILD_PLAN_STAGES_2_7.md` (Stage 3) | The intended design and the Stage 3 gates |
| `pai/maze/env.py` | Procedural grid mazes, landmarks, local egocentric view, odometry |
| `pai/maze/agent.py` | The explorer: place memory (`agent.py:19-64`) and frontier choice (`agent.py:95-117`) |
| `pai/maze/advanced.py`, `pai/maze/visual.py`, `pai/maze/learned_vision.py` | Camera patches, noisy odometry, the cause monitor, visual probes |
| `pai/maze/key_chain.py` | Uses the explorer (maze → drawer → key → door); keep its interface working |
| `pai/planning/objective.py` | The shared `expected_free_energy(pragmatic, epistemic, ...)` used by the arm |
| `pai/cognition/` | The shared contracts, cause posterior and memory |
| `scripts/maze_eval.py`, `fresh_maze_eval.py`, `maze_transfer.py`, `maze_advanced_eval.py` | Evaluations |

**What the audit found (the problems to fix):**

- The explorer calls `expected_free_energy(0.0, information_rate)` with **no pragmatic term**, and
  the information term is a geometric count whose weight was tuned to **0.0**
  (`results/maze_transfer.json`). So it is a **nearest-frontier rule**, which is exactly the
  baseline the Stage 3 gate says it must beat, and **no frontier baseline was ever run**.
- The maze cause monitor uses **constant log-evidence** (`advanced.py:23-41`: none −8,
  passage_blocked 0, unknown −4), so the posterior never depends on data; the visual probe
  likelihoods are also hand-written (`visual.py:139-145`).
- **Transfer failed:** after other mazes, coverage 64.2% vs 65.5% without; goals 25/30 vs 24/30.
- **Seeds 40000–40005 were used for debugging**, then reused as "fresh" (only 40006–40019 are clean).

## 2. The task

1. **A real generative model of the maze.** A probabilistic map (e.g. occupancy probabilities per
   cell) updated by Bayes from the local observations, with an **observation model learned from
   experience** (how often a seen wall/floor/landmark is really there, under the camera noise and
   odometry drift that `advanced.py`/`visual.py` simulate). The map has uncertainty the agent can
   reason about.
2. **Action selection by expected free energy.** For candidate routes or next moves, compute the
   **expected information gain** about the map (expected entropy reduction under the learned
   observation model) plus the **pragmatic value** (preference for the goal once it is believed
   seen, and a cost for travel and risk), with `pai/planning/objective.py`. No term may be zero by
   construction; report the weights and how they were chosen (on development mazes only).
3. **Learned cause likelihoods.** Replace the constant log-evidence and the authored probe
   likelihoods with likelihoods **fitted from experience** (e.g. counts of residuals/collisions under
   each cause in training mazes). Add at least one confusable cause (e.g. odometry slip vs a blocked
   passage, or a moved landmark vs a camera shift) so cause inference is not trivial, and score the
   posterior against the simulator's truth.
4. **Baselines and transfer, honestly.** On the same fresh mazes: random walker, **nearest
   frontier**, and **information-gain frontier** (the classical Yamauchi-style and greedy
   entropy-reduction explorers), and the new EFE agent, with and without the learned prior from
   earlier mazes (transfer). Report coverage-vs-steps curves, time to find the goal, and success
   rates with 95% intervals.

## 3. Rules you must follow

- **No ground truth in the agent.** The policy must not read the maze grid, `truth()`, the goal
  position or simulator internals; only the local observations and its own odometry. Evaluation
  code may use truth for scoring.
- **Fresh seeds.** Do not use seeds below 50000 for final numbers (40000–40019 were seen during
  earlier development). Develop on your own development seeds and list them in the doc; final
  numbers come from a separate, never-used range.
- **Honest reporting.** Report only what you ran, with exact commands. If the EFE agent does not
  beat the frontier baselines, report that; a clear negative result is acceptable, a hidden one is not.
- **Keep the existing tests passing:** `tests/test_maze.py`, `tests/test_maze_advanced.py`,
  `tests/test_key_chain.py`, `tests/test_next_builds.py`, `tests/test_cognition.py`. If you must
  change the explorer's interface, update `key_chain.py` accordingly and keep its tests passing.
- **Every result file records** the git commit hash, the seeds and the parameters.
- **Stay in your files:** `pai/maze/*`, new `scripts/maze_*.py`, `tests/test_maze*.py`,
  `docs/MAZE.md`, new `results/maze_v2_*` files (keep the old ones). `pai/planning/objective.py` and
  `pai/cognition/*` only if needed and backward compatible. Do not touch other stages.
- The walking body is being rebuilt separately (Stage 2). Keep using the existing abstract moves
  (`pai/maze/mobility.py`); do not cite `pai/maze/walking.py` as locomotion.

## 4. Environment

Windows 11, Python 3.12, NumPy, PyTorch (CPU is enough here), MuJoCo 3.14 (not needed for the grid
maze). Run commands from the repository root with `.venv\Scripts\python.exe`. Commits: the author is
the repository owner (configured in the repo); do **not** add AI co-author lines; do not commit
`runs/` or large data.

## 5. Deliverables

1. Code: the probabilistic map and learned observation model, the EFE explorer, learned cause
   likelihoods, the baselines; fast unit tests (Bayes map update, expected information gain against
   a hand-computed small case, the EFE score uses both terms, the cause likelihoods are fitted from
   data, no truth access in the policy).
2. `docs/MAZE.md` updated: design, what is learned vs given, the results tables and curves,
   ablations (epistemic term off, learned prior off), failures, seed lists, reproduce commands.
3. `results/maze_v2_*.json` with per-episode records and metadata, and one GIF in `docs/media/maze/`
   showing the belief map and the chosen target (< 3 MB).
4. A pull request from `maze-efe` to `main` with a short summary.

## 6. How the result will be checked

The project owner's other assistant will verify:

- that the policy never reads the grid or truth;
- that the information gain is computed from the learned model;
- that the baselines ran on the same fresh mazes;
- the numbers, by recomputing them from the result files;
- that the listed tests pass.

A smaller honest gain over the frontier baseline is worth more than a large unverified one.
