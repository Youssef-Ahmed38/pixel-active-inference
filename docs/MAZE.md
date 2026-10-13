# Maze exploration slice

This is an early implementation of sprint week 6's **simple mobile-body** demo. It follows the
same hierarchy as the tabletop slice: a goal (cover the maze or find a marked place), a planner
that values information as well as progress, a body action, local observations, and an episode log.
It does **not** reuse the arm's learned dynamics weights, HRM, or MuJoCo locomotion controller.
Those models operate on tabletop entity tokens and end-effector actions, whereas this slice uses
discrete places and adjacent mobile moves. This distinction is important when reporting transfer.

| Level | Maze implementation | Relation to the arm agent |
|---|---|---|
| L2 | Coverage or goal-place preference | A goal above body control |
| L1 | Route to observed goal, otherwise maximize expected unseen area per travel step | Pragmatic plus epistemic action choice; current gain is a geometric proxy, not a learned ensemble |
| L0 | Cardinal move of a simple mobile body | Replaceable later by a walking skill |
| Perception | Range-limited local view, occluded by walls; odometry position | No full map or goal coordinates given to the policy |
| Memory | Observed cells, visits and place connections | An explicit place-memory analogue of episodic memory |
| Evaluation | Paired scenes versus a walker with identical movement and observations | Week-6 coverage gate |

The generator makes a connected corridor maze with optional loops and a hidden goal. The random
walker selects among visible passable neighbours. Both agents get the same maze seed, sensor and
step budget. Evaluation-only ground truth computes coverage as distinct reachable floor cells
visited, divided by all reachable floor cells. Goal discovery and first arrival are also recorded.

## Completion gate

With a 600-move cap, the explorer reached the goal **and visited every reachable floor cell in
30/30 mazes**. On a fresh set of 30 mazes starting at seed 1000 it again reached the goal and
visited every reachable floor cell in **30/30**. The explorer stops once its known reachable
places have all been visited; the cap is only a failure limit. Results:
[first set](../results/maze_complete.json), [fresh seeds](../results/maze_complete_holdout.json).

This is 100% on these 60 generated mazes under the stated sensor and movement assumptions, not
a guarantee for every possible maze. A 21×21 generated maze has roughly 200 reachable floor cells,
so the 160-move evaluation below cannot possibly achieve full visited-cell coverage.

## Fixed-budget comparison

Thirty independently generated 21×21 mazes, 160 moves per policy, local sensor range 3.
Results are in [results/maze_eval.json](../results/maze_eval.json).

| Policy | Mean coverage | Reached hidden goal | Mean first-goal step, failures capped at 160 |
|---|---:|---:|---:|
| Place-memory explorer | 65.2% | 22/30 | 95.6 |
| Random walker | 10.4% | 5/30 | 142.4 |

This passes the week-6 coverage-versus-random gate for the simple mobile body. It is an initial
single-configuration result. A separate 30-maze run starting at seed 1000 gave **64.1% versus
9.1%** coverage and 22/30 versus 4/30 goal arrivals
([holdout result](../results/maze_eval_holdout.json)). Other maze sizes and map designs remain
to be evaluated.

Run:

```bash
python scripts/maze_eval.py --episodes 30 --budget 600 --require-complete
python -m pytest -q --noconftest tests/test_maze.py
```

## Visual and mobile extension

The extension keeps the same frontier planner and changes its input and body interface:

- `pai.maze.visual` receives a local RGB image and noisy movement estimate. It decodes colored
  floor, walls, goal and landmarks; stores remembered views; matches new images against its map;
  and corrects camera offset when image-map agreement changes. The policy receives no absolute
  position from the camera. The movement skill's executed/stalled result is also used as a motion
  prior. Color decoding is hand specified; this is **template-based place recognition**, not a
  trained visual encoder or a learned SLAM system.
- `pai.maze.mobility` executes each requested adjacent move as a sampled, speed-limited continuous
  trajectory and reports a stall when it hits a wall. It is a kinematic mobile base, not a
  humanoid gait or dynamic MuJoCo locomotion.
- `pai.maze.advanced` injects a new passage block only when the remaining map stays connected and
  shifts the camera during exploration. The agent reports an unexpected blocked route or camera
  shift, marks the route closed or recalibrates, then replans. Environment events are the scoring
  truth; they are not passed to the agent.

The [advanced evaluation](../results/maze_advanced.json) spans 15×15 and 21×21 mazes, sensor
ranges 2 and 3, three seeds per setting, odometry noise 0.35 cells per move, and both disturbances.
All **12/12** runs reached the goal and visited every reachable cell under a 700-move cap. The
agent reported both injected causes in **12/12** runs, with no additional cause reports in those
runs. Its mean pose error was 0 cells, compared with 5.73 cells for raw integrated noisy odometry;
the executed/stalled movement result contributes to that correction, so this is not a pure vision
ablation.
The [demo GIF](media/maze/exploration.gif) shows (left to right) ground truth for the viewer, the
agent's remembered map, and its local camera; its [sidecar](media/maze/exploration.json) records
the injected events and the agent's cause reports.

## Transfer test

`scripts/maze_transfer.py` tunes the frontier information-gain weight on small 15×15 mazes, then
freezes it and tests on unseen 21×21 mazes. It compares the transferred setting with the original
fixed setting and a random walker. The [result](../results/maze_transfer.json) shows that the
learned setting **did not improve coverage** on larger mazes: 64.2% versus 65.5% at 160 moves.
It reached the goal in 25/30 versus 24/30, an inconclusive one-maze difference. This is a real
negative transfer result, not evidence of reusable knowledge across maze distributions.

The next substantive gap is a learned visual encoder and place recognition under appearance
changes, followed by a dynamic walking controller. The 100% completion gate above belongs to
the simpler discrete maze setup; it must not be assumed for those harder settings.

```bash
python scripts/maze_advanced_eval.py --seeds 3 --budget 700
python scripts/maze_transfer.py --train-episodes 12 --test-episodes 30
python scripts/maze_demo.py --seed 0
python -m pytest -q --noconftest tests/test_maze.py tests/test_maze_advanced.py
```
