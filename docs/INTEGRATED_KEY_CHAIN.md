# Integrated maze, drawer, key, and door task

`pai.maze.key_chain` connects the existing maze explorer and discovery agent in
one episode. A generated maze contains a drawer landmark and a locked exit.
The agent searches without a map, reaches the drawer, opens it through the
discovery skill interface, carries the newly revealed object to the door,
tries it in visible fixtures, opens the door, and moves into the exit cell.
The same place memory is kept across the drawer-to-door goal switch.

The environment bridge uses maze geometry to place the fixtures and fixture
part IDs to restrict which skills are reachable at each site. These are scene
construction and action-availability rules; neither navigation nor discovery
receives the maze grid, the matching-key flag, or lock truth. Movement into the
exit cell stalls until the door's observed open feature is true.

## Measured gate

At a 500-move and 120-action-per-fixture budget, the integrated task passed
6/6 symbolic 15×15 mazes and 4/4 physical-door runs with independent seeds.
The physical runs use `hd_lever_pull_right`, a Robotiq gripper, both locks
engaged, and one key hidden in the drawer. Each run requires drawer opening,
carrying the key, opening the door, and entering the exit cell. Results are in
`results/key_chain_symbolic_0_5.json` and `results/key_chain_physical_0_3.json`.

```powershell
.venv\Scripts\python.exe scripts/key_chain_eval.py --seeds 6 --out results/key_chain_symbolic_0_5.json
.venv\Scripts\python.exe scripts/key_chain_eval.py --physics --seeds 4 --out results/key_chain_physical_0_3.json
.venv\Scripts\python.exe -m pytest -q tests/test_key_chain.py
```

This is skill composition across two simulators. Maze movement uses cell
observations and discrete steps, while the physical fixture is a stationary
MuJoCo door scene. The carried key stays in the fixture simulator's grasp
state while the maze skill moves.

## Stage 4 challenge

The next gate places two visually similar keys in the drawer and forces the
first one to be incompatible. The agent inserts and turns it both ways, then
uses the two stalled turns to diagnose an incompatible key. It withdraws the
object, returns to the drawer using the same place map, carries the second
candidate to the door, and completes the chain. Neither key identity nor
matching status is available to the agent. In the evaluation, simulator truth
is checked only after each episode to score the diagnosis.

The held-out evaluation uses three door types from the registry's test split,
four unseen 21×21 layouts per type, and a separate 15×15 training episode for
one-shot recipe learning. Cold and recipe-assisted runs share every held-out
seed and scene setting. Both passed 12/12; all 12 incompatible-key reports
matched the simulator answer key. The learned recipe reduced mean fixture
actions from 34.9 to 29.7, and was faster on 11/12 paired runs. These numbers
are in `results/key_chain_stage4_eval.json`.

The physical two-key version passed four seeds on the trained
`hd_lever_pull_right` type with a 30-action discovery budget and bounded
contact repair. Its result is in `results/key_chain_physical_two_keys_0_3.json`.
The physical runs do not force a wrong key first; the controlled wrong-key
diagnosis and held-out door-type transfer gate is symbolic.

An additional physical check on seed 20 of each of the three held-out door
types passed the lever-push, ball-knob push, and knob-pull variants. The
ball-knob push type also passed seeds 20–23 in
`results/key_chain_physical_ball_push_20_23.json`. The body-skill fix holds
the observed knob angle during the push and follows through when the fingers
slide on the round head; attempting to regrasp had let the door closer shut
the partly opened door. These runs support transfer on the measured types and
seeds, not every physical door or body.

```powershell
.venv\Scripts\python.exe scripts/key_chain_stage4_eval.py
.venv\Scripts\python.exe scripts/key_chain_eval.py --physics --keys 2 --seeds 4 --fixture-budget 30 --out results/key_chain_physical_two_keys_0_3.json
.venv\Scripts\python.exe -m pytest -q
```

The repository suite passes 257 tests with this build.

The integrated body remains a skill-level bridge across two simulators.
Camera-only maze navigation, a mobile manipulator, and explanation accuracy
across causes beyond incompatible keys remain separate research tasks.
