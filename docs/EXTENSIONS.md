# Live causes and additional tasks

These additions extend the arm and maze alignment described in [ALIGNMENT.md](ALIGNMENT.md).
They use the existing domain agents and physical skills rather than replacing their planners.

## Live cause thinker

`pai.cognition.thinker` now has streaming arm and maze wrappers around the same HRM schema.
The arm wrapper converts each calibrated 12-channel residual into the trained nine-channel
evidence vector. When the existing surprise monitor triggers, a confident thinker cause can
corroborate an earlier correction; a high-confidence calibrated diagnosis is not vetoed by
a thinker false negative. The Bayesian model still estimates physical parameters such as the
camera offset. `scripts/online_arm_eval.py` runs paired controller episodes using the local
v7 calibration and a fixed camera shift. With paired Torch RNG seeds 123–125, both the
Bayesian baseline and the live-thinker path succeeded in 3/3 episodes with the same steps.
The thinker therefore has no demonstrated task-success gain in this small comparison.

The maze wrapper receives unexpected collisions and changes in the visual localizer's frame.
On a confident blocked-passage decision it invalidates that route; on a camera-shift decision
it records acceptance of the localizer's correction. The baseline already replans and
recalibrates, so any success difference must be measured rather than assumed.

## Fresh mazes

`scripts/fresh_maze_eval.py` uses seeds outside the thinker training set, 15×15 and 21×21
layouts, changed loop density, a per-episode RGB palette jitter, noisy odometry and varied
block and camera-shift times. It runs paired episodes with and without the thinker and writes
every outcome. At the original 700-step budget, both conditions reached the goal and
fully covered 6/6 mazes. Seed 40005 exposed a visual-localization limit: a shifted camera
in a repetitive corridor did not provide enough distinctive overlap to infer the offset.
The explorer now keeps an independent route graph from successful body moves and collision
outcomes. When visual frontiers run out, it probes physical frontiers from that graph and
returns along confirmed paths. This recovers full coverage within 700 steps on the six
fixed fresh layouts. A wider 20-seed run also reached the goal and full coverage in 20/20
layouts for both conditions. The thinker did not improve coverage or steps in these paired runs.

## Doors and drawers

`pai.cognition.discovery` adapts each offered door or drawer action to `SkillCommand`, each
observable feature to `StateBelief`, and each skill result to `SkillOutcome` and surprise
evidence. It runs the existing `DiscoveryAgent` through that adapter and stores the trajectory
in `CognitiveMemory`. The agent does not receive `World.truth()`.

`scripts/discovery_cognitive_eval.py` tests door and drawer goals separately. Twelve symbolic
test cases for each task passed; one physics-backed smoke case for each task passed with the
Robotiq gripper. Those are latch/unlocked cases, not a claim about every key, deadbolt or body.

## Multi-step arm goals

`CognitiveController.run_sequence` pursues ordered relations, rechecks earlier relations
after later actions and retries disturbed goals. `scripts/multistep_arm_eval.py` evaluates
`on(blue, plate)` followed by `on(red, blue)` with the existing learned world model and MPPI.
The script can also evaluate `on(blue, plate)` followed by `in(red, bowl)` with `--sequence tidy`.
Success requires *both* final predicates, not merely a successful report for each step.
With fixed Torch seeds 123–125, both the stack and tidy tasks succeeded in 3/3 episodes.
The previously failing stack began its second pickup with the gripper low over the first
placed block; a sideways transit moved that block off the plate. The policy now raises the
gripper vertically before travelling to a distant object. These are small fixed-seed checks,
not a general reliability guarantee.

Run the focused checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q --noconftest tests/test_cognitive_extensions.py tests/test_cognition.py tests/test_maze.py tests/test_maze_advanced.py
.venv\Scripts\python.exe scripts/fresh_maze_eval.py --seeds 6
.venv\Scripts\python.exe scripts/discovery_cognitive_eval.py --seeds 12
.venv\Scripts\python.exe scripts/multistep_arm_eval.py --seeds 123 124 125
.venv\Scripts\python.exe scripts/online_arm_eval.py --seeds 123
```
