# Next builds: causal probes, longer goals, pixels, walking and locks

The subsequent contact-aware repair build and its tower ablation are documented in
[CONTACT_REPAIR.md](CONTACT_REPAIR.md).

These are extensions of the existing shared arm, maze and discovery contracts. They use local
simulation and do not require the unfinished Job A or Job C data runs.

## Causal probes and repair

`pai.cognition.experiments` chooses a binary probe from predicted outcomes under competing
causes and computes its posterior after the observed result. The visual maze explorer now
retains a second route graph from successful body moves. If its camera-based map has no
frontier, it probes reachable passages and records whether a predicted wall was actually
passable. This route is selected by distance, visual evidence and a small expected-information
gain term. The agent does not read the maze grid or truth from the planner.

In `results/causal_probe_eval.json`, the 20 fresh layouts at 700 steps reached the goal and
full coverage in 19/20 without physical probes and 20/20 with them. This is evidence for the
physical probe mechanism on this distribution, not evidence that the neural thinker improves
task success. The paired live-thinker evaluation still shows equal outcomes with and without
the thinker.

`CognitiveController.run_sequence` accepts a repair factory. The arm's repair factory clears
a block sitting on an object that needs to be moved, then retries the original relation. A
toy end-to-end test verifies that an obstructed relation fails without clearing and succeeds
after the intervention.

## Longer arm tasks

`scripts/multistep_arm_eval.py` accepts repeated `--goal` relations and named three-goal
sequences. The `stack_tidy` sequence leaves blue on the plate, red on blue and yellow in the
bowl. It passed 3/3 fixed seeds (123–125). A custom four-goal task also passed 3/3: blue on
the plate, red on blue, and yellow and green in the bowl. The three-block `tower` remains a
harder measured case: seed 123 failed because later placements disturbed earlier contacts,
despite repair passes. These results do not establish
reliability across new placements or arbitrary sequence lengths.

## Learned maze vision

`pai.maze.learned_vision.TileColorModel` fits class-conditional RGB distributions from local
simulator frames and mask labels. The model is saved to `runs/maze_tile_vision.npz` and can be
loaded into `VisualLocalizer`; the agent receives only the RGB frame at inference. On ten
unseen palette seeds with colour bias and jitter, learned tile accuracy was 99.78% versus
99.61% for the fixed palette; maze goal and full coverage were 10/10. The resulting camera
map's floor/wall agreement with simulator truth was 99.87% over mapped cells. This is a learned tile
classifier, not a learned object-slot model or full visual place encoder.

## Walking body

`pai.maze.walking.WalkingMobile` alternates support feet and samples a centre-of-mass path
behind the existing adjacent-move skill. It can fall on an injected perturbation and recover.
The visual explorer treats a fall as a temporary body stall rather than a blocked passage.
`results/walking_maze_eval.json` records 10/10 goal and full-coverage episodes and 10/10
recoveries. The same ten seeds also passed with the learned tile classifier and shifted
palettes (`results/walking_learned_vision_eval.json`). This is a planar gait simulator; the roadmap's MuJoCo humanoid and learned gait
remain unbuilt.

## Physical locks

`scripts/discovery_cognitive_eval.py` now selects a physical door type, lock state and key
location, recording all primitive actions and a post-episode truth summary. Symbolic
deadbolt and key cases passed 3/3 each. A physical deadbolt passed on the Robotiq gripper.
A physical table-key case passed on `hd_lever_pull_right`. The same key task failed on
`hd_knob_pull_left` at 400 actions; the existing discovery benchmark already recorded this
body and fixture combination as difficult. With the key hidden in a drawer, the physical
`hd_lever_pull_right` case passed 1/2 seeds. In the failed seed the key bolt remained locked
after 300 actions and 130 stalled skills; extending that episode to 400 actions did not
unlock it. The current adapter records the result; reliable
physical picking and insertion remain open work.

Run the new checks:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_next_builds.py
.venv\Scripts\python.exe scripts/causal_probe_eval.py
.venv\Scripts\python.exe scripts/learned_maze_vision_eval.py
.venv\Scripts\python.exe scripts/walking_maze_eval.py
.venv\Scripts\python.exe scripts/multistep_arm_eval.py --sequence stack_tidy
.venv\Scripts\python.exe scripts/discovery_cognitive_eval.py --physics --seeds 1 --type-name hd_lever_pull_right --lock-state key --key-place table --tasks door
```
