# Stage 7: bicycle simulation and observation-learning precursor

`pai/bike/env.py` builds a self-contained MuJoCo bicycle with a floating frame,
two spinning wheels, steering, and a leaning torso rider. Rear-wheel torque,
steering and torso servos are the only actuators; no external frame force holds
the bike upright. Internal collision masks prevent connected bike parts from
colliding with each other; all parts can contact the ground. The simplified rider
uses the roadmap fallback rather than a humanoid with hands and feet.

Initial speed is 4 m/s with randomized initial roll in ±0.03 radians. Success is
10 metres of forward displacement without falling; fall means roll beyond 0.7
radians or frame height below 0.4 metres. The task therefore measures maintaining
balance from a moving start, not mounting, starting from rest or obstacle riding.

## Current evidence

On fresh seeds 10–19, the classical balance controller passed **10/10** rides,
and the uncontrolled comparison passed **0/10**. See
`results/bike_stage7_holdout.json`.

`observation_learning.py` records expert motion at the physics timestep, infers
acceleration from observed velocity differences, and runs known-model inverse
dynamics to recover rear torque and steering/lean servo targets. A linear student
learns only the recovered targets. Saved expert controls audit inference and are
not used in fitting; a regression test alters them and checks unchanged weights.
The implementation follows [MuJoCo inverse-dynamics semantics](https://mujoco.readthedocs.io/en/stable/computation.html).

Ten teaching episodes yielded action-inference MAEs of 0.00172 Nm drive torque,
0.00309 radians steering target, and 0.00152 radians lean target. On fresh seeds
20–29, the observed-motion student and classical expert each passed **10/10**;
the uncontrolled comparison passed **1/10**. See `results/bike_observation.json`.
These small, easy moving-start trials are a baseline, not a general cycling proof.

The optional recorder renders actual simulation video; a verified sample is
`docs/media/bike/expert.mp4`. Learning currently uses
motion states, not video pixels. The dynamics and actuator model are supplied,
not learned from observation. It does not demonstrate locomotion transfer or
locomotion transfer.

## Diverse starts and practice

Increasing initial tilt to ±0.1 radians and varying speed from 2.5–4.5 m/s exposed
weaknesses: the original student passed only 6/10 development rides. The revised
teaching records broader starts and fits only control-decision boundaries;
intermediate states are consequences of the chosen action. On seeds 40–59,
student and expert each passed **17/20**, uncontrolled **0/20**. This is retained
in `results/bike_observation_diverse.json`, including the failed episodes.

`practice.py` mutates policy weights, evaluates every candidate on the same
practice layouts, and accepts only improved practice scores. It stores candidates
separately from the imitation checkpoint. With ten practice seeds 100–109 and
24 updates, the observed-motion initialization first passed all ten practice
layouts at attempt 16. A paired scratch initialization never did. Attempt zero
is the starting policy; each later attempt evaluates one candidate on all ten
layouts. This includes rejected candidates in the training cost.

On fresh seeds 120–129, the observed-motion policy improved from **7/10 to 10/10**
after practice. Scratch went from **1/10 to 0/10**: its accepted practice candidate
did not generalize. See `results/bike_practice_diverse.json`. This is one optimizer
seed and a small holdout, not a statistical proof of sample-efficiency transfer.
The incumbent imitation models remain available; practiced models are separate.

## Replicated practice

`bike_replicates.py` repeats the paired practice comparison using optimizer seeds
1000, 1001 and 1002, saving every policy in its own directory. All runs use the
same ten practice layouts and the same twenty fresh evaluation layouts, seeds
140–159. The shared observed-motion initialization passes 12/20 before practice.

| Optimizer seed | Observed policy after practice | Scratch after practice | First all-successful observed practice attempt |
|---|---:|---:|---:|
| 1000 | 20/20 | 2/20 | 16 |
| 1001 | 20/20 | 1/20 | 3 |
| 1002 | 19/20 | 0/20 | 15 |

Scratch never passed all practice layouts within 24 updates. These are three
optimizer runs on twenty shared layouts, not sixty independent physical layouts.
The observation benefit repeats in this experiment, while the one failed ride
and absent Stage 2 comparison keep the complete transfer gate unfinished.
See `results/bike_replicates.json` and the linked per-run reports.

## RGB orientation and hybrid balance

The first image branch pools RGB pixels and learns three frame orientation
angles. Runtime prediction reads an image only; MuJoCo pose labels are confined
to teaching and evaluation. A 600-frame static-pose model passed a separate
200-frame pose check with mean errors **0.0162, 0.0062, 0.0121 radians** for
roll, pitch and yaw. However, it failed on moving rides: errors rose to
**0.0626, 0.0514, 0.2675 radians**, worse than a zero-angle predictor. Both reports
are retained; the static result does not demonstrate video generalization.

A separate model trains on ten full moving rides, selects ridge regularization
on four other rides, and evaluates five fresh rides. Episode-averaged errors
were **0.0332, 0.000315, 0.1934 radians**, better than zero-angle means of
**0.0717, 0.000797, 0.4282 radians**. One evaluation ride timed out while turning;
its larger errors remain included. Frames within a ride are correlated, so
each ride has equal weight. Checkpoints remain separate under
`runs/bike/vision` and `runs/bike/vision_motion`.

`ImageBalanceController` feeds image-derived roll/yaw and a smoothed temporal
roll rate into the engineering balance controller. Forward speed remains a
sensor input, and camera tracking uses the supplied root position. On five
fresh paired starts, both hybrid visual balance and sensor baseline passed
**5/5**. This is a hybrid observation interface, not full pixel-based riding,
video-only imitation, full pose extraction or Stage 2 transfer.

The subsequent twenty-start check passed only **12/20** for hybrid vision and
**13/20** for the original sensor controller. The small smoke result was not a
complete generalization gate. Full failures are retained in
`results/bike_vision_control_holdout20.json`.

The first saved practiced policy (optimizer seed 1000) was also checked on twenty
new starts, seeds 890–909, without retuning it. With sensor inputs it passed
**20/20**. Replacing its attitude/rate inputs with this image model passed
**0/20**, despite retaining speed and steering/lean encoders. The motion image
model was taught on the engineering controller's trajectories; the practiced
policy has different joint motion and needs a separately validated visual
interface. This experiment is rejected for use with that practiced policy.
Report: `results/bike_vision_practiced_holdout20.json`. The report records both
model hashes. None of these results changes the incumbent sensor policy.

The next separate image candidate is taught from the practiced policy's own
motion, using ten training rides (910–919), four validation rides (920–923) and
five fresh pose-evaluation rides (930–934). Episode-averaged errors improved to
**0.0220, 0.00129, 0.0276 radians**, against zero-angle means of
**0.1197, 0.00202, 0.4976 radians**. The model records its teacher policy hash and
remains separate in `runs/bike/vision_practiced/orientation.json`. Better pose
errors alone do not establish closed-loop riding success.
Report: `results/bike_vision_practiced_training.json`.

On twenty further unseen starts (940–959), this matched visual/practiced pair
passed **19/20**, versus **20/20** for the practiced sensor policy. The remaining
failure is retained in `results/bike_vision_matched_practice_holdout20.json`.
This is a substantial improvement over the mismatched image model's 0/20,
while the visual generalization gate remains short of a complete pass.

The next candidate uses a 16×16 RGB grid (instead of 8×8), forty separate training
rides (960–999), eight validation rides (1000–1007) and five pose-evaluation
rides (1010–1014). Fresh episode-average errors are **0.00491, 0.000326,
0.00547 radians** for roll, pitch and yaw. Its model is separate in
`runs/bike/vision_practiced16/orientation.json`; no previously evaluated failed
ride was added to training. Report: `results/bike_vision_practiced16_training.json`.

The frozen finer-grid candidate then passed **20/20** further unseen paired
starts (1100–1119), matching the practiced sensor policy's **20/20**. Report:
`results/bike_vision_practiced16_holdout20.json`. This passes the defined hybrid
visual riding benchmark; it does not complete full image-only state estimation,
arbitrary-camera generalization, starts from rest or Stage 2 transfer.

```powershell
.venv\Scripts\python.exe scripts/bike_vision_train_motion.py --policy-checkpoint runs/bike/practice_seed1000/practiced_observed_motion.json --train-seeds (960..999) --validation-seeds (1000..1007) --eval-seeds (1010..1014) --grid-size 16 --artifacts runs/bike/vision_practiced16 --output results/bike_vision_practiced16_training.json
.venv\Scripts\python.exe scripts/bike_vision_control.py --checkpoint runs/bike/vision_practiced16/orientation.json --policy-checkpoint runs/bike/practice_seed1000/practiced_observed_motion.json --seeds (1100..1119) --output results/bike_vision_practiced16_holdout20.json
```

Reports: `results/bike_vision.json`, `results/bike_vision_motion.json`,
`results/bike_vision_motion_training.json`, `results/bike_vision_control.json`.
The motion benchmark also saves `docs/media/bike/vision_motion.mp4`.

```powershell
.venv\Scripts\python.exe scripts/bike_vision.py
.venv\Scripts\python.exe scripts/bike_vision_motion.py
.venv\Scripts\python.exe scripts/bike_vision_train_motion.py
.venv\Scripts\python.exe scripts/bike_vision_control.py
```

## Reproduce existing motion imitation and practice

```powershell
.venv\Scripts\python.exe scripts/bike_stage7.py --seeds 10 11 12 13 14 15 16 17 18 19 --output results/bike_stage7_holdout.json
.venv\Scripts\python.exe scripts/bike_observation.py
.venv\Scripts\python.exe scripts/bike_practice.py
.venv\Scripts\python.exe scripts/bike_replicates.py
.venv\Scripts\python.exe -m pip install -e .[bike-video]
.venv\Scripts\python.exe scripts/bike_observation.py --video docs/media/bike/expert.mp4
.venv\Scripts\python.exe -m pytest tests/test_bike.py -q
```

## Remaining

Motion extraction from videos, independent noisy observations, independent-layout
practice comparisons, Stage 2 transfer, perturbations, starts from rest, pedaling geometry,
and a full rider remain.
The 10-metre baseline gate passes; the complete Stage 7 research gate does not.
