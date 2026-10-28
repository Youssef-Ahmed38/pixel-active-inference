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

## Video-only apprenticeship and attached-camera riding (2026-10-28)

**Frozen final seeds 16000–16019:** the video imitation student reaches the 10-metre goal on **20/20** starts from its attached camera and proprioception. The practiced video branch reaches **19/20**, so practice did not improve held-out success. The existing state student is 20/20, hybrid 18/20, and classical expert 12/20 on the same physical starts. This is a limited, strongly calibrated video benchmark, not unstructured real-world video learning or a complete Stage 7 pass.

### What is learned and what is given

`video_sensors.py` adds five **massless coloured visual markers**, two fixed external cameras (512×512, 50 Hz), and a camera rigidly mounted on the frame. The cameras/markers do not change body masses, inertias, actuators or contacts. Fixed-camera pixel centroids are triangulated, then rigid alignment and known hinge axes reconstruct motion. No expert pose/control label is an input to this reconstruction or policy fit. Learning files contain only `rgb` and `times`; the expert states and controls are in separate audit files opened only for post-fit scoring. Cached states are **RGB estimates**, fingerprinted against their pixel source and decoder, not simulator states.

Given: tag identities/geometry, calibrated stereo and on-board cameras, known initial joint rest, wheel radius and rolling approximation, MuJoCo dynamics, servo gains/damping/inertia, actuator limits, a geometric horizon/road decoder, and the linear no-joint-feedback policy architecture. The pose decoder is authored inverse graphics, **not a learned neural pose encoder**. Learned: policy weights from visually inferred targets, plus optional practice mutations selected from own-sensor fitness. There is no expert-action cloning, supervised expert-state fitting, locomotion transfer, or cloud training.

Four expert recordings (12000–12003) contain 958 synchronized time frames / 1,916 RGB views. One expert recording times out and remains included. An 11-frame cubic smoother uses future frames only during offline demonstration processing. Missing markers cause abstention/interpolation; the fit uses only 540 fully observed temporal windows and weights each teaching ride equally. Robust ridge regression uses inferred actions only. The two joint-feedback rows are fixed to zero as a given architecture: including current joint angles encouraged trivial copying of the reconstructed servo targets instead of learning balance. All remaining coefficients, including their signs, are fitted.

The inverse remains **given and approximate**. The selected low-bandwidth inverse uses rolling longitudinal inertia and each servo’s own inertia, gravity and damping; it does not identify full contact dynamics from video. Reconstructed wheel rotation assumes rolling without slip. The original full constrained inverse was too sensitive to video-rate acceleration and contact-height noise.

### Development failures retained

| Candidate | Development seeds 13000–13002 | Result |
|---|---|---|
| Full inverse, unrestricted feedback | 0/3 goals | Failed at about 1.3–1.5 m |
| Servo inverse, unrestricted feedback | 0/3 goals | About 2.8–5.0 m |
| Servo inverse, no joint feedback | 3/3 goals | Selected before final evaluation |
| No joint feedback, constant torso target | 0/3 goals | Rejected |

The selected imitation branch also passes 13003–13005. No final layout was used for this selection. See the `bike_video_development_*.json.gz` artifacts; failed variants were not overwritten.

### Own senses at deployment

The frame-mounted camera is addressed by its fixed camera ID, not by a root-tracking `lookat`. 128×128 RGB arrives at 10 Hz; wheel/joint encoders and the body gyro arrive at 50 Hz. Wheel speed is encoder rotation × 0.32 m, **not root linear velocity**. Absolute lean comes from the visible ground/sky horizon, heading from the marked lane’s vanishing point, and gyro integration bridges camera samples. The policy receives only RGB plus an `OwnSenses` record; it cannot access the env, root pose/velocity, simulator quaternion, world coordinates, or a tracking camera. Joint encoders are available but the selected architecture does not use their feedback in the high-level action.

This benchmark gives a black sky, flat ground, white lane edges and ideal simulated proprioception. The learner has not been shown to work on arbitrary scenery, camera calibration, lighting, noise, unmarked monocular teaching video, wheel slip, starts from rest, obstacles or a full humanoid rider.

### Practice uses the same senses

Optimizer seed 15000, starts 14000–14005, 24 mutations plus the initial policy: 150 practice rides. Accepted mutation attempts: `[6]`. The six sensed goals were already reached at attempt zero. Fitness uses integrated wheel speed × cos(image/gyro heading), maximum camera/gyro lean, and a given 0.6-rad own-lean cutoff. Simulator displacement, fall and success fields are audit-only. This self-odometry objective is approximate and is not a claim of sample-efficiency improvement. The mutation that improved practice fitness introduced a held-out failure at 16006.

### Frozen paired evaluation

Each condition uses the same twenty initial states: speed uniform 2.5–4.5 m/s, roll ±0.1 rad, 10 m forward displacement, ten-second limit. Initial state hashes match across all five riders. The legacy expert/state/hybrid run in their original scene; the new scene adds only nonphysical visual aids. The legacy hybrid retains its root-tracking camera and externally supplied speed, so it is explicitly privileged. The state/hybrid policies are the existing optimizer-1000 practiced student and its matched 16×16 RGB model, not newly fitted baselines.

| Rider | Goals reached | Success / Wilson 95% interval |
|---|---:|---|
| expert | 12/20 | 60.0% [38.7, 78.1] |
| state_student | 20/20 | 100.0% [83.9, 100.0] |
| hybrid_student | 18/20 | 90.0% [69.9, 97.2] |
| video_imitation | 20/20 | 100.0% [83.9, 100.0] |
| video_practiced | 19/20 | 95.0% [76.4, 99.1] |

These are twenty shared starts, not 100 independent layouts. A 20/20 sample is not proof of 100% reliability. Both video branches were registered before final collection; neither was retuned or substituted after seeing final failures.

| Failed rider | Seeds and failure type |
|---|---|
| expert | 16001 (timeout, -6.86 m), 16002 (timeout, -6.54 m), 16004 (timeout, -5.94 m), 16006 (timeout, 9.66 m), 16010 (timeout, -3.27 m), 16015 (timeout, -6.98 m), 16017 (timeout, -4.74 m), 16019 (timeout, -6.41 m) |
| hybrid_student | 16006 (fall, 5.88 m), 16010 (fall, 9.81 m) |
| video_practiced | 16006 (timeout, 2.34 m) |

### Post-fit teaching audit (not fitting labels)

| Teaching seed | Supported windows | Roll MAE (rad) | Speed MAE (m/s) | Inferred torque / steer / torso target MAE |
|---|---:|---:|---:|---|
| 12000 | 77 | 0.0008 | 0.0137 | 2.861 Nm / 0.0407 rad / 0.1981 rad |
| 12001 | 100 | 0.0032 | 0.0114 | 2.458 Nm / 0.0432 rad / 0.1915 rad |
| 12002 | 267 | 0.0097 | 0.0172 | 4.623 Nm / 0.0519 rad / 0.1660 rad |
| 12003 | 96 | 0.0067 | 0.0132 | 2.922 Nm / 0.0213 rad / 0.1770 rad |

The inferred actions are imperfect, especially torso targets; successful riding must not be described as exact expert-control recovery. Audit fields were first compared after the policies and final run were frozen.

### Truth boundary, video, and limitations

All 40 new-rider episodes replay from their actual 1362 lossless RGB frames and proprioception, with identical features/actions to 1e-10 tolerance and no scoring-state input. Every practice score and accept/reject decision is recomputed from its own-sensed fields. Tests also forbid root array access in the encoder adapter, verify physical camera mounting and unchanged dynamics, and alter expert audit attributes without changing fitted weights. All 26 bike tests pass.

On-board clip: [video-only student ride](media/bike/video_onboard_student.mp4), seed 16000, 31 original sensor frames at 10 Hz, 13204 bytes. It is the first successful practiced final episode (otherwise the first failure); it was encoded from recorded decision images without rerunning the environment. The failed practiced episode remains in the results.

This closes the calibrated marker-based video apprenticeship / attached-camera moving-start test. It does not establish natural-video motion discovery, learned inverse dynamics, an unstructured learned vision encoder, broad cycling competence, Stage 2 transfer, or reliable benefit from practice. Existing checkpoints and default bike behavior remain unchanged.

### Seeds and reproduction

Teaching 12000–12003; development/model selection 13000–13005; practice starts 14000–14005; practice optimizer 15000; frozen final starts 16000–16019. Static camera debug seed 20000 was development only. The final plan inventories recorded previous bike seed fields and rejects overlap. Models, pixel inputs, cache lineage and final source hashes are retained.

```powershell
$env:PYTHONPATH='.'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe scripts/bike_video_study.py collect --seeds 12000 12001 12002 12003
.venv/Scripts/python.exe scripts/bike_video_study.py train --seeds 12000 12001 12002 12003 --inverse servo --features no_joints --policy results/bike_video_no_joints.json
.venv/Scripts/python.exe scripts/bike_video_study.py dev --policy results/bike_video_no_joints.json --seeds 13000 13001 13002 --output results/bike_video_development_no_joints.json.gz
.venv/Scripts/python.exe scripts/bike_video_study.py dev --policy results/bike_video_no_joints.json --seeds 13003 13004 13005 --output results/bike_video_development_additional.json.gz
.venv/Scripts/python.exe scripts/bike_video_study.py practice --policy results/bike_video_no_joints.json --seeds 14000 14001 14002 14003 14004 14005 --updates 24 --output results/bike_video_practiced.json
.venv/Scripts/python.exe scripts/bike_video_final.py register
.venv/Scripts/python.exe scripts/bike_video_final.py run
.venv/Scripts/python.exe scripts/bike_video_final.py score
.venv/Scripts/python.exe scripts/bike_video_final.py audit
.venv/Scripts/python.exe scripts/bike_video_final.py media
.venv/Scripts/python.exe scripts/bike_video_report.py teaching-audit
.venv/Scripts/python.exe scripts/bike_video_report.py
.venv/Scripts/python.exe -m pytest tests/test_bike*.py -q --basetemp=data/pytest_bike_video_new
```

Use clean output paths for reproduction. Collection/final resumption retain completed episodes and reject changed frozen parameters; teaching fits do not overwrite existing models. On this managed Windows host, the actual pytest run wrapped a fresh directory with inherited permissions because mkdir(mode=0700) is inaccessible; the recorded test results are `bike_video_tests_frozen.json.gz`. No existing temp directories were removed.

Artifacts under `results/bike_video_*`: RGB `.npz` (gzip-wrapped above 5 MB), separate truth audits, visually estimated motion caches, training/development failures, both policy checkpoints, every practice attempt, frozen plan, 100 final episode records including raw camera/proprioception traces, summary, replay audit and media provenance. Result paths are project-relative; no user folders or cache paths are recorded.
