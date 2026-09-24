# Pixel Active Inference (PAI)

Hierarchical, pixel-based active inference for a simulated Franka Panda, building on
PixelAI (Sancaktar et al., 2020). Everything runs in simulation (MuJoCo).

**Status: Phase 0 complete.** Foundation plus a PixelAI reproduction baseline.
The full plan is in [docs/ROADMAP.md](docs/ROADMAP.md) (main target, Stage 1 phases, Stages 2–7), the 8-week
application sprint in [docs/SPRINT.md](docs/SPRINT.md), verified literature in [docs/LITERATURE.md](docs/LITERATURE.md), and time and compute estimates in
[docs/TIME_AND_COMPUTE.md](docs/TIME_AND_COMPUTE.md).
The phase table below predates the roadmap.

| Phase | Content | State |
|---|---|---|
| 0 | MuJoCo Panda env, disturbance suite, PixelAI baseline, benchmarks, Kaggle templates | done (Kaggle notebook untested) |
| 1 | Latent world model (DINOv2 latents + action-conditioned transformer ensemble), 256px diffusion decoder, learned precision fusion, new L0 | – |
| 2 | L1 expected-free-energy planning (MPPI in latent space) with epistemic term, surprise escalation | – |
| 3 | L2 goals: image / language / abstract task goals, subgoal generation | – |
| 4 | Continual learning, online body-parameter inference, full robustness suite | – |
| 5 | Baselines (IBVS+IK, ACT, Diffusion Policy, DreamerV3/TD-MPC2, OpenVLA), evaluation, write-up | – |

## Phase 1a: tabletop scene (in progress)

`TabletopEnv` ([pai/envs/tabletop.py](pai/envs/tabletop.py), [configs/tabletop.yaml](configs/tabletop.yaml)):
Panda, table, 4 coloured blocks, a plate and a bowl; three cameras; segmentation masks per object.
- **Control:** end-effector deltas + yaw + gripper through damped least-squares IK, or joint velocities.
- **Compliant arm** (gravity compensation, 15% servo stiffness): a 40 N push deflects the hand 20 mm, 5x the stiff arm.
- **Relations** from privileged state ([pai/envs/predicates.py](pai/envs/predicates.py)): on, in, grasped, lifted, upright, near, left_of.
- **Ground-truth event log** ([pai/envs/events.py](pai/envs/events.py)): contacts, grasp/release, slips, relation
  changes and disturbances, debounced and timestamped. This is the answer key for the agent's explanations.
- **New disturbances:** object friction, and mass on any body.
- **Scripted pick-and-place** ([pai/envs/scripted.py](pai/envs/scripted.py)): 97.5% success over 80 episodes
  (4 object/target pairs x 20 layouts) with the compliant arm.

## Phase 1b: vertical slice (in progress, sprint week 1)

`on(red, plate)` end to end through every level, on privileged object state
([pai/agents/slice_agent.py](pai/agents/slice_agent.py)):

| Level | Module |
|---|---|
| L2 goals | [pai/goals/relations.py](pai/goals/relations.py): relation as ordered subgoal preferences; L2 sets the gripper mode; carrying subgoals have a maintenance condition (object held) and fall back to re-grasping |
| L1 planning | [pai/planning/mppi.py](pai/planning/mppi.py): MPPI in the learned world model, expected free energy (pragmatic + epistemic), smoothness prior, cost-relative temperature |
| World model | [pai/world/](pai/world/): transformer over entity tokens, 5-member ensemble, per-feature learned variance; wrist force predicted from state but never read |
| Surprise and causes | [pai/causes/inference.py](pai/causes/inference.py): calibrated surprise; Bayesian comparison of none / push / heavier_object over position and force errors |
| Memory and reports | [pai/memory/](pai/memory/): episode records, readable reports, video overlay |

Results (60 episodes: 20 each for no disturbance, a 30–50 N push at a random time, and a block
0.3–0.6 kg heavier than it looks; surprise calibrated on 8 separate clean episodes):

| Condition | Task success | Cause identified | Estimate vs truth |
|---|---|---|---|
| none | 100% | 100% | – |
| push | 100% | 95% | onset within 0.06 s on average |
| heavier block | 95% | 95% | extra mass estimated (not yet scored against the true mass) |
| **overall** | **98%** (gate > 70%) | **97%** (gate > 70%) | |

Details are in `results/slice_eval.md` and the episode reports in `results/slice_reports.md`.
Earlier runs are kept for comparison:

| Run | Change | Task | Cause |
|---|---|---|---|
| `results/slice_v1/` | first run, release 2 cm above the plate | 77% | 92% |
| `results/slice_v2/` | gripper mode from L2, re-grasp fall-back | 85% | 93% |
| `results/slice_v4/` | per-step surprise calibration | 82% | 93% |
| `results/slice_v5/` | plate contact fix, hold still while opening | **98%** | **97%** |

The v4 failures were "released, but not on the plate". Replaying the recorded actions showed that a
slightly tilted block touched the thin plate cylinder at a single contact point and pivoted through
it. A 3 mm contact margin on the plate fixed all replayed cases. The one v5 failure is a heavier
block that was never grasped within the step limit.

- World model on 100 held-out episodes: 1.4 mm (0.1 s), 5.9 mm (0.5 s), 21 mm (2 s) for moving
  entities, vs 14 / 39 / 65 mm for a commanded-motion baseline.
- Without wrist force sensing, cause identification was 0/4. Persistent causes (a heavier block)
  vanish from one-step prediction errors. Predicting the wrist force from the state, without
  reading it, is what makes them visible.

Fixed along the way:
- Payload disturbances now scale inertia and refresh MuJoCo's derived constants
  (`mj_setConst` on scratch data).
- The Menagerie gripper closes with only ~2 N, so it is scaled 10x.
- Slip events are debounced.

## Phase 0 results: PixelAI baseline

Decoder: 11.4M parameters, 60k renders, 30k steps (~45 min on an RTX 5060 laptop GPU),
validation MSE 1.9e-4. Agent gains: beta = 10, pi_mu = 40 (sweep in `results/sweep_*.json`).
Full tables: [results/pixelai_suite.md](results/pixelai_suite.md). 20 episodes per row.

| Goal distance (median start) | 5 cm | 7 cm | 11 cm | 12 cm | 23 cm |
|---|---|---|---|---|---|
| Success (within 2 cm) | 100% | 100% | 100% | 80% | 35% |

| Disturbance (11 cm goals) | none | push | occlusion | lighting | camera shift | payload | sensor noise |
|---|---|---|---|---|---|---|---|
| Success | 100% | 100% | 100% | 100% | **5%** | 100% | 95% |

What these show:
- **PixelAI is local.** The gradient of the pixel error fades once the predicted arm stops
  overlapping the goal arm, so success collapses with distance. Phase 1 moves goals and
  inference into a learned latent space.
- **A 5 cm camera shift breaks it.** The goal image comes from the original viewpoint. The
  agent drives the arm to the pose that *looks* like the goal from the new viewpoint, and
  vision and proprioception then disagree with no way to resolve it. Phase 4 targets this
  with online body and camera parameter inference.
- **The high occlusion and lighting scores flatter vision.** With pi_v = 1 on a per-pixel mean
  error, proprioception dominates state estimation, and vision mostly supplies the goal.
  A vision-only perception test (pi_q = 0) is needed to stress the visual pathway.
- **Push and payload are absorbed by the stiff position servos** (see configs/disturbances.yaml).
  A compliant control mode is needed before these measure the agent itself.

## Setup

```bash
python -m venv --system-site-packages .venv      # reuses an existing CUDA torch install
.venv/Scripts/python -m pip install -e .[dev]    # Linux/macOS: .venv/bin/python
python scripts/fetch_assets.py                   # Franka Panda from MuJoCo Menagerie (pinned commit)
python -m pytest
```

On Windows laptops with hybrid graphics, `pai.envs` loads the NVIDIA driver before the
first GL context. Otherwise OpenGL runs on the integrated GPU, about 60x slower (26 vs 1625 fps).
On headless Linux (Kaggle, Colab) set `MUJOCO_GL=egl`.

## Phase 0 workflow

```bash
python scripts/benchmark.py                                   # throughput on this machine
python scripts/collect_data.py                                # 60k (q, image) pairs -> data/pixelai
python scripts/train_decoder.py                               # PixelAI decoder, resumes automatically
python scripts/run_eval.py --no-vision agent.goal_mode=joints # privileged sanity check (no decoder)
python scripts/run_eval.py --decoder runs/pixelai_decoder/decoder.pt              # image-goal reaching
python scripts/run_eval.py --decoder runs/pixelai_decoder/decoder.pt --task perception
python scripts/run_eval.py --decoder runs/pixelai_decoder/decoder.pt --disturbance occlusion
```

Every script takes `--config` plus `key.sub=value` overrides, e.g. `train.batch_size=128`.
Disturbance presets are in [configs/disturbances.yaml](configs/disturbances.yaml): push,
occlusion, lighting, camera_shift, payload, sensor_noise.

### Kaggle (2x T4)

Use [notebooks/kaggle_train_decoder.ipynb](notebooks/kaggle_train_decoder.ipynb). It clones
the repo at a pinned commit and renders with EGL. It runs `torchrun --nproc_per_node=2` when
two GPUs are present and writes checkpoints to `/kaggle/working`. To continue a run in a new
session, attach the previous version's output and set `RESUME_FROM`.

## Layout

```
pai/envs/       MuJoCo scenes (MjSpec around Menagerie panda.xml): PandaEnv, TabletopEnv, disturbances,
                predicates, event log, scripted pick-and-place
pai/world/      entity tokens, transition data, ensemble entity world model (with force prediction)
pai/goals/      relational goals as subgoal preferences, with maintenance conditions
pai/planning/   MPPI with expected free energy
pai/causes/     calibrated surprise, Bayesian cause inference (incl. "unknown"), counterfactual credit
pai/memory/     episodic memory with trajectories, readable reports, video overlay
pai/perception/ frames + masks, DINOv2 features, object slots, slots -> entity tokens
pai/thinker/    HRM thinker prototype, synthetic cause tasks with exact Bayesian teachers, baselines
pai/models/     PixelAI decoder g(q) -> image
pai/agents/     PixelAI agent (Phase 0) and the vertical-slice agent (Phase 1b)
pai/data/       parallel data collection, memory-mapped dataset
pai/train/      resumable single/multi-GPU training (AMP: bf16 where supported, fp16 on T4)
pai/eval/       reaching / perception episodes and metrics
scripts/        CLI entry points (slice_collect, slice_train_wm, slice_eval, eval_world_model, ...)
configs/        default.yaml (Phase 0), tabletop.yaml (Phase 1), disturbances.yaml, thinker*.yaml
notebooks/      Kaggle templates (decoder training; run any script)
```

## The PixelAI agent

The full derivation is in [pai/agents/pixelai.py](pai/agents/pixelai.py). The belief over
joint angles and their velocities, (mu, mu'), follows free-energy gradients. There are three
prediction errors, each weighted by a precision: proprioception `q - mu`, vision
`s - g(mu)`, and dynamics `mu' - f(mu)`. The goal enters as an attractor f(mu) that pulls the
belief towards the goal image. Actions (joint velocities) reduce the same free energy.

Tuning notes, so nobody has to rediscover them:
- The belief dynamics have damping ratio sqrt(k_mu pi_mu / (4 beta)). With pi_mu = 1 the
  belief itself oscillates. Keep pi_mu >= 4 beta.
- The updates are explicit Euler steps, so the visual precision pi_v has a stability ceiling.
  Too large a value overshoots into a different local minimum or diverges. Adaptive step
  sizes are a Phase 1 item.
