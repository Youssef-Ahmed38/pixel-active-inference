# Stage 2: a humanoid that learns to stand, walk, turn and run

The Unitree G1 humanoid (MuJoCo Menagerie, `scene_mjx.xml`) learns locomotion from its own
exploration and practice, on this laptop (one RTX 5060, CPU simulation of 2048 bodies at once).
Goals are velocity and heading commands, so the maze and the bike stages can reuse the controller
through `pai/locomotion/skill.py` (`move(velocity, heading)`, `turn`, `stop`).

<img src="media/locomotion/g1_locomotion.gif" width="400">

*Released controller (v1) on a fresh body (seed 50100): stand, walk 0.8 m/s, turn to +90°, run
2.0 m/s, slow down, take a push; 0 falls ([MP4](media/locomotion/g1_locomotion.mp4)).*

## What is learned and what is given

| Part | Learned or given | Where |
|---|---|---|
| Body model (predicts the next proprioceptive state from the motor command; 5-member ensemble) | **learned** from motor babbling, then from all practice | `world_model.py`, `babble.py` |
| Leg controller (12 servo targets every 20 ms) | **learned** by PPO practice | `ppo.py`, `policy.py` |
| Curiosity: ensemble disagreement as an exploration bonus, annealed to zero | **learned** model, given weight | `ppo.py` |
| Curriculum stand → balance (pushes) → walk → turn → run | given order, **automatic** advancement on practice performance | `ppo.py:ready_to_advance` |
| Preferences (the goal): command tracking, upright, smooth, a foot-height rhythm from a clock whose frequency and duty factor scale with the commanded speed | **given** (authored preference terms) | `env.py:_reward` |
| Waist and arms | **given**: held at the default posture | `body.py` |
| Randomised dynamics (friction, masses, payload, servo strength) | given ranges, sampled per body | `env.py` |
| Fall-cause inference (push / slip / own gait error) from unexplained body-model surprise | **learned** body model, given hypotheses | `falls.py` |

The rhythm preference is a stated simplification: it tells the body *that* feet should alternate
at a speed-dependent tempo, not which joint angles to use. Removing it is open work.

## Motor babbling: curiosity explores better

Two explorers with the same budget (307,200 transitions each, `results/locomotion/babble.json`):

| Explorer | State spread | Model error on random data | Model error on curious data |
|---|---|---|---|
| random (smooth noise) | 0.86 | 0.200 | 0.324 |
| **curious** (picks the action the ensemble disagrees about most) | **1.10** | **0.139** | 0.183 |

The curious explorer covers more of the body's states, and the model it learns predicts the random
explorer's data better than the random explorer's own model does.

## Results on fresh seeds

Training: seed 0. Development checks: seeds 777, 51000+, 52000+. **Final gates: seeds 50000–50006,
never used before**; video 50100. Each test uses 20 bodies with randomised dynamics.

| Test | Gate | **v1** (released) | v2 (push-trained) |
|---|---|---|---|
| Stand 30 s | stands | ✅ 20/20, drift 0.11 m | ✅ 20/20, drift 0.76 m |
| Walk 10 m at 0.8 m/s | ≥ 90% at ≥ 0.5 m/s | ✅ 20/20, 0.77 m/s | ✅ 20/20, 0.71 m/s |
| Turn to a commanded heading | within 15° | ✅ 20/20, median 2.5° | ❌ 10/20, median 15.8° |
| Run at 2.0 m/s | ≥ 1.5 m/s | ✅ 20/20, 1.99 m/s | ❌ 3/20, 1.29 m/s |
| Push recovery, 0.5 m/s (83 N) | — | 93% | 100% |
| Push recovery, 1.0 m/s (167 N) | — | 53% | **100%** |
| Push recovery, 1.25 m/s (208 N) | — | 17% | **93%** |
| Push recovery, 1.5 m/s (250 N) | — | 0% | 60% |

Files: `results/locomotion/final_v1.json`, `final_v2.json`.

**v1** trained 92 minutes (125 M steps) through the whole curriculum. It passes every gate.
**v2** continued v1 for 57 minutes with pushes up to 1.25 m/s (v1 had practised up to 0.6 m/s).
It became far more robust to pushes but **forgot part of its heading and speed control**
(catastrophic forgetting): practice dominated by large pushes crowded out precise tracking. v1 is
the released controller; v2 is kept as the push-robust variant. Next: v3, continued from v1 with
pushes increased gradually and the tracking preferences weighted higher, to keep both.

## Fall causes (development check, not yet a fresh gate)

On 180 injected falls (60 each, seed 51006, calibration on 90 undisturbed falls, seed 61000):
accuracy **59%**. Own gait error is recognised 97%, push 42%, slip 38%. Pushes and slips are often
taken for the body's own error, so the cause model needs better evidence (e.g. sole forces). This
has not been run on fresh seeds.

## Reproduce (laptop GPU)

```powershell
.venv\Scripts\python.exe scripts/locomotion_babble.py --config configs/locomotion.yaml
.venv\Scripts\python.exe scripts/locomotion_train.py --config configs/locomotion.yaml
.venv\Scripts\python.exe scripts/locomotion_eval.py --config configs/locomotion.yaml --policy runs/locomotion/policy_v1/policy.pt --ckpt runs/locomotion/policy_v1/ckpt_last.pt --name final_v1
.venv\Scripts\python.exe scripts/locomotion_video.py --policy runs/locomotion/policy_v1/policy.pt --seed 50100
.venv\Scripts\python.exe -m pytest -q tests/test_locomotion.py
```

v1 was trained with the curriculum's default pushes (`run` stage `push_dv` 0.6); v2 with the
`stages` block in `configs/locomotion.yaml` (`run` stage `push_dv` 1.25). Checkpoints are in
`runs/` (not in git).
