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

Training: seed 0. Development checks: seeds 777, 51000+, 52000+. **Final gates: v1 and v2 seeds
50000–50006, v3 seeds 54000–54006, each never used before**; videos 50100 (v1) and 54100 (v3). Each
test uses 20 bodies with randomised dynamics.

| Test | Gate | v1 | v2 (push-trained) | **v3** (released, 2026-10-28) |
|---|---|---|---|---|
| Stand 30 s | stands | ✅ 20/20, drift 0.11 m | ✅ 20/20, drift 0.76 m | ✅ 20/20, drift 0.57 m |
| Walk 10 m at 0.8 m/s | ≥ 90% at ≥ 0.5 m/s | ✅ 20/20, 0.77 m/s | ✅ 20/20, 0.71 m/s | ✅ 20/20, 0.74 m/s |
| Turn to a commanded heading | within 15° | ✅ 20/20, median 2.5° | ❌ 10/20, median 15.8° | ✅ 20/20, median 1.6° |
| Run at 2.0 m/s | ≥ 1.5 m/s | ✅ 20/20, 1.99 m/s | ❌ 3/20, 1.29 m/s | ✅ 20/20, 1.95 m/s |
| Push recovery, 0.5 m/s (83 N) | — | 93% | 100% | 100% |
| Push recovery, 1.0 m/s (167 N) | — | 53% | **100%** | **100%** |
| Push recovery, 1.25 m/s (208 N) | — | 17% | **93%** | **93%** |
| Push recovery, 1.5 m/s (250 N) | — | 0% | 60% | 33% |

Files: `results/locomotion/final_v1.json`, `final_v2.json`, `final_v3.json` (30 pushed bodies per row).

<img src="media/locomotion/g1_locomotion_v3.gif" width="320"> v3 on a fresh body (seed 54100):
stand, walk, turn to +90°, run, and a 1.25 m/s side push, no fall
(`media/locomotion/g1_locomotion_v3.mp4`).

**v1** trained 92 minutes (125 M steps) through the whole curriculum. It passes every gate.
**v2** continued v1 for 57 minutes with pushes up to 1.25 m/s (v1 had practised up to 0.6 m/s).
It became far more robust to pushes but **forgot part of its heading and speed control**
(catastrophic forgetting): practice dominated by large pushes crowded out precise tracking. v1 is
the released controller; v2 is kept as the push-robust variant.

**v3 keeps both.** It continued v1 in three phases (`configs/locomotion_v3a/b/c.yaml`, 297 M steps,
3.5 h in total): A with pushes up to 0.9 m/s, B up to 1.25 m/s, both with the speed and heading
preferences weighted higher (lin 2.5, yaw 1.6, defaults 2.0 and 0.8); after B the development check
(seeds 52000+) kept every gate except headings (14/20), so C practised heading commands 60% of the
time (was 30%) with yaw weight 2.2 and half the learning rate. The development check after C passed
everything (`results/locomotion/dev_v3.json`, `dev_v3c.json`), and the final gates were then run once
on new seeds 54000+: every gate passes, and push recovery at 1.0 m/s rises from 53% (v1) to 100%,
at 1.25 m/s from 17% to 93%. It is weaker than v2 at 1.5 m/s (33% vs 60%). v3 is the released
controller.

## Fall causes

### Learned sensor likelihoods: fresh gate

**Released v3, final seed 55006: 148/160 correct (92.5%; Wilson 95% CI 87.3-95.7%), versus 65/160 (40.6%) for the unchanged authored method on exactly the same falls.** Balanced accuracy is **92.0% versus 43.6%**. No walking policy, learned body-model weights, or environment physics were retrained or changed.

| Metric | Learned likelihood | Old authored likelihood |
|---|---:|---:|
| Overall accuracy (Wilson 95%) | 148/160 = 92.5% [87.3, 95.7] | 65/160 = 40.6% [33.3, 48.4] |
| Gait recall (Wilson 95%) | 39/48 = 81.2% [68.1, 89.8] | 47/48 = 97.9% [89.1, 99.6] |
| Push recall (Wilson 95%) | 51/52 = 98.1% [89.9, 99.7] | 11/52 = 21.2% [12.2, 34.0] |
| Slip recall (Wilson 95%) | 58/60 = 96.7% [88.6, 99.1] | 7/60 = 11.7% [5.8, 22.2] |
| Balanced accuracy (variant-cluster bootstrap 95%) | 92.0% [87.5, 96.2] | 43.6% [38.6, 48.6] |

Balanced accuracy is the unweighted mean of the three recalls, not a single binomial proportion, so its interval uses 5,000 bootstrap resamples of the **shared dynamics variant** rather than an inapplicable Wilson formula. Wilson intervals for overall accuracy and class recall treat falls as binomial observations; they are descriptive because randomized dynamics variants are reused across bodies. There are 32 variant groups and only one final environment seed. The paired accuracy gain is **51.9 percentage points**, with a variant-cluster bootstrap 95% interval **44.9-58.5 points** (seed 80501). New-only correct: **93**; old-only correct: **10**.

Confusion matrices (true rows, predicted columns; unknown counts as wrong for these three labelled causes):

Learned:

| true / predicted | gait | push | slip | unknown |
|---|---:|---:|---:|---:|
| gait | 39 | 9 | 0 | 0 |
| push | 1 | 51 | 0 | 0 |
| slip | 0 | 2 | 58 | 0 |

Old:

| true / predicted | gait | push | slip | unknown |
|---|---:|---:|---:|---:|
| gait | 47 | 0 | 1 | 0 |
| push | 41 | 11 | 0 | 0 |
| slip | 52 | 1 | 7 | 0 |

### Design: what is sensed, given and learned

`pai/locomotion/fall_causes.py` accepts only the frozen body model's prediction error from proprioception and the body's own motor action, sole heights/velocities, and read-only ankle force/torque readings. It never accepts injected push vectors, friction coefficients, event logs, cause labels, commands, or body indices as inference features. The experiment harness alone owns the intervention and scoring labels. Goal commands are deliberately excluded from the cause feature vector, although physical responses can still reflect the different commands in this controlled benchmark.

The evaluator adds force and torque sensors at the existing left/right foot sites through a local body-spec adapter. These act as ideal **ankle load-cell proxies**, including inertial/joint reaction loads; they are not separately resolved ground reaction forces. Sensor addition changes no mass, inertia, friction, actuator, collision, policy observation dimension, or integration setting. A regression test compares physical model arrays and trajectories with and without instrumentation. Existing sole position/velocity and pelvis linear velocity remain the stated simulator-based sensing simplification; no pixel or hardware sensor validation is claimed.

The monitor retains every transition of the first episode, including the terminal observation supplied before automatic reset. It can reset a body's memory explicitly. Its 16 trace channels include raw pelvis velocity residuals, normalized angular/gravity/joint residual magnitudes, surprise, load-weighted foot sliding, ankle shear/load ratios, load sum/imbalance, number of loaded feet, foot speed and uprightness. A loaded foot uses the given L0 proxy `sole_height < 0.03 m` and `abs(ankle_Fz) > 20 N`. No true contact flag or friction value is read. Full first-episode traces are retained as compressed float32 records in JSON with channel names and a lossless decoder.

Hand-designed evidence statistics include multi-scale integrated residual peaks (5/10/25 steps), medians/90th percentiles of proprioceptive residuals and loaded slipping, support distribution, and strongest unknown-onset before/after contrasts. Searches do not know the intervention time. Both all-history and pre-collapse summaries are kept, avoiding a forced removal of immediate-push evidence. Stable load scale is estimated from an early sensory baseline; the initial 15 samples are discarded. These features are engineered measurement summaries, not learned causal rules. Onset estimates are descriptive contrast maxima, not validated Bayesian onset posteriors.

**Learned:** global log1p feature centers/scales, and a mean plus full regularized covariance for each of gait, push and slip, fitted from labelled development falls. Posterior is normalized prior times fitted multivariate Gaussian likelihood. **Given:** feature definitions, model family, equal known-cause priors, 0.03 unknown prior, covariance floors and the broad Student-t catch-all family (3 degrees of freedom, scale multiplier 4). The unknown component's pooled center/scale comes from development evidence. It has no labelled real-unknown examples and no measured open-set recall. Zero unknown outputs on the final set do not validate unknown detection. The test suite only verifies rejection of a synthetic extreme outlier. Posterior probabilities are not empirically calibrated and can be confidently wrong.

### Development selection and the window hypothesis

Development seeds **53006, 53007, 53008** each start 60 bodies per condition. Their **463 scored falls** comprise 133 gait, 150 push and 180 slip falls. Model selection uses leave-one-environment-seed-out validation, then refits the selected distribution on all three development sets. No final or final-calibration fall enters that fit. The three covariance shrinkage candidates are 0.1, 0.5 and 0.9. Best development balanced accuracy per feature family:

| Feature family | Best shrinkage | Seed-held-out balanced accuracy |
|---|---:|---:|
| Full episode + ankle loads | 0.1 | 88.2% |
| Recent history + ankle loads | 0.1 | 92.6% |
| Old U/S only, but learned likelihoods | 0.9 | 69.7% |

The frozen winner uses **64 recent-history features**, shrinkage 0.1. Its input crops the last 100 control transitions and discards the first 15 of that crop under the same preprocessing rule, leaving 85 (1.7 s); pre-collapse summaries additionally omit the final 10 steps. The full-episode candidate and its onset records are available, but **full-history retention is not the selected classifier's source of improvement here**. The learned U/S-only ablation improves over the authored model but remains well below the load/time-course features.

The final pushed bodies fell **0.20-0.76 s** after injection (median 0.28 s). There were **zero push falls more than two seconds after injection**. Thus this gate does **not** support the proposed explanation that old push evidence had fallen outside the ring buffer, and it does not validate detection of long-latency pushes. A synthetic regression demonstrates that full-history extraction preserves an early impulse after the old ring forgets it; real delayed-fall validation remains unbuilt.

### Fresh protocol, records and failures

The released policy is `runs/locomotion/policy_v3/policy.pt`; the frozen learned body model is loaded from `runs/locomotion/policy_v3/ckpt_last.pt`; config is `configs/locomotion_v3c.yaml`. Final calibration seed **65000** supplies **67 undisturbed gait falls from 90 starts**, used **only by the old method** to fit its original gait U/S calibration. The new model remains frozen. Final seed **55006** starts 180 bodies, all alive at intervention time 3.0 s. During the following 4.0 s, 48 gait, 52 push and 60 slip bodies fall; 12 gait and eight push bodies do not fall and are reported as nonfalls, not silently classified. There are no pre-injection falls. The injection procedure, dose ranges and gait/slip commands are the existing evaluator's, unchanged.

GPU neural inference is chunked at **48 bodies**, with one Torch CPU thread and six simulator worker threads. The policy and body-model checkpoint hashes were verified unchanged after the gate. No policy/body-model training or physics edits were made. Final simulator episodes were collected once; subsequent scoring and audits replay the saved records without re-running or tuning the gate. Model/data/source hashes and commands are recorded. `fall_causes_frozen_source.json` preserves the exact collection/scoring sources used at the gate; only the post-processing `audit` CLI entrypoint was consolidated afterwards, without changing the collector, feature extractor, fitted likelihood or scoring functions.

The original U/S buffer is populated from the same body-model transitions as the new monitor. An independent reconstruction from the saved sensor traces matches every old U/S pair within **1.6e-7** (float32 storage rounding of sole speed). The gate also re-extracts every full-history feature vector from its compressed trace and verifies exact equality. The comparison therefore uses the **same 160 falls**, not two different random runs. Main records include selected model inputs, whole-history features/onsets, both posteriors and the separate intervention answer key.

Known failures and limits:

- **Nine gait falls are called push**, versus only one gait error under the old method. Gait recall falls from 97.9% to 81.2%; the aggregate improvement must not hide that loss.
- **One push is called gait; two slips are called push.** Some wrong predictions have posterior mass near one, so calibrated uncertainty/abstention remains needed.
- The gate classifies a controlled injected condition, not a counterfactually proven cause of every fall. Gait uses out-of-practice velocity/yaw commands and slip also changes walking speed; although commands are not features, their sensory consequences can confound class identity. Concretely, gait falls follow commands of 3.5 or -2.0 m/s with yaw up to 3 rad/s, pushes happen at 0.6 m/s and slips at 1.2 m/s, so part of the gain may be recognising the speed regime rather than the cause. The control that would settle it (pushes and slips at matched commanded speeds) has not been run. Natural gait failures, mixed causes, hardware noise and real unknown causes remain untested.
- MuJoCo emitted instability warnings during development. The first two development collections retained console warnings (4 and 5 respectively); explicit warning capture was then added, with four warnings in the third set. All falls were retained. The final batch recorded two QACC warnings at collection steps 283 and 331, both with simulator body time 0.436 s. These are consistent with automatically reset bodies after their first fall, but warnings are batch-level and cannot be assigned conclusively to a scored body. No warning-based exclusions were made; numerical validity is a limitation of this simulator protocol.
- The final result is one fresh environment seed, with reused domain variants. Wilson intervals assume per-fall binomial sampling; grouped intervals better reflect that dependence but do not establish universal robustness.

### Exact commands

Run from the main checkout, using project Python with `PYTHONPATH=.`. These are the recorded commands; existing output files are protected against overwrite. Use new `fall_causes_*_repro.json` names for reproduction, and recognize that reusing these seeds is not a new holdout.

```powershell
$env:PYTHONPATH='.'
$env:PYTHONIOENCODING='utf-8'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py collect --seed 53006 --per 60 --output results/locomotion/fall_causes_dev_53006.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py collect --seed 53007 --per 60 --output results/locomotion/fall_causes_dev_53007.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py collect --seed 53008 --per 60 --output results/locomotion/fall_causes_dev_53008.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py fit --inputs results/locomotion/fall_causes_dev_53006.json results/locomotion/fall_causes_dev_53007.json results/locomotion/fall_causes_dev_53008.json --output results/locomotion/fall_causes_model_frozen.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py collect --seed 65000 --per 90 --conditions gait --output results/locomotion/fall_causes_calibration_65000.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py collect --seed 55006 --per 60 --output results/locomotion/fall_causes_final_55006.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py score --input results/locomotion/fall_causes_final_55006.json --model results/locomotion/fall_causes_model_frozen.json --calibration results/locomotion/fall_causes_calibration_65000.json --output results/locomotion/fall_causes_gate_55006.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe scripts/locomotion_fall_causes.py audit --input results/locomotion/fall_causes_final_55006.json --gate results/locomotion/fall_causes_gate_55006.json --output results/locomotion/fall_causes_audit_frozen_55006.json
C:\projects\pixel-active-inference\.venv\Scripts\python.exe -m pytest (Get-ChildItem tests/test_locomotion*.py).FullName -q -p no:cacheprovider --basetemp=data/pytest_locomotion_falls_fresh
```

On the managed Windows executor, fresh pytest temp directories need normal inherited permissions rather than Python's owner-only `mkdir(0o700)` ACL. The test invocation used a scoped in-process `os.mkdir` wrapper applying only to its fresh workspace temporary base; existing ACLs and test expectations were unchanged. All **16 locomotion tests pass** (4.01 s, including the four new evidence/likelihood tests). Policy, model and physics invariance checks, trace roundtrip, early-impulse retention, loaded-foot gating, learned-label permutation, normalized posteriors and synthetic unknown rejection are covered.

Exact managed-executor test launcher (choose a fresh base on another run):

```powershell
@'
import os
from pathlib import Path
import pytest
base=Path('data/pytest_locomotion_falls_final_20261028').resolve()
assert not base.exists()
original=os.mkdir
def inherited(path,mode=0o777,*a,**kw):
    target=Path(path).resolve()
    if os.name=='nt' and mode==0o700 and (target==base or target.is_relative_to(base)):
        mode=0o777
    return original(path,mode,*a,**kw)
os.mkdir=inherited
raise SystemExit(pytest.main([*[str(p) for p in Path('tests').glob('test_locomotion*.py')],'-q','-p','no:cacheprovider','--basetemp',str(base)]))
'@ | C:\projects\pixel-active-inference\.venv\Scripts\python.exe -
```

### Historical authored-likelihood measurements

**Fresh gate (v3, seed 54006, calibration seed 64000): 40% accuracy** (balanced 43%) on 153 scored
falls. Own gait error is recognised 96%, push 19%, slip 13%: the cause model still attributes most
pushed and slipped falls to the body's own error. It needs better evidence (e.g. sole forces).

Development check (v1):

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
