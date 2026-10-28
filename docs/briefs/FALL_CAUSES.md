# Brief: the humanoid says why it fell (Stage 2 fall causes)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (active inference: a learned
generative model, surprise, Bayesian hidden-cause inference, learned rather than scripted, honest
evaluation). This task: make the walking humanoid infer **why it fell**.

## 1. Read first

| File | What it is |
|---|---|
| `docs/LOCOMOTION_STAGE2.md` (section "Fall causes") | Current result and what fails |
| `pai/locomotion/falls.py` | Surprise monitor, the two fall features (unexplained pelvis impulse U, stance-foot slip speed S), calibration on undisturbed falls, Bayesian comparison of gait / push / slip |
| `scripts/locomotion_eval.py` (`falls_experiment`, `test_fall_causes`) | Injects one cause per body (answer key), records falls, scores the posterior |
| `pai/locomotion/env.py`, `world_model.py` | The G1 environment (sensors, contacts) and the learned body model that predicts proprioception from the body's own motor commands |
| `pai/causes/inference.py` | The arm's cause inference (residual windows, onset search, BIC), the pattern to follow |
| `docs/AUDIT_OUTSIDE_BUILDS.md` item 4 | Audit: cause likelihoods should be learned, not authored |

## 2. The problem (measured)

Released controller v3 (`runs/locomotion/policy_v3/policy.pt`, config `configs/locomotion_v3c.yaml`).
Fresh gate (seed 54006, calibration seed 64000): accuracy **40%** (balanced 43%) on 153 falls.
Own gait error is recognised 96%, push 19%, slip 13%. Median features (U, S): gait [3.50, 1.00],
push [3.97, ...]: the push signal is barely above the gait signal. Likely reasons: the features
summarise only the last 100 control steps (2 s) before the fall, so a push earlier than that is
outside the window; U and S are scalar summaries with authored likelihood shapes; foot contact
forces are not used.

## 3. The task

1. **Better evidence from the agent's own senses**: e.g. the time course of the body-model residual
   (onset search over the whole episode since the last stable walking, as the arm does), sole or
   foot contact forces and their distribution between feet, stance-foot sliding while loaded. Only
   signals the robot itself senses; nothing from the simulator's push forces, friction values or
   event log.
2. **Learned likelihoods**: fit the per-cause feature distributions from labelled *development*
   falls (labels from the injection, used only for fitting, like a teacher), on development seeds,
   with an `unknown` catch-all. Report what is learned and what is given.
3. **Honest evaluation**: tune only on development seeds; the final gate is run once with frozen
   parameters on **new** seeds never used for locomotion (not 50000-50006, 51000+, 52000+, 54000+,
   60000+, 61000, 64000; e.g. falls seed 55006 with calibration seed 65000). Report accuracy,
   balanced accuracy, per-class recall with Wilson 95% intervals, the confusion matrix, and the
   old method on the same falls for comparison.

## 4. Rules

- Work in the main checkout on `main`. **Do not create branches, stash, reset or commit**; Claude
  reviews and commits.
- Touch only: `pai/locomotion/falls.py` (or a new `pai/locomotion/fall_causes.py`), the fall-cause
  functions in `scripts/locomotion_eval.py` (or a new `scripts/locomotion_fall_causes.py`),
  `tests/test_locomotion*.py`, the "Fall causes" section of `docs/LOCOMOTION_STAGE2.md`, new
  `results/locomotion/fall_causes_*.json`. Do not retrain or change the walking policy, the body
  model or the environment's physics.
- Training and evaluation run on this laptop's GPU (RTX 5060 8 GB); another job may share it, so
  keep batch sizes modest.
- Keep passing: `tests/test_locomotion*.py`.

## 5. Deliverables

Code and fast unit tests; the updated "Fall causes" section (design, learned vs given, results
table with intervals, failures, seeds, exact commands); `results/locomotion/fall_causes_*.json`
with per-fall records. Finish with a short report: files, commands, seeds, numbers, what is not done.
