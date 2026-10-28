# Brief: learned cause likelihoods for the arm (Stage 1, audit item 4)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (active inference: a learned
generative model, surprise, Bayesian hidden-cause inference, learned rather than scripted, honest
evaluation). This task is Stage 1: the robot arm's inference of hidden causes (none, push, heavier
object, slippery object, camera shift).

## 1. Read first

| File | What it is |
|---|---|
| `pai/causes/inference.py` | The arm's cause inference: calibrated residual windows, one hand-designed residual model per cause (onset search, parameters, BIC), posterior |
| `pai/agents/slice_agent.py`, `scripts/slice_eval.py` | The arm agent and its evaluation with the five conditions; `results/slice_v7/` (exact state) and `results/pixel_v3/` (from camera) are the current results |
| `pai/locomotion/fall_causes.py`, `scripts/locomotion_fall_causes.py`, `docs/LOCOMOTION_STAGE2.md` "Fall causes" | The walking analogue already done: features from the robot's own sensor traces, per-cause likelihoods learned from labelled development episodes, an unknown catch-all, held-out-seed selection, final on new seeds against the old method on the same episodes. Reuse the pattern |
| `docs/AUDIT_OUTSIDE_BUILDS.md` item 4 | The audit: likelihoods should be learned from experience, not authored |
| `docs/LANGUAGE_STAGE6.md` (V3 section) | Language errors that come from the arm monitor: a clean episode reported as push; heavier and slippery blocks reported as push |

## 2. The task

1. **Learned likelihoods.** Keep the arm's own evidence (world-model residuals, force, perception)
   and the existing per-cause fits as features if useful, but learn the per-cause likelihoods (and
   a broad unknown catch-all) from labelled **development** episodes, like the walking fall causes.
   The disturbance labels are used only for fitting; inference sees only the arm's own senses.
2. **Target the known failures:** false pushes in clean and heavier/slippery episodes, and the
   slippery block from camera images (35%). Report per-cause recall.
3. **Both regimes:** exact object state (as `results/slice_v7`) and from camera images (as
   `results/pixel_v3`), if time allows; exact state first.
4. **Keep parameter estimates** (extra mass, camera offset, onset) where the cause is right; report
   their errors as before.
5. **Honest evaluation.** Development seeds for all fitting and selection (held-out seed
   cross-validation); the final run once with frozen parameters on seeds never used for the arm
   (check `results/slice_v*`, `results/pixel_v*`, `results/language_causes_*` for used seeds),
   scored against the old method on exactly the same episodes, with Wilson 95% intervals and a
   paired comparison. Report what is learned and what is given.

## 3. Rules

- Work in the main checkout on `main`. **Do not create branches, stash, reset or commit**; Claude
  reviews and commits.
- Touch only: `pai/causes/*` (add a new learned module rather than deleting the old one, which stays
  the baseline), new `scripts/arm_causes*.py`, `tests/test_causes*.py`, a new section in the README's
  Stage 1 part or a new `docs/ARM_CAUSES.md`, new `results/arm_causes_*` files. Results files larger
  than 5 MB: write them gzip-compressed (`.json.gz`).
- **Do not touch** walking or maze files (`pai/locomotion/*`, `pai/maze/*`, `scripts/locomotion_*`,
  `scripts/maze_*`, `configs/locomotion*`), the language files, or `pai/discovery/*`: other work is
  in progress there. The laptop GPU is shared with humanoid training; keep batch sizes modest.
- Keep passing: `tests/test_causes*.py`, `tests/test_slice.py`, `tests/test_cognition.py`,
  `tests/test_language*.py`.

## 4. Deliverables

Code and fast unit tests; the documentation (design, learned vs given, results tables with
intervals per cause and regime, failures, seeds, exact commands); `results/arm_causes_*` with
per-episode records. Finish with a short report: files, commands, seeds, numbers, what is not done.
