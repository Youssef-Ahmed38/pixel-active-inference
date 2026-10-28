# Brief: a hierarchical thinker that learns through its own perception (go/no-go)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns many skills in simulation with the same principles (active inference: generative models,
prediction error / surprise, hidden-cause inference, expected free energy, curiosity), learned
rather than scripted, evaluated honestly. The paper's planned thesis is:

> *A brain-inspired hierarchical thinker learns the world through its own perception: lower levels
> predict the senses quickly, higher levels infer hidden causes and plan slowly, and surprise drives
> what it learns.*

This task builds that thinker for the **robot arm (Stage 1)** and decides, with a fair test, whether
the hierarchy is actually better than a flat model. **A clear negative answer is a valid, valuable
result**: report it honestly; do not tune until it looks positive.

## 1. Read first

| File | What it is |
|---|---|
| `pai/thinker/hrm.py`, `results/thinker/real_c1000.md` | the current thinker (HRM, Wang et al. 2025): trained to imitate the Bayesian cause teacher; 96.0% agreement vs a same-size transformer 95.8%, mean 1.06 thinking segments: **the hierarchy brings nothing there**. This motivates the new design. |
| `pai/causes/inference.py` | the Bayesian cause inference (causes: none, push, heavier_object, slippery_object, camera_shift) over 12 prediction-error channels (gripper xyz, force xyz, object xyz, scene xyz) |
| `pai/agents/slice_agent.py`, `scripts/slice_eval.py`, `pai/envs/*`, `pai/envs/events.py` | the arm agent, its episodes with the five disturbance conditions, and the simulator's event log (scoring only) |
| `results/slice_evidence.npz`, `results/pixel_evidence.npz` | 100 episodes each of per-step prediction errors `z` with labels (scoring only) |
| `docs/AUDIT_OUTSIDE_BUILDS.md` | what earlier work was criticised for |

## 2. What to build

### 2.1 Data from the arm's own senses
Collect new arm episodes (new seeds, e.g. 90000+ for training, 91000+ development, **92000+ final**,
never mixed) under the five conditions, saving **per-step raw sensory streams** that the agent itself
has (not simulator internals beyond what its sensors give): gripper pose, contact/force sensing,
perceived object pose (from its own perception as the slice agent uses it), scene/camera features,
the commanded action, holding flag, subgoal phase. Save the event log separately, for scoring only.
Aim for at least 200 episodes per condition for training (more if cheap), 40 per condition for
development and 40 per condition for the final.

### 2.2 The hierarchical thinker (`pai/thinker/predictive.py`, new)
A three-level predictive hierarchy, trained **self-supervised from the sensory stream only, with no
cause labels**:

- **Level 1 (fast, every step):** predicts the next sensory vector from the current one, the action
  and the top-down signal from level 2. Its precision-weighted prediction error is the surprise.
- **Level 2 (slower: updates every k steps, or when level-1 error accumulates):** a latent cause /
  context state. It is **inferred by iterative minimisation of the prediction error** (a few gradient
  or amortised-plus-refinement steps per update: this is the "thinking"; allow more iterations when
  the error is large, and record how many were used). It should come to represent the hidden causes
  of the arm's world (push, heavier, slippery, camera moved, none) without being told them, e.g. as a
  learned discrete code (mixture / VQ / categorical latent) plus continuous parameters.
- **Level 3 (slowest):** a per-episode context / goal state that conditions level 2 (it may be simple;
  it must exist so the timescale hierarchy can be measured).

Keep it small enough to train on the laptop GPU (RTX 5060, 8 GB) in under ~1 hour per seed.

### 2.3 The fair comparison (the go/no-go)
Same data, same parameter budget (within 10%), same training steps, **3 training seeds each**:
1. **Hierarchical** (2.2).
2. **Flat:** one recurrent world model (e.g. GRU) of the same size predicting the next sensory vector,
   with its hidden state used as the "cause" representation.
3. **Hierarchical without iterative inference** (one amortised pass only: does the thinking matter?).
4. Reference: the existing Bayesian cause inference and the HRM imitation student (they use labels or
   the teacher; report them as references, not as competitors).

Measures, on the **final** seeds, mean ± std over training seeds, with Wilson intervals where counts:
- **Cause discovery without labels:** cluster the level-2 (or flat hidden) representation per episode
  (e.g. the latent at the end of the disturbance window, or a learned discrete code); map clusters to
  true causes on the development split only (Hungarian), then score accuracy and NMI on the final split.
  Also a linear probe accuracy (probe trained on development, scored on final).
- **Prediction:** next-step prediction error overall, and in the 20 steps after a disturbance onset
  (does inferring the cause make the world predictable again faster?).
- **Adaptation:** how many steps after onset until prediction error returns to the clean level.

### 2.4 Brain-like signatures (measured, not claimed)
- **Timescale hierarchy:** autocorrelation time of each level's state; higher levels should be slower
  (cf. Murray et al. 2014, Nature Neuroscience). Report the numbers per level.
- **Prediction-error dynamics:** level-1 error spikes at an unexpected disturbance and attenuates as the
  same disturbance repeats or persists (repetition-suppression / mismatch-negativity analogue).
- **Thinking scales with difficulty:** inference iterations vs surprise and vs condition; and whether
  more iterations improve cause discovery.
- Optional if time allows: train level 1 with **local predictive-coding updates** (each layer learns
  from its own prediction error) and compare with backprop.

## 3. Rules

- **No ground truth in the agent:** labels, onsets and the event log are used only for scoring (and the
  cluster-to-cause mapping on the development split). The thinker learns from its own senses.
- **Fresh evaluation:** final numbers only on the final seeds, frozen models, exact commands; keep
  every negative result and failed variant in the report.
- **Stay in your files:** `pai/thinker/predictive.py` (new) and new modules under `pai/thinker/`, new
  `scripts/thinker_hier_*.py`, `tests/test_thinker_hier*.py`, a new `docs/THINKER_HIERARCHY.md`, new
  `results/thinker_hier_*` files. You may add an option to the episode collection that saves raw
  sensory streams, without changing default behaviour. Do not change other stages.
- **Keep passing:** `tests/test_slice.py`, `tests/test_cognition.py`, `tests/test_thinker*.py`.
- Results files must not contain absolute local paths (paths relative to the repository).
- Training runs on the laptop GPU; do not use cloud compute.

## 4. Deliverables

Code with fast unit tests; `docs/THINKER_HIERARCHY.md` (design, what is learned vs given, data and
seeds, the go/no-go table, brain-signature tables and one figure each, failures and negative results,
exact reproduce commands, and a one-paragraph verdict: does the hierarchy beat the flat model, by how
much, with what uncertainty); `results/thinker_hier_*.json` with per-episode and per-seed records.
