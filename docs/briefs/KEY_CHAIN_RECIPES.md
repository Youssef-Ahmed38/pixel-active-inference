# Brief: the key chain as a learned recipe, not a script (Stage 4, audit item 5)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (active inference: a learned
generative model, surprise, Bayesian hidden-cause inference, curiosity, episodic memory and
recipes, learned rather than scripted, honest evaluation). This task is Stage 4: getting through a
locked door whose key is hidden (e.g. in a drawer).

## 1. Read first

| File | What it is |
|---|---|
| `docs/AUDIT_OUTSIDE_BUILDS.md` section 5 | The audit of Stage 4: what is scripted and must become learned |
| `docs/DISCOVERY.md`, `docs/INTEGRATED_KEY_CHAIN.md` | The discovery agent (learns by experiment how a door opens) and the integrated key chain |
| `pai/discovery/agent.py`, `belief.py`, `recipes.py`, `mock.py`, `physics.py` | Discovery agent: belief over causal rules, interventions chosen by expected information gain, recipes |
| `pai/maze/key_chain.py` | The current key chain: a scripted state machine (lines ~251-360) |
| `scripts/key_chain_eval.py`, `scripts/key_chain_stage4_eval.py`, `tests/test_key_chain.py` | Current evaluation and tests |

## 2. The problem (from the audit)

The chain find_drawer -> pick -> find_door -> insert -> turn -> pass is **coded as stages**, with
authored priors: keys are "newly revealed objects with shape `elongated`", inserts are tried
smallest-first, "both turns stalled => incompatible key" is a rule, the fallback probes are sorted
by authored shape names, fixture role IDs (`fixture.pid["drawer"/"door"/...]`) tell the agent which
parts belong to the door, and the script sets `scene.door_open = True`. The failed-opening
explanation score covers one cause class with the first key forced wrong.

## 3. The task

1. **Learn the chain as a recipe.** The agent should discover the sub-goals and their order from its
   own experiments (the discovery agent's causal-rule belief and expected information gain) and
   store a successful sequence as a recipe with preconditions (e.g. "door openable requires key in
   keyhole and turned"; "key reachable requires drawer open"), then reuse and repair it in new
   layouts. Remove the authored priors listed above; anything still given must be stated.
2. **No role IDs or script-set world state.** The agent sees parts and objects without labels saying
   which part belongs to which fixture, and the door opens only through the simulator's own
   dynamics / mock rules, never because the agent's code sets it.
3. **Explanations of failed openings** scored against the event log over all cause classes (locked,
   latched, wrong key, key not turned, blocked), without forcing any outcome.
4. **Honest evaluation.** Development seeds for all tuning; final seeds never used for Stage 4 before
   (check `results/key_chain_*`, `results/discovery_*` for used seeds), run once with frozen
   parameters. Report: success rate with Wilson 95% intervals, actions to success, first-time
   (cold) vs with-recipe (warm) on unseen layouts and a new door type, against the current scripted
   chain and a no-recipe ablation on the same layouts. Mock world first; physics if time allows.

## 4. Rules

- Work in the main checkout on `main`. **Do not create branches, stash, reset or commit**; Claude
  reviews and commits.
- Touch only: `pai/maze/key_chain.py` (or a new `pai/discovery/key_chain_recipe.py`), `pai/discovery/*`,
  new `scripts/key_chain_v2*.py`, `tests/test_key_chain*.py`, `tests/test_discovery*.py`, a new
  section in `docs/DISCOVERY.md`, new `results/key_chain_v2_*` files.
- **Do not touch anything for walking or the maze walker** (`pai/locomotion/*`, `pai/maze/walker.py`,
  `pai/maze/env.py`, `scripts/locomotion_*`, `scripts/maze_*`, `configs/locomotion*`): Claude is
  training the humanoid at the same time. The laptop GPU is busy with that training; this task
  should run on the CPU.
- Keep passing: `tests/test_key_chain*.py`, `tests/test_discovery*.py`.

## 5. Deliverables

Code and fast unit tests; the new section in `docs/DISCOVERY.md` (design, learned vs given, results
table with intervals, examples of a learned recipe and of explanations next to the event log,
failures, seeds, exact commands); `results/key_chain_v2_*.json` with per-episode records. Finish with
a short report: files, commands, seeds, numbers, what is not done.
