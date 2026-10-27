# Brief: the agent explains why things happened, in English and Arabic (Stage 6)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (active inference: a generative
model, surprise, hidden-cause inference, memory, learned rather than scripted, honest evaluation).
This task is **causal explanations in language** for the robot-arm agent.

## 1. Read first

| File | What it is |
|---|---|
| `docs/LANGUAGE_STAGE6.md` | The language stage so far: learned English/Arabic lexicon and templates, parsing, realization, scene questions, narration, arm instructions |
| `docs/AUDIT_OUTSIDE_BUILDS.md` §7 and item 6 | Independent audit: explanations are not yet connected to cause evidence; learning is from exact frames, not scenes |
| `pai/language/*` | `grounding.py`, `teacher.py`, `session.py`, `observation.py`, `conversations.py` |
| `pai/causes/inference.py` | The agent's Bayesian cause inference (none, push, heavier_object, slippery_object, camera_shift, unknown) with posteriors and parameters (e.g. extra mass, push onset, camera offset) |
| `pai/memory/episodic.py`, `pai/memory/report.py` | Episode records and the existing English episode reports |
| `pai/agents/slice_agent.py`, `scripts/slice_eval.py` | The arm agent and its evaluation with the five disturbances; `results/slice_v7/` has 100 episodes with event logs |
| `pai/envs/events.py` | The simulator's ground-truth event log (only for scoring) |

Since commit aa8d259 the arm's relations come from its own perception, not the simulator. Language
results computed before that used leaked relations and must be re-scored.

## 2. The task

1. **Explanations from the agent's own evidence.** After an episode, the agent says in English
   and in Arabic what happened and **why**: the cause it inferred, with its own estimates (e.g. "the
   block was about 0.4 kg heavier than it looked", "my camera moved about 4 cm to the left, so I
   recalibrated"), what it changed in response, and whether the goal was reached. The content must
   come only from the agent's beliefs (cause posterior and parameters, its episode record), never
   from the simulator's event log. When its evidence is weak it must say so (abstain or hedge).
2. **Why and what-if questions.** Answer held-out questions about an episode in both languages:
   "why did you drop the block?", "what would have happened without the push?" (answered from the
   agent's own counterfactual machinery, `pai/causes/credit.py`, where available; otherwise abstain).
3. **Learn the cause words, don't hard-code them.** Words and phrases for causes are taught by a
   teacher from paired observations (episodes with their causes and the teacher's sentences), in
   the same way the existing lexicon is induced. State clearly what is given (e.g. the teacher's
   dictionaries) and what is learned. The two languages must be learned independently.
4. **Instructions under disturbance.** Instructions given in either language are executed by the arm
   under the five disturbance conditions; the agent reports what went wrong. Compare the success to
   the arm's own success rate in the same conditions (no language).

## 3. Rules

- **No ground truth in the agent's words.** The simulator's event log and disturbance parameters
  are used only to score explanations, never to produce them.
- **Fresh evaluation.** New episodes on seeds never used before (e.g. `scripts/slice_eval.py
  --episode-seed 60000`); state development vs final seeds.
- **Scoring:** explanation accuracy against the event log (cause correct, parameter within a stated
  tolerance, correct abstention), per language; question answering on held-out questions;
  instruction success vs the arm's own rate.
- **Stay in your files:** `pai/language/*`, new `scripts/language_*.py`, `tests/test_language*.py`,
  `docs/LANGUAGE_STAGE6.md` (add a section; keep the existing content), new `results/language_*`
  files. Read anything else; do not change other stages.
- **Keep passing:** `tests/test_language*.py`, `tests/test_slice.py`, `tests/test_cognition.py`.

## 4. Deliverables

Code and fast unit tests; the new section in `docs/LANGUAGE_STAGE6.md` (design, learned vs given,
results tables per language, example explanations in both languages next to the ground truth,
failures, seeds, reproduce commands); `results/language_causes_*.json` with per-episode records.
