# Audit: work built in the separate session (commit db5bf0e and its three predecessors)

Date: 13 October 2026. Read-only audit; the only file written is this one.
Scope: `pai/cognition`, `pai/maze` (with walking), the key chain, contact repair,
`pai/chess` (Stage 5), `pai/language` (Stage 6), `pai/bike` (Stage 7), with their
scripts, tests, docs and result files. Judged against each build's own stated gates,
the roadmap of that time (main target: one agent, the same brain across stages) and the stage
build plan of that time (both since replaced; earlier versions are in the repository history).

Every number below was either read from a result file during this audit or produced
by a test command run during this audit. No simulator evaluation was re-run.

## 1. Test runs (this audit)

Command per group: `.venv/Scripts/python -m pytest -q -p no:cacheprovider tests/test_<group>*.py`

| Group | Files | Result |
|---|---|---|
| cognition | `test_cognition.py` | 5 passed |
| cognitive extensions | `test_cognitive_extensions.py` | 7 passed |
| contact repair | `test_contact_repair.py` | 2 passed |
| key chain | `test_key_chain.py` | 9 passed |
| maze | `test_maze.py`, `test_maze_advanced.py` | 12 passed |
| walking + cross-domain thinker | `test_next_builds.py`, `test_cross_domain_thinker.py` | 5 passed |
| chess | `test_chess_*.py` (8 files) | 124 passed (221 s) |
| language | `test_language*.py` (3 files) | 27 passed |
| bike | `test_bike*.py` (3 files) | 19 passed |
| **Total** | | **210 passed, 0 failed** |

Note on test strength: the two contact-repair tests are unit tests of the release
height and the clearance order (`tests/test_contact_repair.py:11,29`); no test runs
the physical insertion reseating. The walking test checks one injected fall.

## 2. Cross-cutting findings (apply to several stages)

1. **One brain is not yet true outside the arm and maze.** What chess, language and
   the bike reuse from the shared agent is the dataclasses in
   `pai/cognition/contracts.py` and the generic loop `CognitiveController.run`
   (`pai/cognition/controller.py:16-60`). Import map from this audit:
   - `pai/chess` imports `pai.cognition.contracts` and `pai.thinker.hrm` (HRM only in
     a rejected value candidate);
   - `pai/language` imports contracts and the controller;
   - `pai/bike` imports contracts and the controller.

   None of them uses the Stage 1 world model (`pai/models`), the expected-free-energy
   objective (`pai/planning/objective.py`), surprise-gated memory (`pai/memory`),
   Bayesian cause inference (`pai/causes`), recipes, or curiosity. Each is a separate
   hand-designed system in the shared vocabulary.
2. **The EFE objective is used in name only in the maze.**
   - `pai/maze/agent.py:108-109` calls `expected_free_energy(0.0, information_rate)`
     with zero pragmatic term. The score is then a monotone function of
     (1 + 0.03·unseen)/route length, which is a nearest-frontier rule.
   - `results/maze_transfer.json` selected `gain_weight = 0.0` on the training mazes:
     the epistemic bonus did not help there.
   - The explorer therefore *is* the frontier-exploration baseline that the Stage 3
     gate asks it to beat.
3. **Cause inference is hand-written in the maze and the key chain.**
   - `MazeCauseMonitor` uses constant log-evidence `{none:-8, passage_blocked:0, unknown:-4}`
     (`pai/maze/advanced.py:23-41`), so the posterior never depends on data. It is the
     rule "collision on a believed-free cell → passage blocked".
   - The visual probe likelihoods are also authored (`pai/maze/visual.py:139-145`).
   - The HRM cause thinker's maze evidence is two hand-set flags at value 3.0, and its
     labels come from simulator truth (`pai/cognition/thinker_data.py:71-108`).
     `results/cross_domain_thinker.json` shows `maze_transfer` 39/39 = `maze_scratch` 39/39,
     so the task is trivial and shows no arm→maze transfer.
   - In `results/fresh_maze_eval_20.json` the online thinker and the baseline are
     identical (mean steps 617.8 both), so the thinker has no measured effect.
4. **Simulator truth enters the arm belief.** `ArmDomainAdapter.observe` passes
   `env` `"predicates"` into `StateBelief.relations` (`pai/cognition/adapters.py:126`).
   Those relations are computed by `compute_predicates` from simulator state
   (`pai/envs/tabletop.py:347`). This breaks the contract stated in
   `pai/cognition/contracts.py:4-5`.

   The leak drives:
   - goal success (`StateBelief.satisfies`, controller `:24,49`);
   - contact repair (`stack_clearance_repairs`, `pai/cognition/contact_repair.py:30-55`);
   - every language answer and narration (`pai/language/session.py:38-43,76-84`).

   Language attributes come from the privileged token slots `kind one-hot | colour RGB`
   (`pai/world/entities.py:8,55-56`). `LearnedAttributes` (`pai/language/observation.py:19-40`)
   therefore "learns" prototypes of ground-truth one-hots.

   The docs label this "privileged state", which BUILD_PLAN allows as a first step.
   The risk is that downstream results (language Q&A, narration-vs-log) are then
   circular: they are scored against the same truth they were given.
5. **Seed hygiene problems** (details in each stage table):
   - maze seeds 40000–40005 were used to debug a failure, then reused for the
     walking and learned-vision evaluations;
   - arm seed 123 was a contact-repair development seed and then a language gate seed;
   - key-chain seed 20 was a body-skill development seed and then a Stage 4 held-out seed.

## 3. Shared cognition layer (`pai/cognition`)

| Item | Status | Evidence |
|---|---|---|
| Portable contracts (`StateBelief`, `SkillCommand`, `SkillOutcome`, `SurpriseEvidence`, `CognitiveEpisode`) | complete | `contracts.py:14-96` |
| Domain adapters arm/maze/discovery | complete; arm adapter leaks truth (see §2.4) | `adapters.py:71-158`, `discovery.py:50-104` |
| Shared controller loop + sequence/repair passes | complete | `controller.py:12-90` |
| Shared memory | partial: append-only list + exact-match recall; no embedding, no surprise gating, no consolidation; not used by chess/bike | `memory.py:12-38` |
| Shared cause posterior | partial: generic softmax only; likelihoods remain per-domain and hand-set in maze | `causes.py:10-36`, `maze/advanced.py:23-41` |
| Shared HRM cause thinker | built; trivial maze task; no task-success effect | `thinker.py`, `results/cross_domain_thinker.json`, `results/fresh_maze_eval_20.json` |
| Causal experiment planner | built (information gain); likelihoods authored by caller | `experiments.py:21-50`, `maze/visual.py:139-145` |

Design conformance: this is a good portable vocabulary. It is a shared *interface*,
not yet a shared *brain*. Planning, world model and cause likelihoods remain per domain.

## 4. Stage 3: maze (+ walking) — `pai/maze`

| Gate / item (ROADMAP §Stage 3, BUILD_PLAN Stage 3) | Status | Evidence |
|---|---|---|
| Procedural maze, landmarks, egocentric camera | complete (grid world, RGB tile image) | `env.py`, `visual.py` |
| Place memory linked into a topological map | complete (explicit cells/edges), not learned embeddings | `agent.py:19-64` |
| Coverage beats **random** | complete: 65.2% vs 10.4% (seed 0), 64.1% vs 9.1% (seed 1000) | `results/maze_eval.json`, `results/maze_eval_holdout.json` (matched doc) |
| Coverage beats **frontier-exploration** baseline | **missing**: no frontier baseline was run; the explorer is itself a nearest-frontier rule (§2.2) | `agent.py:98-117`, `results/maze_transfer.json` (gain weight 0.0) |
| Finds goals faster in new mazes after other mazes (transfer) | **failed / negative**: transferred 64.2% vs fixed 65.5%, goal 25/30 vs 24/30 | `results/maze_transfer.json` (matched doc) |
| Runs on long corridors, walks near turns (reuses Stage 2) | **missing**: no speed choice exists | `walking.py`, `mobility.py` |
| Never re-learns walking / regression | n/a: no learned gait exists | `walking.py:1-5` (own docstring) |
| HRM thinker plans routes | **missing**: HRM is only a cause classifier | `cognition/thinker.py` |
| Full completion gate (own) | complete: 30/30 + 30/30 full coverage and goal, 600-move cap | `results/maze_complete*.json` (matched: coverage 1.0, goal rate 1.0) |
| Advanced visual/noisy/disturbed (own) | complete: 12/12 goal and full coverage; block and shift detection 1.0; pose error 0 vs 5.73 cells raw odometry | `results/maze_advanced.json` (matched) |
| Walking body | **scripted**: the fall happens at a fixed step, and recovery is incremented in the same branch (`walking.py:195-203`). The 10/10 recovery rate holds by construction. No dynamics, no learned gait | `results/walking_maze_eval.json` |

Leakage / scripting:
- The policy does not read `grid` or `truth()`; the only reads are L0/simulator bodies
  (`mobility.py:42`, `walking.py:60`) and scoring.
- Visual colour decoding is hand-specified. The learned tile model is a supervised
  Gaussian per colour (`learned_vision.py:296-317`).

Numbers vs files: the doc numbers for `maze_eval`, `maze_eval_holdout`, `maze_complete*`,
`maze_advanced` and `maze_transfer` all match the files.

Fresh seeds:
- **Not fresh: `walking_maze_eval.json` and `walking_learned_vision_eval.json`** use
  seeds 40000–40009.
  - 40005 was debugged (`fresh_maze_debug_40005.json`, `fresh_maze_recheck_40005.json`).
  - The explorer was changed until 40000–40005 passed (`docs/EXTENSIONS.md:29-34`).
- The same issue affects the 6/6 in `fresh_maze_eval.json`.
- In the 20-seed run, only 40006–40019 are clean.

## 5. Stage 4: key chain + contact repair — `pai/maze/key_chain.py`, `pai/cognition/contact_repair.py`

| Gate / item | Status | Evidence |
|---|---|---|
| Maze + drawer + key + locked door, one map across goal switch | complete as a composition across two simulators | `key_chain.py:233-370` |
| Solves unseen layouts needing key chains | partial: symbolic 6/6 (15×15), physical 4/4 (one door type), held-out symbolic 12/12 cold and 12/12 warm | `results/key_chain_symbolic_0_5.json`, `key_chain_physical_0_3.json`, `key_chain_stage4_eval.json` (all matched) |
| Explanation of failed openings > 70% (vs event log) | partial / weak: only one cause class (incompatible key) is scored, and the first key is forced wrong (`scripts/key_chain_stage4_eval.py:27-33`). "Accuracy" is report precision on that one class (`:57-59`). Locked/deadbolt/stuck explanations are not scored | `key_chain_stage4_eval.json` (12/12) |
| One-shot reuse of a new door type | partial: recipe from 1 training episode reduced mean fixture actions 34.9 → 29.7, faster on 11/12. Mock world only | `key_chain_stage4_eval.json` (matched) |
| Faster than without Stage 1 recipes | partial: that recipe is a key-chain recipe, not a Stage 1 arm recipe | same |
| Humanoid/mobile manipulator with hands | missing (fixture simulator is separate; the key "stays in the grasp" while the maze body moves) | `docs/INTEGRATED_KEY_CHAIN.md` |
| Contact repair: arm tower | complete on 3 seeds: 0/3 → 3/3 | `results/contact_repair_arm_ablation.json` (matched) |
| Contact repair: key reseating | **contradictory**: ablation 0/2 → 2/2, but `results/contact_repair_key_final.json` (contact_repair true) has success 0.5 (1/2). The doc says the 1/2 came from a removed tighter guard; the file has no code hash to show which code produced it | `contact_repair_key_*.json` |

Design conformance (the main issue for Stage 4): the chain is a **scripted state machine**,
not a recipe the agent discovered (`key_chain.py:251-360`). The stages
find_drawer → pick → find_door → insert → turn → discovery → pass are coded.
Other parts are authored too:
- **Hand-written priors:**
  - keys are "newly revealed objects with shape `elongated`" (`:266-268`);
  - inserts are tried smallest-first (`:292-293`);
  - "both turns stalled ⇒ incompatible key" is a rule (`:323-327`);
  - the fallback `_repair_opening` sorts probes by authored shape names
    `bar/round/ball/wing` (`:211-225`).
- **Role IDs:** fixture role IDs from `fixture.pid["drawer"/"door"/"handle"/...]` scope
  reachable actions (`:244-247`). This is documented as scene construction, but it
  tells the agent which parts belong to the door.
- **Door coupling:** `scene.door_open = True` is set by the script after an observed
  goal feature (`:353`).

Contact repair (`stack_clearance_repairs`, insertion retries) is explicit engineering
in L0 or the controller. That is acceptable for L0, but it is not learned. It also
relies on privileged relations (§2.4).

Seeds:
- Stage 4 held-out seeds 20–23 overlap the seed used to develop the ball-knob body fix
  (`results/key_chain_physical_ball_push_seed20.json`, `docs/INTEGRATED_KEY_CHAIN.md:67-72`).
- Contact-repair arm seed 123 was the failing development seed. The gate seeds 126–128
  are clean.

## 6. Stage 5: chess — `pai/chess`

| Gate / item | Status | Evidence |
|---|---|---|
| Teacher; rules never coded into the agent | complete. `rules.py` inference has no chess-library import (grep). Library only in teacher/training/eval/play harness. Geometry/token-transfer priors are declared | `docs/CHESS_RULES.md`, `pai/chess/rules.py` |
| Legal-move accuracy > 99% on unseen positions | **complete**: seed 108: 566 positions, precision/recall 1.0, strict suite passed; seed 109: 562 positions, same | `results/chess_rules.json`, `results/chess_rules_seed109.json` (matched doc) |
| Complete state transitions, endings, draw claims | complete (own gates) | `docs/CHESS_STATE.md`, `CHESS_EVALUATION.md` |
| HRM thinker searches ahead | **missing**: search is alpha-beta/MCTS (`search.py`, `mcts.py`). HRM is only a value candidate that did not beat the baseline (`results/chess_hrm.json`, val MAE 387 cp); no learned halting | |
| Self-play improvement | **failed**: candidate rejected (validation MAE 392.9 → 397.5 cp; all 4 self-play games truncated) | `results/chess_selfplay.json` (matched) |
| Elo estimate vs several Stockfish levels | **unmet**: 12/12 losses at UCI_Elo 1320/1800/2400 × 32/128 nodes; one opening, so the colours are paired. No finite Elo | `results/chess_matches_completed.json` (re-counted: 12 complete, 6×0-1 as White, 6×1-0 as Black) |
| Rating improves with thinking time | **unmet**: 32 and 128 nodes both 0 points; MCTS 128 nodes avoided 1/4 poisoned-rook traps | `results/chess_mcts.json` (matched) |
| Contextual value candidate | partial: MAE 337.5 → 315.4 cp (seed 703) but regressed on tactics; not adopted | `results/chess_context_value.json` (matched) |
| Physical chess with the arm | missing (optional) | |

Design conformance:
- Rule learning is honest teacher-based induction, but with trees and a lookup table,
  not the Stage 1 causal world model.
- Value is distilled from Stockfish, which is acceptable as teaching.
- Planning is classical search, not the EFE objective.
- No surprise or cause inference, and no curiosity.

It is a well-tested but separate chess engine in the shared contract.
Leakage: `play.py:51-60` takes the agent's moves from `planner.search` only. The
teacher executes them and does not mask them.

Seeds: the rule gate on 108/109 is declared fresh, and 100–107 are marked as
development. Value seeds (700/701/703) are separated.

## 7. Stage 6: language — `pai/language`

| Gate / item | Status | Evidence |
|---|---|---|
| Conversation simulator about scene events | partial: teacher templates (`teacher.py:19-38`). Recorded-scene rows from arm episodes (`conversations.py:24-71`). No people talking about scenes with referential ambiguity; no Stage 3/4 scenes; no LLM speakers | |
| Word grounding by cross-situational learning | **design gap**: `GroundedLanguage.fit` receives `(text, exact meaning frame)` pairs (`grounding.py:37-45`). Each word's meaning is the intersection of the labelled frames, with no competing scene referents. This is supervised parsing, not learning from situations | `results/language_stage6.json`: 20 words learned |
| Word-meaning accuracy | complete on the controlled grammar: 456/456 held-out parses and roundtrips, 84/84 goals per language | `results/language_stage6.json` (matched) |
| Instructions with unseen word combinations | partial: held-out combinations reuse all words; templates match exact token length (`grounding.py:72`). Physical: 4/4 sequences, 8/8 goals, but English and Arabic are paired with identical planner RNG and steps (148, 117), so these are 2 physical runs | `results/language_arm.json` (matched) |
| Self-descriptions match the event log | partial / circular: narration reads relation gains from beliefs that contain simulator predicates (§2.4). Recorded-scene audit 118/118 per language | `results/language_conversations.json` (matched) |
| Describes its own explanations ("…because it slipped") | **missing**: hypothesis sentences exist (483/483 grammar), but 0 recorded hypotheses and 0 verified causes | `results/language_hypotheses.json` (matched) |
| Speech | missing (declared) | |

Seeds: arm seeds 123/124 are not fresh. 123 was the contact-repair development
failure (`docs/CONTACT_REPAIR.md`, "previously failing seed 123"). Scene layouts
130–133 are clean.

## 8. Stage 7: bicycle — `pai/bike`

| Gate / item | Status | Evidence |
|---|---|---|
| MuJoCo bicycle, simplified rider (fallback) | complete (fallback): torque/steer/lean actuators; no pedals or humanoid | `env.py` |
| Expert demonstrations rendered as video | complete (video recorded) | `docs/media/bike/expert.mp4` |
| Learning by **watching** (inverse dynamics with own body model) | **partial / privileged**: "watching" reads the expert's exact `qpos/qvel` from the simulator (`observation_learning.py:61-65`). Inverse dynamics uses the true model and hard-coded servo gains 100/12/15 (`:18-31`). No video is used to learn actions | `results/bike_observation.json`: student 10/10, uncontrolled 1/10 (matched) |
| Rides a set distance | complete (10 m from a moving start): classical 10/10, student 10/10; diverse 17/20 | `bike_stage7_holdout.json`, `bike_observation_diverse.json` (matched) |
| Fewer practice attempts than an agent that did not watch | partial / positive: 3 optimizer replicates on the same 20 layouts. Observed policy 20/20, 20/20, 19/20 vs scratch 2/20, 1/20, 0/20. Scratch never solved practice in 24 updates | `results/bike_replicates.json` (matched) |
| Fewer attempts than an agent without Stage 2 | **missing**: there is no Stage 2 learned controller | |
| Practice explains falls with cause inference | **missing**: practice is random-search hill-climbing (`practice.py:36-58`). No surprise or cause code in `pai/bike` (grep) | |
| Starts from rest, pushes, pedals, full rider | missing (declared) | |
| RGB attitude + hybrid control (own extension) | complete for the final candidate: 20/20 vs sensor 20/20 on 1100–1119. The earlier candidates failed (12/20 vs 13/20; 0/20; 19/20) and are preserved | `bike_vision_*holdout20.json` (all matched) |

Design conformance:
- The learned student is a 7×3 linear map fitted to inferred actions.
- The "agent" is not the shared brain: no world model, no EFE, no causes.
- The visual controller wraps the hand-tuned `BalanceController` by default
  (`vision_control.py:12`) or the practiced linear policy.
- Speed and camera tracking come from simulator sensors and root position.

Seeds: the sequential holdouts are each new. The final visual candidate is the fourth
iteration after failures, which is honest but is model selection across holdouts.

## 9. Prioritised list: what to build or fix next

1. **Remove truth from the arm belief** (`pai/cognition/adapters.py:126`): derive relations
   from perceived tokens (the existing `pai/envs/predicates.py` logic applied to perceived
   positions) or a learned relation head. Then re-score:
   - language Q&A and narration;
   - contact-repair goals;
   - controller success.

   Without this, the Stage 6 "matches the event log" gate is circular.
2. **Build the Stage 2 learned walker** (MuJoCo humanoid, PPO on the free-energy
   objective, fall-cause inference). The maze "walking" result
   (`walking.py:195-203`, recovery by construction) must not be cited as locomotion.
   Stage 3's run/walk choice and Stage 7's "without Stage 2" transfer gate both depend
   on it.
3. **Make the maze explorer an actual EFE agent and add the required frontier
   baseline.** The epistemic term should come from a learned map/observation model's
   uncertainty, not a geometric count with weight 0. Run nearest-frontier and
   information-gain-frontier baselines on fresh seeds (≥ 40020), and re-run the
   walking/learned-vision evaluations on seeds never debugged.
4. **Learn cause likelihoods instead of authoring them.** Replace constant
   log-evidence (`maze/advanced.py:23-41`) and authored probe likelihoods
   (`maze/visual.py:139-145`) with likelihoods fitted from experience. Give the HRM
   thinker raw residual evidence in the maze, not 3.0 flags
   (`thinker_data.py:92-98`), and add confusable causes so transfer can be measured
   (currently transfer = scratch = 100%).
5. **Stage 4: replace the scripted key-chain state machine** (`key_chain.py:251-360`) with
   recipe/precondition learning: "door stalls ⇒ hypothesis locked ⇒ need object". Drop
   the `elongated` and shape-name priors (`:266-268`, `:211-225`). Score explanation
   accuracy over several failure causes (locked, deadbolt, wrong key, stuck) against
   the event log, on seeds not used for development (≠ 20). Run the held-out
   door types physically.
6. **Stage 6: real cross-situational learning.** Feed utterances with the *scene*
   (several candidate relations and entities) instead of the exact frame
   (`grounding.py:37-45`). Use Stage 3/4 scenes, and words held out in some
   compositions. Connect explanations to verified cause evidence (currently 0).
   Use fresh arm seeds and run the two languages independently.
7. **Stage 5: connect chess to the shared thinker.** Use HRM-guided search with learned
   halting and measure the thinking-time curve. Use several openings and more games per
   level for a finite Elo. Keep the rule gate frozen.
8. **Stage 7: watch from pixels.** Extract motion from the rendered video, not
   `qpos/qvel`. Learn the inverse model rather than using hard-coded servo gains.
   Add fall-cause inference to practice, plus independent practice/eval layouts per
   replicate.
9. **Housekeeping:**
   - record a code hash in every result file (resolves the contact-repair key 1/2 vs
     2/2 ambiguity);
   - add a physics test for insertion reseating;
   - keep a seed registry listing development seeds per stage, so "fresh" claims can
     be checked mechanically.
