# Roadmap: an active-inference agent that understands, experiments, and explains

## Main target: one agent model that learns like a human

The target is **the agent model**, not any single robot or task. One brain, the same
architecture and the same learning principles, learns a curriculum of abilities the way a child
does. It explores its own body, then its surroundings, then objects, then games and language,
and it reuses what it already knows at each step.

| Stage | The agent learns to... | How it learns | Reuses |
|---|---|---|---|
| **1** | manipulate objects with a robot arm | experiment, explain causes, store recipes | - |
| **2** | **walk and run** (humanoid body) | explore its own body: motor babbling, curiosity, falling and recovering | stage 1's brain |
| **3** | **explore a maze**, walking and running | curiosity-driven exploration, memory of places, planning | stage 2's gait |
| **4** | **open doors and drawers** (inside the maze) | trial and error, cause and effect ("handle turned -> door opened") | stages 1-3 |
| **5** | **play chess**: learn it from a teacher, then play strong simulated opponents | teaching, then self-play and games against engines at rising strength | the HRM thinker |
| **6** | **learn a new language** by listening to people talk in different situations | linking heard words to the objects, relations and actions it already understands | stages 1-5 |
| **7** | **ride a bike**, after watching simulated riders | learning by observation, then practice | stage 2's balance and control |

Stages 1-7 are a **multi-year research programme**. Stage 1 is what the PhD application
shows; it is the proof that the brain works. Details for stages 2-7 are at the end of this document.

**What makes this possible:** only L0 (body control) knows which body it is in. Everything above,
from planning and goals to memory, causes, recipes and the HRM thinker, sees only objects, relations,
actions and prediction errors. A block, a maze wall, a door handle, a chess piece and a bike pedal
look the same to them. Code written for stage 1 must respect this boundary.

## The research claim

> An active-inference agent that models the world as objects and causes can understand goals
> as relations, reason about how to reach them in imagination, learn by experimenting, and
> explain *why* things happened, both surprises and successes. It verifies each explanation
> by experiment, and it becomes more robust and more sample-efficient than end-to-end policies.

Every capability below is a **mechanism with a metric**. The simulator knows the true cause of
every event, so we can score the agent's explanations exactly. The real world cannot give us that.

Technique choices for each module, with verified citations: [LITERATURE.md](LITERATURE.md).
Week-by-week plan to the application deadline: [SPRINT.md](SPRINT.md).

## How we build it: five rules

1. **Vertical slice first.** In the first weeks, get one task working end to end through *every*
   level, crudely: understand goal → imagine → act → remember → explain. Then deepen each level.
   If each phase is perfected in isolation, we learn about integration problems in month 5.
2. **Privileged state first, then pixels.** Reasoning, memory and causal learning are first
   built on the simulator's true object states. Learned perception from pixels replaces them
   later. Research on "why" is then never blocked by perception bugs, and the ablation "true
   state vs learned perception" becomes a result in itself.
3. **Every explanation must be testable.** The agent never just stores a story. A hypothesis
   ("the cube is heavier than expected") must predict something, and the agent runs an
   experiment to check it. This is what separates understanding from pattern matching.
4. **Gates, not dates.** Each phase ends with a measurable exit check. We do not move on until
   it passes, or we have written down why it did not.
5. **Contributions = ablations.** Each mechanism must be removable by config, so each claim
   ("causal memory helps") is backed by a with/without comparison.

## Architecture: five levels

```
L4 Reflection   after each episode: analyse surprises and successes, propose hypotheses,
                design experiments to test them, consolidate what was learned
L3 Memory       episodic memory (every attempt), causal graph, recipe library (what worked and why)
L2 Goals        goals as relations between objects ("red_cube ON plate"), from language / image / task;
                self-generated goals for practice
L1 Planning     imagine action sequences in the world model; expected free energy = reach the goal
                (pragmatic) + learn something (epistemic); trial and error when imagination is unsure
L0 Body         fast free-energy minimisation: body perception, learned sensor trust, motor control

Hidden-cause inference runs across levels: every persistent surprise is explained by competing
causes (occlusion, push, moved camera, heavier object, low friction, unknown) and the best
explanation changes the generative model.
```

The same currency links every level: prediction error weighted by precision. A level that
cannot explain its errors passes them up. A level that has decided something sends predictions
(priors) down.

## The thinking layer: a brain-inspired neural cognitive core

L3 (memory) and L4 (reflection) are not a symbolic reasoner or an LLM. They are **neural networks
modelled on the functions of brain systems**. This is brain-*inspired* at the level of
computation, not a simulation of biological neurons.

| Brain system | Function | Our module |
|---|---|---|
| Prefrontal cortex | working memory, deliberation | **Thinking network: HRM** (Hierarchical Reasoning Model). A slow high-level module holds hypotheses and plans; a fast low-level module checks them against memory and imagined rollouts. Learned halting stops it when confident, so it thinks longer on harder problems. |
| Hippocampus | episodic memory, pattern completion, replay, imagining | **Memory network:** content-addressable episodic memory (modern Hopfield / attention), recall of similar experiences, imagined and counterfactual rollouts |
| Cerebellum | fast forward models for movement | **L0 body model** |
| Basal ganglia + dopamine | action selection, learning from success | **Policy selection** by expected free energy; dopamine ~ precision (confidence) over policies |
| Acetylcholine | expected uncertainty, trust in the senses | **Sensory precision** per sense and per object |
| Noradrenaline | unexpected uncertainty, "something changed" | **Surprise signal:** triggers cause inference and temporarily raises learning rates |
| Sleep | consolidation, replay, recombination | **Offline phase:** replay into the world model, plus dreaming of self-generated goals |
| Fast and slow plasticity | short-term adaptation, lifelong learning | **Fast weights** (within an episode) plus **slow weights** (across the lifetime) |

The acetylcholine/noradrenaline mapping follows Yu & Dayan, "Uncertainty, Neuromodulation, and
Attention" (Neuron, 2005), verified in [LITERATURE.md](LITERATURE.md). The thinker's backbone is
HRM; memory writes are surprise-gated, Titans-style.

**Decision (2026-09-23): the thinker is HRM**, trained by backpropagation. HRM has no built-in
uncertainty, so two things keep the agent's sense of confidence:
- **Uncertainty goes in:** the thinker receives the lower levels' precisions, prediction errors and
  ensemble disagreement as inputs.
- **Uncertainty comes out:** its readout heads output probability distributions over causes, plans
  and success factors, not single answers. Their calibration is measured.

L0–L2 stay precision-weighted active inference. A predictive-coding variant of the thinker is
not planned; it stays an optional later comparison.

**Evidence so far (2026-09-24), for the week-4 gate.** On two synthetic cause-inference tasks with an
exact Bayesian teacher, HRM matched the teacher but did *not* beat a same-size one-pass transformer:

| Task | HRM agreement / KL | Transformer agreement / KL | Training time, HRM vs transformer |
|---|---|---|---|
| single cause (`results/thinker/synthetic_eval.md`) | 95.0% / 0.0064 | 97.0% / 0.0026 | 10.1 vs 1.1 min |
| up to 3 causes, explaining away (`results/thinker/multicause_eval.md`) | 87.9% / 0.080 | 93.2% / 0.029 | 13.8 vs 1.2 min |
| **real agent episodes**, 5 causes, 100 episodes of run v7 (`results/thinker/real_v7.md`) | 81.7 ± 8.5% / 0.64 | 90.0 ± 8.2% / 0.74 | 197 vs 46 s per seed |
| **real agent episodes**, 5 causes, 1000 episodes (`results/thinker/real_c1000.md`) | **96.0 ± 1.1%** / 0.154 | 95.8 ± 1.5% / 0.163 | 414 vs 64 s per seed |

On real episodes (60 training episodes per seed, 3 seeds) the transformer again agrees more with the
teacher, while HRM has the lower KL. With 20 test episodes per seed, one episode is 5 points, so
the gap is within about one standard deviation; more real episodes are needed before the gate decides.
With 1000 real episodes (600 for training per seed) HRM reaches 96.0 ± 1.1% agreement (worst seed 95.0%):
**the week-4 gate (> 90%) is passed**. It is on par with the transformer (95.8 ± 1.5%), not better, and
trains 6x slower; its advantage is still expected on sequential problems, not on cause inference.

HRM does spend more thought on harder cases in one sense (1.0 -> 1.9 rounds from 0 to 3 simultaneous
causes), but extra rounds barely improve accuracy. Likely reason: cause inference is combinatorial,
not sequential, so one pass suffices. HRM's advantage should appear on genuinely sequential problems
(multi-step planning, search, chess). The gate compares both thinkers on real agent episodes and on
a sequential planning task before committing.

**One thought cycle:** recall memories → imagine outcomes in the world model → update beliefs
about causes and plans → repeat until confident → act.

**Readable, not a black box.** Readout heads decode the thinking state into checkable outputs:
cause probabilities, active relations, success factors and recipe preconditions. The heads are
trained and scored against simulator ground truth.

**Teacher, then student, then beyond.** The explicit Bayesian cause inference and counterfactual
credit assignment (Phase 3) come first, as a teacher. The neural thinker learns to reproduce them
far faster (amortised inference). It is then trained to go beyond them: new causes, cluttered
scenes, longer reasoning. The research question is whether a neural thinker learns to
approximate Bayesian reasoning, and where it surpasses it.

**Training:**
- meta-learning across many simulated episodes and tasks (learning to learn)
- the free-energy objective
- readout supervision from ground truth
- plan-outcome feedback

After deployment it keeps adapting through fast weights and sleep-phase replay.

**Metrics:**
- thinking time vs problem difficulty
- success vs number of thought steps
- student–teacher agreement
- cause accuracy on causes the teacher's hypothesis library does not contain
- adaptation speed after a change

---

# Stage 1: the robot arm (Phases 1–5)

## Phase 1: Object world, vertical slice, object-centric world model (4–5 weeks)

**Why first:** a lone arm reaching a pose has nothing to understand. Relations, causes and
recipes need objects.

### 1a. Object scene and ground truth (week 1)
- Table, 3–5 blocks (colours, sizes), a plate, a bowl and optionally a drawer. Randomised
  placement, colours, lighting and camera jitter.
- Gripper control: Cartesian end-effector deltas + open/close, in addition to joint velocities.
- **Compliant control mode** (impedance or torque), so recovering from a push is the agent's job
  and not the stiff servo's. This fixes a known Phase 0 weakness.
- **Ground-truth event logger:** contacts, grasps, slips, applied forces and every disturbance or
  property change (mass, friction, camera pose), each with a timestamp. This is the answer key
  for scoring explanations.
- **Predicate functions** from simulator state: `on(a,b)`, `in(a,b)`, `grasped(a)`,
  `left_of(a,b)`, `near(a,b)`, `upright(a)`. They are the ground truth for goal relations.

### 1b. Vertical slice on privileged state (weeks 2–3)
One task, "put the red cube on the plate", end to end with crude versions of every level:
- State = true object poses. World model = small learned dynamics model over object states.
- L2: goal = `on(red_cube, plate)` as a preference.
- L1: MPPI planning in the learned model with the expected-free-energy objective.
- L3: every episode saved to episodic memory.
- Hidden causes: 3 hypotheses (none / push / heavier object), scored by free energy.
- L4: a structured episode report (surprises, inferred cause, outcome).

**Gate:** the loop completes the task above 70% of the time, and the report names the right cause
for the injected push/mass disturbances above 70% of the time.

### 1c. From pixels to objects (weeks 3–5)
- **Object-centric perception:** slot attention on frozen DINOv2 features (SlotContrast-style), one latent "slot" per
  object. Start supervised by the simulator's segmentation masks, then reduce supervision and
  report the gap.
- **World model over slots:** a transformer that predicts every slot's next latent from all slots
  plus the action, in frozen feature space as in DINO-WM (no pixel decoder in the planning loop). Interactions between objects appear as attention between slots, which later
  seeds the causal graph. An ensemble of about 5 gives epistemic uncertainty.
- **Per-slot learned precision:** each slot predicts its own uncertainty, so an occluded object's
  trust drops automatically.
- 256×256 decoder from slots, for visualisation and pixel-level prediction error.

**Gate:** multi-step prediction (1, 5 and 20 steps) beats a non-object baseline world model, and
learned slots match the true objects (a standard segmentation match score) above an agreed bar.

---

## Phase 2: Thinking before acting, experimenting when unsure (3–4 weeks)

- **Imagination planning:** MPPI/CEM over action sequences in the slot world model. Plans are
  scored by expected free energy:
  - pragmatic: log-probability that the goal relations hold at the end
  - epistemic: expected information gain = ensemble disagreement on predicted outcomes
- **Trial and error as a policy, not a bug:** when imagined outcomes disagree too much, a short
  *information-seeking* action (poke, lift slightly, look from another angle) is worth more than
  a goal-directed one. Expected free energy chooses between them automatically.
- **Plan-then-verify loop:** after acting, compare the real outcome with the imagined one. A
  mismatch updates the world model and triggers cause inference (Phase 3).
- **Surprise escalation:** persistent unexplained error at L0 → replan at L1 → reconsider goal at L2.

**Gate:** multi-step tasks (e.g. stack 2 blocks) succeed; the epistemic term measurably reduces
real trials needed on a new object; far goals no longer fail the way PixelAI did (35% at 23 cm).

---

## Phase 3: Memory, causes and recipes (5–6 weeks, the core novelty)

### 3a. Episodic memory
Every attempt is stored: observations (slot latents), actions, predictions, prediction errors,
inferred causes, goal and outcome. It is indexed so the agent can recall "situations like this
one". Replay continually improves the world model (continual learning), with rehearsal so old
skills are not forgotten.

### 3b. Explaining the unexpected: hidden-cause inference
- A **library of cause hypotheses**. Each is a specific change to the generative model:
  occlusion, external push, camera moved, object heavier/lighter, friction changed, object
  swapped, and **unknown**.
- Each hypothesis is scored by the free energy of recent observations under the modified model.
  The posterior over causes is proportional to exp(−F). This is Bayesian model comparison,
  native to active inference.
- **Unknown** wins when nothing fits. That triggers exploration and marks the event for reflection.
- Continuous parameters (camera pose, mass, friction) are then **inferred online**, fixing the
  Phase 0 camera-shift failure (5% success) by recalibrating instead of chasing a wrong pose.

### 3c. Explaining the expected: why did it work?
- **Counterfactual credit assignment:** after a success, the agent replays the episode in its world
  model with one factor changed at a time: skip an action segment, move the grasp point, change the
  approach angle. Factors whose change breaks the success are **causes of the success**.
- **Causal graph:** which object and action variables influence which. It is seeded by attention
  in the world model and then **confirmed by the agent's own interventions**. Acting on the world
  is the strongest evidence for cause and effect.
- **Recipes:** each success is distilled into
  `goal relation + preconditions (what must be true) + action sketch + why (the causes found above)`.
  Recipes are recalled by matching the current situation. They become a strong prior for the
  planner, so a solved problem is solved faster next time; this acts as habit formation.

### 3d. The neural thinker learns from the explicit teacher
- Hippocampus-style memory network over episodic memory. It is trained to recall situations
  whose outcome predicts the current one.
- HRM thinking network (slow and fast recurrent modules, learned halting) with probabilistic readout heads. It is trained
  to match the explicit cause posteriors and success factors from 3b–3c, then fine-tuned on the
  free-energy and outcome objectives.
- Fast weights for within-episode adaptation, and a noradrenaline-style surprise signal that
  gates learning rate.

**Gate:**
- the thinking network matches teacher cause inference above 90% agreement at a fraction of the compute
- cause identification accuracy against the simulator logs (per cause type, and "unknown" for new causes)
- success at reproducing a task after a *single* success, in new layouts
- the causal graph matches the simulator's true contact and physics structure above chance, by a margin

---

## Phase 4: Understanding goals like a human; its own ideas (3–4 weeks)

- **Relational goals.** A goal is a set of relations over objects with a preference strength
  (the active-inference "prior preferences"). A learned relation classifier over slots
  (supervised by Phase 1 predicates) turns the scene into relations, so goals survive changes in
  lighting, viewpoint and object appearance, where picture matching fails.
- **Three goal inputs, one goal representation:**
  - language: an LLM translates the instruction into relations (optional; a rule parser is the fallback)
  - image: relations inferred from the goal image, so the *meaning* is matched, not the pixels
  - abstract task: "tidy up" becomes a set of relations
- **Subgoals:** a long goal is broken into an ordered set of relations, guided by recipes (their
  preconditions) and by planning.
- **Its own ideas: self-generated goals.** In free time, the agent proposes goals where its
  *learning progress* is highest: not too easy, not impossible. It practises them first in
  imagination ("dreaming") and then for real. Skills it was never asked to learn become recipes.

**Gate:** the same instruction as language, image or a different viewpoint gives the same outcome;
self-practice improves later success on held-out tasks against an agent without free practice.

---

## Phase 5: Reflection, full evaluation, write-up (4–5 weeks)

- **Sleep phase:** offline replay consolidates the world model and memory; dreaming generates new
  goals and counterfactuals. The thinking network is trained on its own replayed experience.
- **L4 Reflection**, carried out by the neural thinker, after each episode (and in batches):
  1. structured analysis: surprises, inferred causes and confidence, success factors, open questions
  2. hypothesis generation by the thinking network, from the unknown-cause queue and memory
     recall. An LLM may be added as an optional comparison, never as the core thinker.
  3. **experiment design:** for each hypothesis, choose the action with maximum information gain
     about it
  4. run, confirm or reject, and store the verdict. Only confirmed knowledge enters the causal
     graph and the recipes.
- **Evaluation**, all with seeds and confidence intervals:

  | Capability | Metric |
  |---|---|
  | Goal understanding | success across language / image / viewpoint-changed goals |
  | Thinking / planning | success versus task horizon and goal distance |
  | Trial and error | real trials needed on novel objects |
  | Learning from experience | success on task N+1 after tasks 1…N (forward transfer), no forgetting |
  | Explaining surprises | cause-identification accuracy against simulator ground truth |
  | Explaining success | one-shot reproduction rate; causal-graph accuracy |
  | Own ideas | held-out success with vs without self-practice |
  | Reflection | fraction of hypotheses confirmed; time to resolve an unknown cause |
  | Robustness | full disturbance suite, recovery time, time outside tolerance |

- **Baselines:**
  - PixelAI 2020 (Phase 0)
  - visual servoing + IK
  - ACT
  - Diffusion Policy
  - DreamerV3 or TD-MPC2
  - OpenVLA-style, if it fits
- **Ablations:** remove each of memory, cause inference, recipes, epistemic term, self-practice,
  reflection, and learned perception (privileged state instead).
- **Deliverables:** paper (arXiv / workshop), project page with videos (*the agent explaining its
  own failures* is the headline demo), clean public repo, ROS 2 bridge demo.

---

## Timeline and compute

| Phase | Weeks | Cumulative | Kaggle T4 GPU-h (est.) |
|---|---|---|---|
| 1. Objects + slice + world model | 4–5 | 4–5 | 30–60 |
| 2. Planning + trial and error | 3–4 | 7–9 | 5–15 |
| 3. Memory, causes, recipes | 5–6 | 12–15 | 15–30 |
| 4. Goals + own ideas | 3–4 | 15–19 | 10–20 |
| 5. Reflection + evaluation | 4–5 | 19–24 | 140–290 (mostly baselines) |

About 5–6 months full time. The GPU-hour figures are estimates; each is re-measured with
`scripts/benchmark.py` at the start of its phase. LLM calls (language goals, optional
reflection) cost API credits, not GPU time.

**Team split:**
- You: core agent (L0–L3), the research-critical path
- Teammate: scene and tasks (1a), baselines (Phase 5, own Kaggle account), evaluation harness
- Laptop overnight: online RL baselines and long evaluation runs

**If time runs short:** Phases 1–3 alone already answer "does it explain surprises and successes?".
That is a complete, strong project. Phases 4–5 make it outstanding.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Object-centric perception from pixels is hard | Rule 2: privileged state first, mask supervision next, and report the gap honestly |
| Causal discovery is open-ended | A finite hypothesis library plus "unknown"; the agent's own interventions provide strong evidence |
| The LLM invents plausible but false explanations | The LLM only *proposes*; nothing is believed until an experiment confirms it |
| Scope creep | Vertical slice, phase gates, and the "if time runs short" line above |
| Baselines win on some tasks | Report it. The claim is about robustness, sample efficiency and explanation, not winning everywhere |

---

# Stages 2-7: from the arm to a body that walks, explores, plays, talks and rides

Week-by-week build steps for these stages: [BUILD_PLAN_STAGES_2_7.md](BUILD_PLAN_STAGES_2_7.md).

Each stage adds one new ability and must reuse earlier ones. Each gets its own gate before the
next starts. Durations are rough and assume Stage 1 is finished.

## Stage 2: Walk and run (3-5 months)

- **Body:** a humanoid model for MuJoCo, such as the Unitree humanoids in MuJoCo Menagerie
  (check which models are current when the stage starts). It trains with GPU-parallel physics
  (MuJoCo's JAX backend, MJX), because locomotion needs millions of practice steps.
- **How it learns:**
  - *motor babbling:* random and curiosity-driven movements build a body model (which joint moves what)
  - *curiosity:* the epistemic term drives it towards movements it cannot yet predict
  - *falling is information:* each fall is a surprise that cause inference explains (lost balance
    forward or sideways, slipped, pushed)
  - *goals as preferences:* "stay upright", then "move forward", then "move fast"
  - a gait is stored as a skill recipe; running is discovered as a faster regime of the same body model
- **Honest note:** learning to walk *purely* by active inference from scratch is research-level
  hard. The plan: L0 motor skills are learned with a fast policy trained by
  reinforcement-style optimisation of the free-energy objective (amortisation, as BMPC does in
  Stage 1), while the upper levels stay active inference. This is the same split as Stage 1.
- **Gate:**
  - walks a set distance without falling
  - recovers from pushes
  - runs faster than it walks
  - fewer falls over time; measure the learning curve against a standard RL locomotion baseline

## Stage 3: Explore a maze by walking and running (1-2 months)

- Procedurally generated mazes with rooms, corridors, dead ends and landmarks.
- **How it learns:** curiosity plus episodic memory of places (hippocampus-like), so it builds a
  map, avoids re-exploring and plans routes. It chooses running on long straight corridors and
  walking near turns, reusing Stage 2.
- **Gate:**
  - coverage vs time against random and frontier-exploration baselines
  - finds a goal in a new maze faster after seeing other mazes (transfer)
  - never re-learns walking

## Stage 4: Doors and drawers (2-3 months)

- The maze gets doors (hinged, sliding, locked) and furniture with drawers. Keys hidden in
  drawers open locked doors. This forces **multi-step causal chains**: find the drawer -> open it ->
  take the key -> unlock the door -> go through.
- The humanoid needs arms and hands (a humanoid model with hands, or a simplified gripper).
  Manipulation skills from Stage 1 transfer through the shared upper levels.
- **How it learns:**
  - trial and error on handles
  - cause inference ("the door did not open *because* it is locked")
  - recipes with preconditions ("open door requires key")
- **Gate:**
  - solves unseen maze layouts that need key chains
  - explains why a door failed to open (scored against the simulator's event log)
  - one-shot reuse of a new door type

## Stage 5: Chess, from a teacher to strong opponents (2-4 months, can run alongside Stages 2-4)

- **Learning from a teacher:** the rules are not coded in. A simulated teacher shows legal and
  illegal moves, demonstrates openings and tactics, and corrects mistakes. The agent infers the
  rules as causal regularities ("a bishop moves diagonally"), which reuses the Stage 1 mechanism.
- **Then practice:** self-play plus games against the open-source engine Stockfish at increasing
  strength levels, as the "top player" opponent. The HRM thinker does the look-ahead, and
  thinks longer in sharp positions.
- **Honest scope:** superhuman chess needs AlphaZero-scale compute. On our hardware the realistic
  target is to learn the rules from the teacher alone, and then reach a *measured* rating by
  playing a range of Stockfish levels and tracking an Elo curve. We report whatever level it
  reaches. It is also possible to play physically, by moving pieces with the arm.
- **Gate:**
  - legal-move accuracy after teaching
  - Elo estimate vs engine levels
  - rating vs thinking time (does thinking longer help?)

## Stage 6: Learn a new language by listening (3-6 months)

- **Setting:** simulated people talk to each other *about what is happening* in the scene: "she
  put the red cup in the bowl", "open the door", "the drawer is stuck". The conversations are in
  a language the agent has never been taught, e.g. Arabic or English. They are generated from
  scene events and templates, and later by an LLM playing the humans, so every sentence has a
  known meaning.
- **How it learns:** cross-situational learning, as children do. Words that keep co-occurring with
  the same objects, relations and events become linked to them ("red" <-> red things, "in" <-> the
  in-relation). Grammar is learned as structure over those links. It already understands the
  meanings from Stages 1-4, so language maps onto concepts it has.
- **Then use it:** follow spoken instructions, answer simple questions about the scene, and
  describe its own actions and explanations ("I dropped it because it slipped").
- **Honest scope:** grounded vocabulary, simple sentences and instructions; not fluent open
  conversation. Speech audio can be added later; text tokens first.
- **Gate:**
  - word-meaning accuracy
  - instruction-following success with unseen word combinations (compositional generalisation)
  - accuracy of its own descriptions against the event log

## Stage 7: Ride a bike after watching (3-6 months)

- **Setting:** a simulated bicycle (to be modelled in MuJoCo) and videos of a simulated rider.
- **How it learns:**
  - *observation first:* it watches riders and infers what they do (pedal rhythm, steering into the
    fall, leaning), using its own body model to interpret another body (imitation from observation)
  - *practice:* it tries, falls, and explains falls with cause inference
  - *transfer:* balance and control from Stage 2 give it a head start
- **Gate:**
  - rides a set distance
  - fewer practice attempts than an agent without Stage 2 (proof of transfer)
  - fewer attempts than an agent that did not watch (proof that observation helped)

## Programme-level metrics

The same questions are asked at every stage, so the programme tests one scientific claim:
- **Transfer:** does each stage learn faster because of the earlier ones? (ablation: start from scratch)
- **Explanation:** are its explanations of failures and successes right? (scored against event logs)
- **Curiosity:** does exploration find what matters faster than baselines?
- **Thinking:** does it think longer on harder problems, and does that help?
