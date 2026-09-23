# Build plan: Stages 2–7 (after the application deadline)

This turns the Stages 2–7 section of [ROADMAP.md](ROADMAP.md) into build steps. It starts after
the post-deadline research weeks in [SPRINT.md](SPRINT.md) (weeks 9–14, to early January 2027).
Durations assume Claude does most of the building and you review and decide at gates. Training
time and research uncertainty, not typing, set the pace.

## Timeline

| Stage | Build | Weeks | Approx. dates (2027) | Needs |
|---|---|---|---|---|
| 0 | Cross-cutting infrastructure | 2 | Jan 5 – Jan 18 | Stage 1 |
| 2 | Walk and run | 9 | Jan 19 – Mar 22 | infra |
| 3 | Maze exploration | 4 | Mar 23 – Apr 19 | 2 |
| 4 | Doors and drawers | 6 | Apr 20 – May 31 | 1, 3 |
| 5 | Chess | 8 | Jan 19 – Mar 15, **alongside Stage 2** | thinker (Stage 1) |
| 6 | Language by listening | 8 | Jun 1 – Jul 26 | 1–4 |
| 7 | Bike riding | 9 | Jul 27 – Sep 27 | 2 |

About **9 months, ending around the time a PhD would start**; the realistic range is 29–51 weeks, see
[TIME_AND_COMPUTE.md](TIME_AND_COMPUTE.md). Chess runs in parallel because it
needs the thinker, not a body. Every date moves if a gate fails. That is expected, and the gates
are what keep the programme honest.

---

## Stage 0: Cross-cutting infrastructure (2 weeks)

Built once, used by every later stage.

1. **Body interface.** One adapter per body, the only body-specific code (L0):
   - it provides observations, proprioception and a *skill API* (`move(velocity, heading)`,
     `reach(pos)`, `grasp()`, `stop()`, …)
   - L1–L4 call skills and see objects, relations and events; they never see joints
2. **Unified event log** for every environment: one schema for contacts, falls, slips, doors
   opening, board moves, utterances and disturbances. It is the answer key in every stage.
3. **Skill library.** Learned skills with preconditions, effects and a "why" (Stage 1 recipes,
   generalised). Skills from one stage are available to the next.
4. **GPU physics setup.** MuJoCo's JAX backend (MJX) for parallel simulation of thousands of bodies.
   JAX GPU support on native Windows is limited, so MJX runs in **WSL2 on the laptop** or on
   **Kaggle (Linux, T4)**. Verify current versions when the stage starts.
5. **Stage evaluation harness.** The same four programme questions for every stage:
   - transfer: vs the agent starting from scratch
   - explanation accuracy vs the event log
   - curiosity: exploration efficiency
   - thinking: time vs difficulty
6. **No-forgetting regression suite.** After every stage, re-run the earlier stages' gate tests
   (e.g. the arm tasks). Learning to ride a bike must not break block stacking.

**Gate:**
- the Stage 1 arm runs through the new body interface with unchanged results
- one MJX environment trains on Kaggle

---

## Stage 2: Walk and run (9 weeks)

| Weeks | Build |
|---|---|
| 1 | **Humanoid body:** a humanoid from MuJoCo Menagerie (e.g. a Unitree model; pick the current one with good actuators). MJX environment, domain randomisation (mass, friction, motor strength), pushes. Event log extended with foot contacts, falls, slips and trips. |
| 2–3 | **Body model by motor babbling:** random and curiosity-driven movements generate data; an L0 predictive body model learns "which motor command moves what", with ensemble uncertainty. |
| 4–6 | **Locomotion skill:** preferences as the goal ("upright", then "forward at speed v"), the epistemic term as exploration bonus. A fast policy is trained on this free-energy objective with GPU-parallel reinforcement-style optimisation (PPO in MJX). This is the amortisation split from Stage 1. The commanded speed is an input, so **walking and running are one skill**. |
| 7 | **Fall-cause inference:** each fall is a surprise explained by hypotheses (pushed, slipped, tripped, lost balance forward/sideways), scored against the event log. The explanation feeds recovery (e.g. wider stance after slips). |
| 8 | **Skill API:** `move(velocity, heading)`, `stop()`, `turn()`; the L1 planner and HRM thinker now control the humanoid through skills, as they controlled the arm. |
| 9 | Evaluation and regression suite. |

**Compute:** ~125–370 T4-equivalent GPU-hours including sweeps and retries (see
[TIME_AND_COMPUTE.md](TIME_AND_COMPUTE.md)). Re-measure with the benchmark script at the start of the stage.

**Gate:**
- walks a set distance without falling
- recovers from pushes
- runs measurably faster than it walks
- fall causes identified > 70%
- learning curve compared with a standard RL locomotion baseline

**Fallback:** if learning from the free-energy objective does not produce a gait, start from a
standard RL gait and let the agent adapt it with its own objective. Report the difference.

---

## Stage 3: Explore a maze by walking and running (4 weeks)

| Weeks | Build |
|---|---|
| 1 | **Maze generator:** procedural walls, rooms, corridors, dead ends and landmarks (coloured objects); egocentric camera; ground-truth map for scoring. |
| 2 | **Place memory** (hippocampus-like): episodic memory of places as keyframes plus learned embeddings, linked into a topological map; recognises revisited places. |
| 3 | **Exploration as expected free energy:** the epistemic term is uncertainty about the map (unvisited frontiers, ambiguous places). The HRM thinker plans routes; the agent **chooses running on long corridors and walking near turns**. |
| 4 | Goal search ("find the red object"), transfer across mazes, evaluation. |

**Gate:**
- coverage vs time beats random and frontier-exploration baselines
- finds goals faster in new mazes after training on others
- the regression suite shows walking intact

---

## Stage 4: Doors and drawers in the maze (6 weeks)

| Weeks | Build |
|---|---|
| 1–2 | **Articulated objects:** hinged and sliding doors, locked doors, drawers; keys hidden in drawers. Event log: handle turned, door opened, locked, unlocked. |
| 2–3 | **Hands:** a humanoid with hands, or a simplified gripper on the humanoid's arm. **Fallback:** a mobile base carrying the Stage 1 Panda arm, which keeps the research question while avoiding dexterous-hand control. |
| 3–4 | **Transfer from Stage 1:** the arm's door/drawer demo (sprint week 5) recipes are loaded into the skill library; the humanoid reuses them through the upper levels. |
| 5 | **Causal chains:** find drawer → open → take key → unlock door → pass. Cause inference ("did not open *because* locked"), recipes with preconditions ("open door requires key"). |
| 6 | Evaluation. |

**Gate:**
- solves unseen maze layouts that require key chains
- explanation of failed openings correct > 70%
- one-shot reuse of a new door type
- faster than without the Stage 1 recipes (transfer)

---

## Stage 5: Chess, from a teacher to strong opponents (8 weeks, alongside Stage 2)

| Weeks | Build |
|---|---|
| 1–2 | **Teacher:** a curriculum of lessons generated with the `python-chess` library: how each piece moves, captures, check, checkmate patterns, simple openings. The teacher demonstrates, marks illegal moves and corrects mistakes. **The rules are never coded into the agent.** |
| 3–4 | **Rule learning:** the agent's world model over the board learns which moves are legal and what they cause, as causal regularities, reusing the Stage 1 machinery. Scored by legal-move accuracy on unseen positions. |
| 5–7 | **Play:** the HRM thinker searches ahead (a small MCTS-style look-ahead using its learned model and a learned value), improved by self-play. Opponents: the open-source **Stockfish engine at increasing skill levels**, the "top player" in simulation. Ratings estimated from match results as an Elo curve. |
| 8 | **Optional physical chess:** the Stage 1 arm moves real pieces on a MuJoCo board. Evaluation. |

**Honest scope:** superhuman strength needs AlphaZero-scale compute. The target is rules learned
only from teaching, then a *measured* rating, reported as whatever it reaches.

**Gate:**
- legal-move accuracy > 99% after teaching
- an Elo estimate against several Stockfish levels
- rating improves with thinking time

---

## Stage 6: A new language by listening (8 weeks)

| Weeks | Build |
|---|---|
| 1–2 | **Conversation simulator:** simulated people talk *about what is happening* in scenes from Stages 1–4 ("she put the red cup in the bowl", "the drawer is stuck"). Utterances come from templates driven by the event log, and later from an LLM playing the people, so every sentence has a known meaning. A distinctive choice of language: **Arabic** (or English). |
| 3–4 | **Word grounding** (cross-situational learning, as children do): words that repeatedly co-occur with the same objects, relations and events become linked to them. Scored against the event log's ground truth. |
| 5–6 | **Structure:** a small model learns to map between utterances and grounded meanings in both directions, so grammar emerges as structure over known concepts. |
| 7 | **Use:** spoken instructions become relational goals for L2; it answers questions about the scene and **describes its own actions and explanations** ("I dropped it because it slipped"). |
| 8 | Evaluation. |

**Honest scope:** grounded vocabulary, simple sentences and instructions; not fluent open
conversation. Text first; speech audio later.

**Gate:**
- word-meaning accuracy
- instruction success on *unseen word combinations* (compositional generalisation)
- its self-descriptions match the event log

---

## Stage 7: Ride a bike after watching (9 weeks)

| Weeks | Build |
|---|---|
| 1–3 | **Bicycle model** in MuJoCo: frame, wheels, steering, pedals, and the humanoid seated on it (hands on bars, feet on pedals, attached by constraints). This modelling is a real engineering task in itself. **Fallback:** a simplified rider (torso lean, steering, pedal torque) on a realistic bike. |
| 3–4 | **Demonstrations to watch:** an expert rider driven by a classic balance controller (steer into the fall). Rendered as videos. |
| 5–6 | **Learning by observation:** the agent watches the videos and infers the rider's actions using its own body model (inverse dynamics: what would *I* have to do to move like that?). Those inferred actions initialise its riding skill. |
| 7–8 | **Practice:** it rides, falls, and explains falls with cause inference; Stage 2 balance and control give it a head start. |
| 9 | Evaluation. |

**Gate:**
- rides a set distance
- **fewer practice attempts** than (a) an agent without Stage 2 and (b) an agent that did not watch
  (the two transfer proofs)

---

## Working method for every stage

1. **Week 1 of each stage:** re-run the literature search for that stage (it moves fast), then
   fix the gate numbers before building.
2. **Build the vertical slice first,** crude end to end, then deepen.
3. **Privileged state before learned perception,** as in Stage 1.
4. **Every stage adds to the event log schema, skill library and regression suite.**
5. **Parallel building:** Claude agents work on independent modules at once (e.g. the chess
   teacher while Stage 2 trains). For large batches, ask for a workflow.

## Compute summary (rough; re-measure at each stage)

| Stage | Main cost | Where |
|---|---|---|
| 2 | parallel locomotion training | Kaggle T4 / WSL2 (MJX) |
| 3 | exploration episodes, place memory | laptop + Kaggle |
| 4 | manipulation practice in the maze | Kaggle |
| 5 | self-play and engine matches | laptop (CPU engine) + Kaggle |
| 6 | small language models on simulated conversations | Kaggle |
| 7 | bike practice and video-based learning | Kaggle / WSL2 (MJX) |
