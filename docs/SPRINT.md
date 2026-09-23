# Application sprint: 8 weeks (2026-09-24 to ~2026-11-23)

The 8 weeks are for **building**. Goal by the deadline: **Stage 1 working end to end on the arm**, plus
**demos of doors and drawers, a maze and chess**, and a small application package (videos,
headline numbers, a short summary). Full research (baselines, ablations, seeds, the paper) happens
**after the deadline** and is presented at interviews. Stages 2–7 of [ROADMAP.md](ROADMAP.md) are
presented as the PhD research plan, not as finished work.

Each week ends with a **gate**. If a gate fails, fix it or cut scope (see the cut list). Never
skip a gate: a demo built on a broken module costs more time than it saves.

## Week-by-week

| Week | Dates | Build | Gate (must pass) |
|---|---|---|---|
| **1** | Sep 24 – Sep 30 | **Vertical slice on privileged state (1b)**: learned dynamics over true object states; relation goal `on(red, plate)`; MPPI with expected free energy; episodic memory; cause inference with 3 hypotheses (none / push / heavier object); episode report. Review and integrate the HRM prototype. First git commit and GitHub repo (needed for Kaggle). | Task success > 70%; correct cause for injected push/mass > 70% |
| **2** | Oct 1 – Oct 7 | **Perception**: data collection with the scripted policy plus random and curious exploration (Kaggle + laptop). Frozen DINOv2 features. Object slots supervised by simulator masks. | Slots match the true objects (segmentation score above an agreed bar) |
| **3** | Oct 8 – Oct 14 | **World model over slots** (DINO-WM style, 3–5 member ensemble) replaces privileged state in the loop. | 1/5/20-step prediction beats a non-object baseline; slice success from pixels > 50% |
| **4** | Oct 15 – Oct 21 | **Causes and recipes**: full cause library + online camera and mass inference (fixes the Phase 0 camera-shift failure); counterfactual credit assignment; recipe library. **HRM thinker** trained on real episodes against the Bayesian teacher. | Camera-shift success far above PixelAI's 5%; one-shot reuse of a recipe in new layouts; HRM agrees with teacher > 90% |
| **5** | Oct 22 – Oct 28 | **Demo A: doors and drawers with the arm.** Articulated objects; locked drawer as a hidden cause; "open drawer" recipe with preconditions. | Opens unseen drawer/door types; explains a failed opening correctly (vs event log) |
| **6** | Oct 29 – Nov 4 | **Demo B: maze exploration** with a simple mobile body and the same upper levels: curiosity plus memory of places. | Coverage beats a random-exploration baseline |
| **7** | Nov 5 – Nov 11 | **Demo C: chess rules learned from a teacher**, played by the HRM thinker against low Stockfish levels. Start of the post-deadline baselines on the teammate's Kaggle account. | Legal-move accuracy after teaching; first measured Elo point |
| **8** | Nov 12 – Nov 18 | Mon–Wed: finish and stabilise. **Thu–Sun: minimum application package** (below). | Package reviewed by at least one other person |
| buffer | Nov 19 – Nov 23 | Slack for overruns. Nothing new starts here. | – |

### Minimum application package (end of week 8)

Small on purpose; everything else waits until after the deadline.
- **3–4 demo videos:** the agent explaining its own failure; camera-shift recovery; doors and drawers; the maze or chess
- **Headline numbers with a sanity check,** not a full study: slice success rate, cause-identification
  accuracy vs the event log, camera-shift success vs PixelAI's 5%, PixelAI's distance curve
- **A 2-page project summary** plus a simple project page, and a clean public repo
- Material for the statement of purpose: what was built, what the first results show, Stages 2–7 as the plan

## After the deadline (for interviews)

| When | Work |
|---|---|
| Weeks 9–10 | Full evaluation: seeds and confidence intervals, ablations (memory, cause inference, recipes, epistemic term, thinker) |
| Weeks 10–12 | Baselines: PixelAI, a model-based RL baseline (TD-MPC2 or DreamerV3), imitation baselines (teammate) |
| Weeks 12–14 | arXiv-style paper and updated project page; send updates to potential advisors |
| Then | Stages 2–7, from January 2027: see [BUILD_PLAN_STAGES_2_7.md](BUILD_PLAN_STAGES_2_7.md) |

## Cut list if behind (cut from the top)

1. Demo C (chess)
2. Demo B (maze)
3. Learned slots from pixels: fall back to mask-supervised slots and report it
4. Anything in the application package beyond the videos and headline numbers
5. **Never cut:** the vertical slice, cause inference scored against the event log, the camera-shift
   fix, the HRM thinker, and honest headline numbers. Those are the contributions.

## Who does what

| Who | Each week |
|---|---|
| **Claude** | Builds and tests the week's modules (several agents in parallel where modules are independent); keeps docs and results current; proposes gate decisions with evidence |
| **You** | Review what was built and understand it (you must be able to defend every choice in interviews); launch Kaggle runs; decide at each gate; write your statement of purpose from week 4 onwards |
| **Teammate** | From week 7: baselines on their Kaggle account (for after the deadline); review the package in week 8 |
| **Laptop** | Overnight: evaluation runs and long simulations |

## Outside the code (your side)

- **Weeks 1–3:** shortlist programmes and potential advisors whose work matches (deep active
  inference, world models, cognitive robotics). Check each programme's policy on AI-assisted work.
- **Weeks 3–4:** contact potential advisors, with the Phase 0 results and plan as a short summary;
  confirm recommendation letters.
- **Weeks 4–8:** statement of purpose. The project is its centrepiece; Stages 2–7 are the research vision.

## Risks for this sprint

| Risk | Mitigation |
|---|---|
| Claude usage limits (hit once already) | Large parallel jobs early in a session; keep the plan in files so work resumes after a reset |
| Kaggle quota runs out | Building needs little GPU; heavy evaluation moved after the deadline |
| A research module does not work (slots, HRM) | Every module has a fallback in [LITERATURE.md](LITERATURE.md); the gate decides when to switch |
| Too little time to understand the work | 30 minutes of review per day is part of the plan, not optional |
