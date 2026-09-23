# Literature review: techniques for each module

Compiled 2026-09-23. Every entry below was checked against its arXiv, journal or OpenReview
page. Two titles that were reported wrongly were corrected (Causal-JEPA, and the Ferraro et al.
object-centric world-model analysis). Items marked **lead** were found by search but not opened;
confirm them before citing.

## 1. World model

| Paper | Key idea | Use in this project |
|---|---|---|
| Zhou et al., **DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning**, 2024, [arXiv 2411.04983](https://arxiv.org/abs/2411.04983) | Predicts future *frozen DINOv2 patch features* instead of pixels; plans by optimisation in that space | **Adopt as the backbone.** No pixel decoder in the loop, cheap, fits 8–16 GB |
| Baldassarre et al., **Back to the Features: DINO as a Foundation for Video World Models**, 2025, [arXiv 2507.19468](https://arxiv.org/abs/2507.19468) | Extends DINO-feature prediction to video world models | Best-practice reference for multi-step rollouts in feature space |
| Hansen et al., **TD-MPC2: Scalable, Robust World Models for Continuous Control**, ICLR 2024, [arXiv 2310.16828](https://arxiv.org/abs/2310.16828) | Latent world model + MPPI guided by a learned policy | Strong baseline (Phase 5), and design reference for latent MPPI |
| Zhang et al. (Ballas), **Hierarchical Planning with Latent World Models**, 2026, [arXiv 2604.03208](https://arxiv.org/abs/2604.03208) | Hierarchical MPPI over latent rollouts; compares MPPI with CEM | Directly relevant to L1/L2 planning |

**Trend:** since 2024 the field has shifted from reconstructing pixels (DreamerV3-style) to
predicting in a frozen pretrained feature space. Our 256×256 decoder becomes a visualisation
and pixel-level-check tool, not part of the planning loop.

## 2. Object-centric perception and world models

| Paper | Key idea | Use |
|---|---|---|
| Manasyan et al., **Temporally Consistent Object-Centric Learning by Contrasting Slots** (SlotContrast), CVPR 2025 (oral), [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.pdf) | Slot attention on DINOv2 features with a temporal contrastive loss, so objects keep their identity over time | **Adopt for slots** (Phase 1c) |
| Spieler et al. (Behnke), **Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations**, 2026, [arXiv 2605.14937](https://arxiv.org/abs/2605.14937) | Slots + MPC for goal-conditioned manipulation | Closest match to our Phase 1–2 design; compare against it |
| Ferraro et al. (Matsuo), **When Object-Centric World Models Meet Policy Learning: From Pixels to Policies, and Where It Breaks**, 2025, [arXiv 2511.06136](https://arxiv.org/abs/2511.06136) | Analyses where object-centric world models fail for control from pixels | Read before building; avoid its documented failure modes |

## 3. Planning

| Paper | Key idea | Use |
|---|---|---|
| **Bootstrapped Model Predictive Control** (BMPC), ICLR 2025, [OpenReview](https://openreview.net/forum?id=i7jAYFYDcM), [arXiv 2503.18871](https://arxiv.org/abs/2503.18871) | Distils the planner into a policy and bootstraps from it; reports large data-efficiency gains over TD-MPC2 | **Adopt for amortisation:** plan with MPPI, distil into a fast policy |
| Bansal et al., **LePlanner: An Iterative Amortized Controller For World Models**, 2026, [arXiv 2609.13845](https://arxiv.org/abs/2609.13845) | Amortised iterative controller trained against a JEPA-style world model | Alternative amortiser; very recent, so treat as experimental |

## 4. Uncertainty and curiosity

| Paper | Key idea | Use |
|---|---|---|
| Joseph et al. (Zöllner), **CIG: Exploration via Conditional Information Gain**, 2026, [arXiv 2605.20878](https://arxiv.org/abs/2605.20878) | Explicit information-gain objective for exploration | Closest to the epistemic term of expected free energy; compare with ensemble disagreement |

Default: a small ensemble (3–5 heads) for disagreement, and CIG-style information gain as the
principled comparison.

## 5. Deep active inference

| Paper | Key idea | Use |
|---|---|---|
| Fujii & Murata, **Real-World Robot Control by Deep Active Inference With a Temporally Hierarchical World Model**, IEEE RA-L, [arXiv 2512.01924](https://arxiv.org/abs/2512.01924) | Multi-timescale world model plus abstract action space make expected-free-energy action selection tractable on a real robot | **Closest prior work.** Cite, compare, and build beyond (objects, causes, memory) |
| Heins et al. (Buckley), **AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models**, 2025, [arXiv 2505.24784](https://arxiv.org/abs/2505.24784) | Object-centric Bayesian active-inference agent; reported to beat DreamerV3 with far fewer interactions | Evidence that object-centric active inference is sample-efficient; template for model expansion |
| de Vries et al., **Expected Free Energy-based Planning as Variational Inference**, 2025, [arXiv 2504.14898](https://arxiv.org/abs/2504.14898) | Expected-free-energy planning as ordinary variational inference on an augmented model | **Mathematical backbone** for the L1 objective |
| Nagatsuka et al. (Hayashibe), **Stabilizing the Convergence of Pixel-Based Deep Active Inference Controllers Using Adaptive Smoothing Filters**, Biomimetics 2025, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12838793/) | Smoothing filters help PixelAI-style controllers escape local minima | A stronger PixelAI-family baseline; relevant to our measured locality problem |
| Champion et al. (Bowman), **Deconstructing deep active inference**, 2023, [arXiv 2303.01618](https://arxiv.org/abs/2303.01618) | Examines which parts of deep active-inference agents matter | **Pitfall reference** |
| Millidge, **A Retrospective on Active Inference**, blog, 2024, [link](https://www.beren.io/2024-07-27-A-Retrospective-on-Active-Inference/) | Argues that in practice active inference often reduces to RL or control theory | Opinion, not peer-reviewed, but the critique we must answer |

**Pitfall we must address explicitly:** many deep active-inference agents collapse to model-based RL
once approximated. Each claim must be tied to something RL does not have: precision-weighted
perception, the epistemic term, hidden-cause inference, and inference of the agent's own body parameters.

## 6. Causal reasoning, credit assignment, recipes, goals

| Paper | Key idea | Use |
|---|---|---|
| Yu, Ruan, Xing, **Explainable Reinforcement Learning via a Causal World Model**, IJCAI 2023, [arXiv 2305.02749](https://arxiv.org/abs/2305.02749) | Learns causal structure, then explains action effects as causal chains | Template for "why" explanations (Phase 3c) |
| Nam et al. (Balestriero), **Causal-JEPA: Learning World Models through Object-Level Latent Masking**, ICML 2026, [arXiv 2602.11389](https://arxiv.org/abs/2602.11389) | Object-level masking as a causal inductive bias for world models | Candidate training objective for the slot world model |
| Mesnard et al. (Munos), **Counterfactual Credit Assignment in Model-Free Reinforcement Learning**, ICML 2021, [PMLR](https://proceedings.mlr.press/v139/mesnard21a.html) | Separates an action's causal contribution from luck using hindsight information | Foundation for "why did it work?" credit assignment |
| Liang et al. (Ellis), **VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning**, ICLR 2025 spotlight, [arXiv 2410.23156](https://arxiv.org/abs/2410.23156) | Learns predicates online and composes skills into precondition/effect operators | **Direct template for recipes** and relational goals |
| Gaven et al. (Oudeyer), **MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces**, ICML 2025, [arXiv 2502.07709](https://arxiv.org/abs/2502.07709) | The agent predicts its own learning progress to choose practice goals | **Adopt for self-generated goals** (Phase 4) |

## 7. The brain-inspired thinking layer

| Paper | Key idea | Use |
|---|---|---|
| Wang et al., **Hierarchical Reasoning Model**, 2025, [arXiv 2506.21734](https://arxiv.org/abs/2506.21734) | Two coupled recurrent modules (slow abstract, fast detailed); ~27M parameters | **Backbone for the prefrontal thinker**; fits the laptop GPU |
| Geiping et al. (Goldstein), **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**, NeurIPS 2025, [arXiv 2502.05171](https://arxiv.org/abs/2502.05171) | A recurrent block unrolled to any depth: reason longer in latent space | Mechanism for "think longer when it is harder" |
| Zhu et al., **Scaling Latent Reasoning via Looped Language Models** (Ouro), 2025, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741) | Looped model with learned adaptive halting | Template for the halting gate |
| Behrouz et al. (Mirrokni), **Titans: Learning to Memorize at Test Time**, 2024, [arXiv 2501.00663](https://arxiv.org/abs/2501.00663) | Neural long-term memory that writes in proportion to *surprise* | Hippocampal memory writes gated by our noradrenaline-like surprise signal |
| Di Nepi et al. (Silvestri), **Titans Revisited: A Lightweight Reimplementation and Critical Analysis**, 2025, [arXiv 2510.09551](https://arxiv.org/abs/2510.09551) | Reports that Titans' gains are not robust in every setting | Treat Titans as a mechanism to test, not a proven result |
| Yu & Dayan, **Uncertainty, Neuromodulation, and Attention**, Neuron 46(4):681–692, 2005, [link](https://www.cell.com/neuron/fulltext/S0896-6273(05)00362-4) | Acetylcholine ~ expected uncertainty, norepinephrine ~ unexpected uncertainty | **Neuroscience anchor** for the precision and surprise signals |
| Müller et al. (Hutter), **Transformers Can Do Bayesian Inference**, ICLR 2022, [arXiv 2112.10510](https://arxiv.org/abs/2112.10510) | Prior-fitted networks learn to output Bayesian posteriors in context | **Template for the neural thinker imitating the Bayesian teacher** |
| Sorrenti et al. (Spampinato), **Wake-Sleep Consolidated Learning**, [arXiv 2401.08623](https://arxiv.org/abs/2401.08623) | Explicit wake and sleep phases for continual learning | Design for the sleep phase |
| Tadros et al. (Bazhenov), **Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks**, Nature Communications 2022, [link](https://www.nature.com/articles/s41467-022-34938-7) | Offline sleep-like replay reduces forgetting | Supporting evidence for the sleep phase |

## Where this project can be novel

None of the reviewed work was found to do the following. Each still needs a targeted search
before we claim it as a first.

1. **Body and camera parameters inferred online as hidden causes on a manipulator.** No paper was
   found that treats camera, mass or kinematic calibration as free-energy minimisation on a robot
   arm. This directly fixes our measured camera-shift failure (5% success).
2. **Broken recipes as a trigger for causal inference.** Skill-library work assumes preconditions
   stay fixed; causal-RL work stops at structure learning. An agent that asks "which precondition
   changed?" when a known recipe fails, and then runs a confirming experiment, appears to be open.
3. **Neuromodulator-style gating as a working ML mechanism.** No clean implementation was found of
   acetylcholine-like precision gain plus noradrenaline-like surprise-gated learning rates inside
   a modern agent.
4. **A neural thinker trained to imitate a Bayesian cause-inference teacher and then surpass it**,
   inside an embodied active-inference agent. The parts exist (prior-fitted networks, HRM); the
   combination does not appear to.

## Technique choices (summary)

| Module | Choice | Fallback |
|---|---|---|
| Visual features | frozen DINOv2 | – |
| Objects | SlotContrast-style slots | simulator masks as supervision (rule 2) |
| World model | DINO-WM-style prediction over slots, 3–5 member ensemble | non-object DINO-WM |
| Planning | MPPI with the variational expected-free-energy objective (de Vries) | CEM |
| Amortisation | BMPC-style planner distillation | LePlanner |
| Curiosity | ensemble disagreement, compared with CIG | – |
| Recipes / relations | VisualPredicator-style predicates and operators | hand-written predicates from the simulator |
| Self-generated goals | MAGELLAN-style learning-progress prediction | uniform goal sampling |
| Thinker | HRM two-timescale core + recurrent-depth looping + learned halting | fixed-step recurrent net |
| Episodic memory | surprise-gated Titans-style writes + modern Hopfield retrieval | key–value memory with nearest-neighbour recall |
| Teacher → student | prior-fitted-network training on the Bayesian teacher | direct supervised distillation |
| Sleep | wake–sleep consolidation with replay | periodic replay fine-tuning |
