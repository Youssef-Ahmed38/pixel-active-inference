# Current build status — 10 October 2026

This records runnable evidence rather than marking the whole roadmap complete.
Jobs A/C are excluded from this work.

| Stage | Implemented and checked | Unfinished research gate |
|---|---|---|
| 5 chess | Learned legal rules, complete state transitions, endings, optional draw claims, linear/HRM/contextual value candidates, alpha-beta and MCTS search, bounded quiescence, replay/self-play update, candidate validation, Stockfish matches and PGN continuation | Strong playing policy; accepted self-play gain; reliable rating/time curve; learned chess halting; optional physical board |
| 6 language | Learned English/Arabic lexicon/templates; bidirectional meanings; scene questions; observed narration; token attribute learning; shared single/ordered goals; physical arm execution; recorded-scene conversation teaching and evidence audit | Causal explanations; speech; broad disturbed-task generalization |
| 7 bicycle | Free MuJoCo bicycle and simplified rider; balance baseline; expert video and motion recordings; known-model inverse actions; observed-motion student; practice; three paired optimizer replicates; RGB orientation models; hybrid image-attitude balance | Full video motion extraction; Stage 2 transfer; independent-layout generalization; starts from rest and disturbances; full rider/pedals |

## Latest evidence

- Chess correctness: previous fresh suites and search checks remain linked in
  `CHESS_RULES.md`, `CHESS_STATE.md`, `CHESS_EVALUATION.md`, `CHESS_SEARCH.md`.
- Chess strength: **0 wins, 0 draws, 12 losses**, no execution failures. No finite
  measured Elo point estimate. Initial self-play candidate rejected on fresh value
  validation. See `CHESS_PLAY.md`.
- Language: **456/456** held-out parses and realizations per language; **84/84**
  goals resolved per language; **4/4** bilingual physical sequences and **8/8**
  final goal relations on two layouts. See `LANGUAGE_STAGE6.md`.
- Bike: observed-motion student **10/10** easy fresh rides. Diverse-start teaching
  produced **17/20** on a harder holdout. After practice, a separate paired holdout
  improved **7/10 → 10/10**; scratch **1/10 → 0/10**. See `BIKE_STAGE7.md`.
- Combined focused suite: **132 tests passed** before the last two practice and
  decision-boundary checks; the latest new-module suite passed **31/31**. This is
  not a rerun of every simulator evaluation or a proof that all roadmap work passes.
- Fresh MCTS comparison: legality and execution checks passed on 8/8 boards at
  each budget. At 128 nodes, MCTS avoided 1/4 queen traps versus alpha-beta 4/4.
  Alpha-beta remains the incumbent. See `CHESS_MCTS.md`.
- New chess value candidates: HRM did not beat the original value; contextual
  value reduced fresh MAE from **337.5 to 315.4 cp**, but regressed on the paired
  tactical checks. Neither replaces the incumbent. See `CHESS_VALUE_UPDATES.md`.
- A new paired two-game chess trial: baseline **two losses**, contextual candidate
  **one draw and one loss**. The draw was completed after preserving and resuming
  the 256-ply game history. This small sample does not establish a strength gain.
- Recorded conversations: **118/118** held-out rows per language parsed,
  roundtripped and matched recorded evidence. All four collection layouts
  completed their two-goal arm sequences.
- Replicated bike practice on twenty shared fresh layouts: observation-initialized
  policies **20/20, 20/20, 19/20** versus scratch **2/20, 1/20, 0/20**. These are
  optimizer replicates, not sixty independent physical layouts.
- Latest combined focused suite: **151 tests passed**. The subsequent quiescence
  reference suite passed **3/3**; these are reported separately.
- Search extensions: separate leaf/ordering values, optional prior-PV ordering,
  shared multi-seed comparisons and match interruption recovery are built. The
  new chess focused suite passed **17 tests**, followed by **14 tests** after
  schedule recovery and ordering-teaching exclusion were added. At 128 nodes,
  all four search variants passed 15/15 shared tactical boards on seeds 710–711;
  neither ordering experiment establishes a strength gain.
- Explicit uncertain language: separate English/Arabic hypothesis models passed
  **483/483** held-out parses and realizations per language. The new combined
  language suite passed **27 tests**. Physical causal verification remains open.
- Bicycle RGB perception: static teaching failed on moving rides, then separate
  motion teaching improved all three orientation channels on five fresh rides.
  Hybrid image attitude plus sensor speed passed **5/5** paired fresh starts.
  Supplied camera tracking, full motion extraction and Stage 2 transfer remain
  separate open gates. See `BIKE_STAGE7.md`.
- The larger visual-control check passed only **12/20** against sensor baseline
  **13/20**. On another twenty starts, the existing practiced sensor policy passed
  **20/20**, but its visual hybrid passed **0/20**. Visual control is therefore an
  experimental branch with a failed generalization gate, not a replacement.
- Teaching a separate image model on the practiced policy's own motion then
  passed **19/20** visual hybrid rides on new starts (940–959), against **20/20**
  for the practiced sensor policy. The remaining visual failure is preserved.
- The broader focused selection passed **143/144** tests and exposed a Windows
  search clock-resolution issue. Both planners now use a high-resolution clock;
  the failing check and four ordering/deadline checks subsequently passed.
  This is not a rerun of the entire project suite.
- Finer RGB features and forty new teaching rides subsequently passed **20/20**
  visual hybrid rides on unseen starts (1100–1119), matching **20/20** for the
  practiced sensor policy. This passes the specified hybrid benchmark. External
  speed/joint encoders and supplied camera tracking remain explicit dependencies.

## Next build order

1. Make chess strength improve under an independently validated value/search
   update; extend the existing chess HRM branch and preserve exact-rule gates.
2. Verify language explanations against observed cause evidence and test
   instructions under disturbances.
3. Extend the successful bicycle RGB attitude interface to full motion extraction
   and new camera/appearance conditions; build the actual learned Stage 2 walking
   controller before claiming Stage 2-to-bicycle transfer. The current maze walker
   is a planar gait simulator, not that learned humanoid controller.

Artifacts and reports preserve failures, unfinished games, rejected candidates
and separate trained policies. Do not replace these results with an unsupported
“100% complete” label.
