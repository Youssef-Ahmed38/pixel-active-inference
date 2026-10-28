# Status: where we stopped (updated 2026-10-28, 06:45)

## Results

| Stage | State | Key result (fresh seeds) | Report |
|---|---|---|---|
| 1 arm, exact state | done | v7: task 91%, cause 88%; camera-shift recalibration 90% vs 20% without | README, `results/slice_v7` |
| 1 arm, from camera | done (gate > 50%) | task 75%, cause 66% (slippery block 35% is the weak spot) | `results/pixel_v3` |
| Perception (slots) | done | object IoU 0.68, position 7.8 mm mean | `results/slots/joba_log.jsonl` |
| Thinker (HRM) gate | passed | 96.0% agreement with the Bayesian teacher (transformer 95.8%) | `results/thinker/real_c1000.md` |
| 2 walking (G1) | v3 released; learned fall causes (PR #3, merged) | stand, walk 10 m, headings, run: 20/20 each; pushes 100% at 1.0 m/s, 93% at 1.25 m/s; fall causes learned from own senses 92.5% vs 40.6% (gait recall down to 81%) | `docs/LOCOMOTION_STAGE2.md` |
| 3 maze | v3 + learned G1 walker in the maze, solid walls (PR #4, merged) | exact sensing: info-seeking not significantly better than nearest frontier (corrected after a bug fix); persistent misreads: learned repeat weight + touch 49% vs 6-13%; walker = grid outcome 30/30 mazes, 0 falls; branchy mazes (100 fresh): route EFE 84 vs nearest frontier 67 (p = 0.006), and with persistent misreads the learned repeat weight beats counting repeats under every prior; the walker matches the grid body (30/30); solid walls: 0 falls, same outcomes as visual walls on 60 paired mazes; with noisy sensing the persistent-misread result holds with the G1 walking (14 vs 5, p = 0.004) but it falls in 24/240 episodes; five-panel perception video | `docs/MAZE.md` §10-11 |
| 4 doors/keys | discovery agent | mock 99.9% solved (111.6 actions) vs novelty 80.9%; physics 12/25 with decoys | `docs/DISCOVERY.md` |
| 5 chess | done (PR #1, merged) | 1037 conditional engine-anchor Elo [863, 1134] at 1 s/move; beats old player 17-2-1 | `docs/CHESS_STRENGTH.md` |
| 6 language | causal explanations v3 (PR #2, merged) | cause correct 41/50 per language (clean 9/10; v2 explainer on the same episodes 33/50, p = 0.008); 456/456 held-out parses | `docs/LANGUAGE_STAGE6.md` |
| 7 bicycle | earlier build | practised policy 20/20 fresh starts; learns from simulator states, not video | `docs/BIKE_STAGE7.md` |

Audit fix #1 is done (arm relations come from perception, commit aa8d259); audit #4 is done for
walking (PR #3); the other audit items are open (see the queue).

Everything is merged into `main` (PRs #1-#4). Nothing is running; no PR is open.

## Session log, 2026-10-28 (a second Claude session, in a worktree, while the main session's account was switched)

1. Language v3 (PR #2): silent surprise monitor as learned evidence. Fresh seeds 68000-68009: cause
   correct 41/50 per language, clean 9/10; the frozen v2 explainer on the same episodes 33/50
   (paired 8/0, p = 0.008). Repaired 69 v2 Wilson intervals an earlier dash fix had corrupted.
2. Fall causes (Codex task, PR #3): reviewed and recomputed, 92.5% vs 40.6% on fresh falls (seed
   55006). Matched-speed control (56006/56007): speed does not separate push from slip, but slip
   recall varies by seed (58/60 vs 47/60; 160/180 pooled).
3. Maze (PR #4): spatial-prior comparison and the cache-fix correction (route EFE vs nearest
   frontier on corridor mazes is not significant); branchy mazes (100 fresh: 84 vs 67, p = 0.006);
   the G1 walker on branchy mazes; the five-panel perception video; solid walls (60 paired mazes:
   0 falls, outcomes unchanged); noisy sensing with the walker (persistent misreads 14 vs 5,
   p = 0.004, but 24 falls in 240 walked episodes). Details in `docs/MAZE.md` section 11.
4. Merged PRs #2, #3, #4 (the user asked; #4 needed main merged in: README rows and this file).
   Cleaned the main checkout: its 60 leftover copies were all older versions of merged files (each
   checked against the history), archived anyway in `data/archive/main_checkout_leftovers_20261028.tar`,
   then removed; `main` pulled to the merge of #4. Removed the merged language worktree
   (`C:\projects\pai-lang`) and the stale `C:\projects\pai-ceiling` worktree; the latter's uncommitted
   work is kept in `data/archive/pai_ceiling_uncommitted_20261028.patch` (tracked edits, on branch
   `ceiling` at cadd001) and `pai_ceiling_untracked_20261028.tar` (results/discovery_v3, home_doors).

Running the walker or the G1 tests from another worktree needs `third_party/` and `runs/` from the
main checkout as directory junctions (`cmd //c "mklink /J third_party C:\projects\pixel-active-inference\third_party"`,
same for `runs`); remove a junction with `cmd //c rmdir <path>` (never a recursive delete, which
would follow it into the real folder).

Copilot's automatic PR review comes from the user's personal Copilot setting (cannot be removed by
us). PR descriptions must not mention Claude.

The paper draft (`paper/`, IEEEtran in the user's format) is **local only** by the user's choice:
`paper/` is in `.git/info/exclude`; numbers live in `paper/numbers.tex`; compile on Overleaf from
`data/paper_overleaf.zip`. Updated 2026-10-28 06:50 with all results above (maze section rewritten
around the branchy-maze result and the corridor correction; walker figure added).

## In progress (2026-10-28 11:10; the user is away: work continues without asking)

- **Open PRs for the user to merge:** #5 (Stage 4 key chain as an induced recipe, Codex, reviewed)
  and #6 (walking v4 with wall contact, four fall causes, the maze walker's own eyes; plus the v5
  negative result). Codex's original Stage 4 files are still untracked in the main checkout
  (identical; remove after merging #5, then pull). Branch `walking-walls-eyes` has a worktree at
  `../pai-walls`.
- **v5 not released:** fast-changing goals kept every gate but raised development maze falls
  (13 vs 8 of 60). v4 stays released.
- **Smooth steering for the maze walker** (main checkout, uncommitted: `pai/maze/walker.py`
  `smooth=True`, `scripts/maze_walker_eval.py --smooth`): start walking below 10 degrees heading
  error, keep walking until 40, ramp speed at most 1 m/s^2. Development (20100-20129, v4): falls 8 -> 2
  of 60, goals route EFE 14 -> 16, nearest frontier 2 -> 8. Final running on new mazes 83000-83029:
  `results/maze_walker_persistent_83000_v4{_smooth,}.json`. Then add to PR #6 (docs: MAZE.md
  section 11/12 and LOCOMOTION_STAGE2.md v4 paragraph).
- **Codex: arm cause likelihoods** (`docs/briefs/ARM_CAUSE_LIKELIHOODS.md`, task `arm-causes`):
  stopped once on "model at capacity" during its camera final; resumed with
  `data/codex_tasks/arm-causes/continue1.md`. Review when it reports.

## Next steps (plan agreed with the user, 2026-10-28; in order)

Division of work: **all walking work is Claude's**; Codex works on other stages (now: Stage 4 key
chain as a learned recipe, brief `docs/briefs/KEY_CHAIN_RECIPES.md`, CPU only, no walking files).

**A. Walking with wall contact (Claude).** Started: optional solid training walls in
`pai/locomotion/body.py` (`add_training_walls`, `place_walls`) and `EnvConfig.walls/wall_frac/
wall_radius` in `pai/locomotion/env.py` (off by default; uncommitted, nothing trained yet).
1. Wall test in `scripts/locomotion_eval.py`: 40 bodies walk at 0.6 m/s toward a solid wall
   0.8-1.5 m ahead, keep pushing, then back off (as the maze walker does); falls and contact. Run on
   v3 first (the "before"). If v3 rarely falls there, find the real cause of the maze falls first.
2. Train v4 from v3 (~1 h, laptop GPU): about half of the 32 dynamics variants get four solid walls
   1-3 m from the start; the rest keep open-ground practice (avoid v2's forgetting). No new sensors,
   no wall reward.
3. Development check: every existing gate must still pass (stand, walk 10 m, headings, run,
   pushes); the wall test should improve. If a skill degrades, a short corrective phase (as v3 C).
4. Final, once, on new seeds: all gates + wall test + the walker in noisy mazes on new mazes (v3:
   24/240 falls), v3 vs v4 on the same mazes.
5. Video of a wall bump and recovery; walking and maze reports, README, local paper; PR.
   Release rule: v4 only if every existing gate holds; otherwise keep v3 and report it.

**B. Fall-cause refinement (Claude), on the released walker (after A, so it is fitted once).**
1. Own gait errors called push (9/48): find the features; add own-sense evidence (jolt timing vs own
   leg motion; jolt starting at the torso vs at a foot).
2. Slip recall that varies by seed (58/60 vs 47/60): make the evidence and learned likelihoods hold
   across several development seeds.
3. A fourth cause, **hit a wall** (sudden stop while the legs keep driving), from own senses only.
4. Refit likelihoods on development falls of the released walker; final on several new seeds with
   seed-grouped intervals; per-cause results vs the current and the old method on the same falls.
5. Reports, local paper, PR.

**C. The maze walker sees through its own eyes (Claude; recommended next after A and B).** Today the
maze agent's view still comes from the grid sensor and its position from the simulator. Next, its
map comes from the robot's own head camera and its position from its own odometry (with drift).
That removes the last two pieces handed to it for free in Stages 2+3, and is the strongest claim
for the application: **from pixels and proprioception to an explored maze, with no ground truth.**
The noisy- and persistent-sensor work (learned sensor model, learned repeat-look weight, touch as
an epistemic action) is the preparation for this, since real camera readings are noisy and
persistent.

**Expected achievement after A and B** (to be confirmed by the final tests): a humanoid that keeps
all its skills and survives wall contact; a maze walker that rarely falls; falls explained from its
own senses (gait, push, slip, wall) across seeds; Stage 2 complete as part of the one agent.

**After C, in order:**
1. Arm audit (Stage 1): learned cause likelihoods instead of authored ones.
2. Chess (Stage 5): the HRM thinker inside search, with learned halting; more games per ladder rung.
3. Bicycle (Stage 7): learn from rendered video instead of simulator states; reuse walking balance.
4. Final paper numbers for every stage (local, the user's format).
Codex: Stage 4 key chain (running), then further non-walking tasks; Claude reviews each and opens a PR.

## Housekeeping

- Leftover empty folders `C:\projects\pai-keychain`, `pai-chess-review`, `pai-codex-language-causes` (worktrees already removed; delete when unlocked).
- Archived drafts (not in git): `data/archive/` (old chess drafts, a Kaggle job text, partial language
  files, the abandoned `pai-ceiling` work, the main checkout leftovers of 2026-10-28).
