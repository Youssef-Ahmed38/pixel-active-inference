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
`data/paper_overleaf.zip`. Not yet updated with today's numbers (fall causes 92.5% and the speed
control, language v3, maze correction, branchy mazes, walker with solid walls and noise).

## Next steps (queue, in order)

1. Update `paper/numbers.tex` with the results above.
2. Fall causes: gait errors at practised speeds (now 9/48 called push); more seeds for slip.
3. Walking with wall contact: the walker falls in 24/240 noisy-sensing maze episodes (0 with exact
   sensing) because the policy never practised touching walls; train Stage 2 with wall contact. Then
   the agent's view from the robot's own camera instead of the grid sensor (MAZE.md 11).
4. Audit #4/#5 for the arm: learned cause likelihoods; key chain from recipes instead of a scripted
   state machine.
5. Chess: connect HRM to search with learned halting; more games per ladder rung.
6. Bike: learn from rendered video, not simulator states; Stage 2 balance transfer.

## Housekeeping

- Archived drafts (not in git): `data/archive/` (old chess drafts, a Kaggle job text, partial language
  files, the abandoned `pai-ceiling` work, the main checkout leftovers of 2026-10-28).
