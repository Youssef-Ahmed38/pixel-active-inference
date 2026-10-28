# Status: where we stopped (updated 2026-10-28, 06:00)

## Results

| Stage | State | Key result (fresh seeds) | Report |
|---|---|---|---|
| 1 arm, exact state | done | v7: task 91%, cause 88%; camera-shift recalibration 90% vs 20% without | README, `results/slice_v7` |
| 1 arm, from camera | done (gate > 50%) | task 75%, cause 66% (slippery block 35% is the weak spot) | `results/pixel_v3` |
| Perception (slots) | done | object IoU 0.68, position 7.8 mm mean | `results/slots/joba_log.jsonl` |
| Thinker (HRM) gate | passed | 96.0% agreement with the Bayesian teacher (transformer 95.8%) | `results/thinker/real_c1000.md` |
| 2 walking (G1) | v3 released; fall causes in PR #3 | stand, walk 10 m, headings, run: 20/20 each; pushes 100% at 1.0 m/s, 93% at 1.25 m/s; fall causes learned from own senses 92.5% vs 40.6% (gait recall down to 81%) | `docs/LOCOMOTION_STAGE2.md` |
| 3 maze | v3 + learned walker in the maze (PR #4) | exact sensing: info-seeking not significantly better than nearest frontier (corrected after a bug fix); persistent misreads: learned repeat weight + touch 49% vs 6-13%; walker = grid outcome 30/30 mazes, 0 falls; branchy mazes: route EFE 28/30 vs nearest frontier 19/30 (p = 0.012), same with the walker; five-panel perception video | `docs/MAZE.md` §10-11 |
| 4 doors/keys | discovery agent | mock 99.9% solved (111.6 actions) vs novelty 80.9%; physics 12/25 with decoys | `docs/DISCOVERY.md` |
| 5 chess | done (PR #1 merged) | 1037 conditional engine-anchor Elo [863, 1134] at 1 s/move; beats old player 17-2-1 | `docs/CHESS_STRENGTH.md` |
| 6 language | causal explanations in PR #2 (v3) | cause correct 41/50 per language (clean 9/10; v2 explainer on the same episodes 33/50, p = 0.008); 456/456 held-out parses | `docs/LANGUAGE_STAGE6.md` |
| 7 bicycle | earlier build | practised policy 20/20 fresh starts; learns from simulator states, not video | `docs/BIKE_STAGE7.md` |

Audit fix #1 is done (arm relations come from perception, commit aa8d259); audit #4 is done for
walking (PR #3); the other audit items are open (see the queue).

## Nothing is running. Waiting for the user: three PRs to merge

- **PR #2** `language-causes` (Stage 6 v3, silent monitor as evidence; also repairs 69 v2 intervals
  that an earlier dash fix had corrupted). Ready.
- **PR #3** `fall-causes` (Codex task reviewed: numbers recomputed from the per-fall records, no
  leakage, model frozen before the final falls). Matched-speed control done (seeds 56006/56007):
  speed is not what separates push from slip, but slip recall varies by seed (160/180 pooled).
- **PR #4** `maze-walker` (spatial-prior comparison, cache-fix correction, G1 walker in the maze,
  branchy-maze evaluation and the five-panel perception video, this status file).

**After merging, clean the main checkout before `git pull`:** it still holds the same files as
uncommitted copies (the maze files staged, Codex's fall-cause files and the language files
untracked), and git refuses to overwrite untracked files. From the main checkout:
`git restore --staged .` then delete the copies listed by `git status` that the PRs add, then pull.
Then remove the language worktree (see the local hand-off file).

Copilot's automatic PR review comes from the user's personal Copilot setting (cannot be removed by
us). PR descriptions must not mention Claude.

The paper draft (`paper/`, IEEEtran in the user's format) is **local only** by the user's choice:
`paper/` is in `.git/info/exclude`; numbers live in `paper/numbers.tex`; compile on Overleaf from
`data/paper_overleaf.zip`. Not yet updated with today's numbers (fall causes 92.5%, language v3,
maze correction).

## Next steps (queue, in order)

1. Update `paper/numbers.tex` with the results above.
2. Fall causes: gait errors at practised speeds (now 9/48 called push); more seeds for slip.
3. Maze: branchy mazes with noisy sensing and the learned priors (`scripts/maze_v3_eval.py
   --maze-style branchy`, dev 20000-20029, final e.g. 78000+); walker upgrades: solid walls
   (wall/foot contact pairs), the agent's view from the robot's own camera instead of the grid sensor.
4. Audit #4/#5 for the arm: learned cause likelihoods; key chain from recipes instead of a scripted
   state machine.
5. Chess: connect HRM to search with learned halting; more games per ladder rung.
6. Bike: learn from rendered video, not simulator states; Stage 2 balance transfer.

## Housekeeping

- Stale worktree `C:\projects\pai-ceiling` (branch `ceiling`, an aborted run from 2026-09-26) can be
  removed; its partial edits are superseded.
- Archived drafts (not in git): `data/archive/` (old chess drafts, a Kaggle job text, partial language
  files).
