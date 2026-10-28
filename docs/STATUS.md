# Status: where we stopped (updated 2026-10-28, 04:30)

## Done (on `main`, pushed)

| Stage | State | Key result (fresh seeds) | Report |
|---|---|---|---|
| 1 arm, exact state | done | v7: task 91%, cause 88%; camera-shift recalibration 90% vs 20% without | README, `results/slice_v7` |
| 1 arm, from camera | done (gate > 50%) | task 75%, cause 66% (slippery block 35% is the weak spot) | `results/pixel_v3` |
| Perception (slots) | done | object IoU 0.68, position 7.8 mm mean | `results/slots/joba_log.jsonl` |
| Thinker (HRM) gate | passed | 96.0% agreement with the Bayesian teacher (transformer 95.8%) | `results/thinker/real_c1000.md` |
| 2 walking (G1) | v3 released | stand, walk 10 m, headings, run: 20/20 each; pushes 100% at 1.0 m/s, 93% at 1.25 m/s; fall causes 40% (fresh) | `docs/LOCOMOTION_STAGE2.md` |
| 3 maze | v3: route EFE + noisy sensing | goal 79% vs nearest frontier 60% (p=0.003), ties info-gain baseline 78%; iid noise: no rule wins; persistent misreads: learned repeat weight + touch 47% vs 9-13% (p<1e-6); sensor self-learned | `docs/MAZE.md` §10 |
| 4 doors/keys | discovery agent | mock 99.9% solved (111.6 actions) vs novelty 80.9%; physics 12/25 with decoys | `docs/DISCOVERY.md` |
| 5 chess | done (PR #1 reviewed and merged, 2026-10-28) | 1037 conditional engine-anchor Elo [863, 1134] at 1 s/move; beats old player 17-2-1 | `docs/CHESS_STRENGTH.md` |
| 6 language | causal explanations v2 (Codex, reviewed; PR) | causes 37/50 per language (37/40 committed), grounded cause words 5/5, unseen question wording 9/12, instructions 41/50 = arm baseline | `docs/LANGUAGE_STAGE6.md` |
| 7 bicycle | earlier build | practised policy 20/20 fresh starts; learns from simulator states, not video | `docs/BIKE_STAGE7.md` |

Audit fix #1 is done (arm relations come from perception, commit aa8d259); the other audit items
are open (see the queue).

## Running right now (check these first; updated 2026-10-28 04:30)

1. **Language follow-up on PR #2's branch** (`language-causes`, in a separate worktree; see the
   local hand-off file for its location and setup).
   New `SilenceModel` in `pai/language/association.py`: a silent surprise monitor is learned evidence
   for "nothing unusual" (P = 0.957 from teaching: silent 11/15 clean, 0/60 disturbed). Checkpoint
   `results/language_causes_v3_checkpoint.json` (teaching adds the v2 final episodes). Fresh eval:
   seeds 68000-68009, disturbance base 79000 -> `results/language_causes_v3_final.json`, log
   `data/lang_v3_eval.log`. **Next:** recompute cause accuracy per condition from the raw episodes
   (clean episodes should now say `none`), add a v3 section to `docs/LANGUAGE_STAGE6.md`, commit the
   new files on the branch (not `third_party`, not `runs`), push, update the PR description (no
   Claude mentions). The user merges PR #2.
2. **Codex task `fall-causes`** (brief `docs/briefs/FALL_CAUSES.md`, Sol 6.1, user's thread, main
   checkout). Logs `data/codex_tasks/fall-causes/events.jsonl`; report `.../report.md` when done.
   **Next:** review (diff in `pai/locomotion/falls.py` or new files, tests, recompute from
   `results/locomotion/fall_causes_*.json`, new seeds, no simulator forces/event log in the inference),
   then commit on a branch via a worktree and open a PR. Keep the Codex thread closed in the app while
   it runs (an open thread blocks the CLI: "active writer").
3. **Maze spatial prior** (main checkout, uncommitted: `pai/maze/map_model.py` per-cell unobserved
   information cache fix + `SpatialPrior.fit_from_beliefs`; `pai/maze/efe_agent.py` prior for the
   baselines; `scripts/maze_v3_eval.py --prior default|truth|self`). Dev (20000-20029, self prior):
   route EFE exact 83%, noisy 70%. Final running on seeds 73000-73099 for the three priors ->
   `results/maze_v3_prior_{default,truth,self}.json`, logs `data/maze_dev/prior_*_final.log`.
   **Next:** compare, write `docs/MAZE.md` section 10.3 (note: the cache fix changes the info values,
   so earlier v3 numbers were with the old cache), commit, push.

## Waiting for the user

- **PR #2**: wait for item 1 above before merging.
- Copilot's automatic PR review comes from the user's personal Copilot setting (cannot be removed by
  us). PR descriptions must not mention Claude.
- The paper draft (`paper/`, IEEEtran in the user's format) is **local only** by the user's choice:
  `paper/` is in `.git/info/exclude`; numbers live in `paper/numbers.tex`; compile on Overleaf from
  `data/paper_overleaf.zip`. Keep it updated as results land; do not commit it.

## Next steps (queue, in order)

1. Finish the three running items above.
2. Maze: use the learned walker in the maze (Stage 2 to 3).
3. Audit #4/#5: learned cause likelihoods; key chain from recipes instead of a scripted state machine.
4. Chess: connect HRM to search with learned halting; more games per ladder rung.
5. Bike: learn from rendered video, not simulator states; Stage 2 balance transfer.

## Housekeeping

- Stale worktree `C:\projects\pai-ceiling` (branch `ceiling`, an aborted run from 2026-09-26) can be
  removed; its partial edits are superseded.
- Archived drafts (not in git): `data/archive/` (old chess drafts, a Kaggle job text, partial language
  files).
