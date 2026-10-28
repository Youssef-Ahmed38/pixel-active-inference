# Status: where we stopped (updated 2026-10-28, 02:10)

## Done (on `main`, pushed)

| Stage | State | Key result (fresh seeds) | Report |
|---|---|---|---|
| 1 arm, exact state | done | v7: task 91%, cause 88%; camera-shift recalibration 90% vs 20% without | README, `results/slice_v7` |
| 1 arm, from camera | done (gate > 50%) | task 75%, cause 66% (slippery block 35% is the weak spot) | `results/pixel_v3` |
| Perception (slots) | done | object IoU 0.68, position 7.8 mm mean | `results/slots/joba_log.jsonl` |
| Thinker (HRM) gate | passed | 96.0% agreement with the Bayesian teacher (transformer 95.8%) | `results/thinker/real_c1000.md` |
| 2 walking (G1) | v1 released | stand, walk 10 m, headings, run: 20/20 each; pushes 93% at 0.5 m/s, 53% at 1.0 m/s; fall causes 59% (dev) | `docs/LOCOMOTION_STAGE2.md` |
| 3 maze | v3: route EFE + noisy sensing | goal 79% vs nearest frontier 60% (p=0.003), ties info-gain baseline 78%; noisy: no rule wins (2nd set); sensor model self-learned | `docs/MAZE.md` §10 |
| 4 doors/keys | discovery agent | mock 99.9% solved (111.6 actions) vs novelty 80.9%; physics 12/25 with decoys | `docs/DISCOVERY.md` |
| 5 chess | done (PR #1 reviewed and merged, 2026-10-28) | 1037 conditional engine-anchor Elo [863, 1134] at 1 s/move; beats old player 17-2-1 | `docs/CHESS_STRENGTH.md` |
| 6 language | earlier build only; causal explanations in progress (below) | 456/456 held-out parses per language | `docs/LANGUAGE_STAGE6.md` |
| 7 bicycle | earlier build | practised policy 20/20 fresh starts; learns from simulator states, not video | `docs/BIKE_STAGE7.md` |

Audit fix #1 is done (arm relations come from perception, commit aa8d259); the other audit items
are open (see the queue).

## Running right now (check these first)

1. **v3 walking training** (laptop GPU): started from v1, tracking weighted higher, pushes 0.9 then
   1.25 m/s. Chain: `data/run_v3.sh` (phase A `configs/locomotion_v3a.yaml`, phase B `..._v3b.yaml`,
   then a dev check on seeds 52000+). Logs: `runs/locomotion/train_v3a.log`, `train_v3b.log`,
   status `runs/locomotion/train_v3.status`, dev result `results/locomotion/dev_v3.json`.
   At iteration 3000 (phase A): survival 84%, tracking held.
   **Next:** if the dev check keeps v1's headings/run and improves pushes, run the final gates on
   NEW fresh seeds (`scripts/locomotion_eval.py ... --seed-offset 4000`, i.e. 54000+; 50000+ and
   52000+ are already used), record a video, add v3 to `docs/LOCOMOTION_STAGE2.md`, commit the two
   v3 configs with it.
2. **Codex task `language-causes`** (Stage 6 causal explanations, brief
   `docs/briefs/LANGUAGE_CAUSES.md`), running in the user's Codex thread in the main checkout.
   Logs: `data/codex_tasks/language-causes/events.jsonl`, report `.../report.md` when finished.
   **Next:** review it (diff in `pai/language/`, tests, recompute scores from `results/language_*`,
   fresh seeds, no event-log/ground truth in the agent's words), send feedback with
   `python tools/codex_delegate.py resume --task language-causes --feedback <file> --wait`, or commit
   the reviewed paths on a branch and open a PR.

## Waiting for the user

Nothing at the moment.

## Next steps (queue, in order)

1. Finish the two running items above.
2. Fall-cause inference for walking (59%: push and slip taken for own gait error); e.g. sole forces.
3. Maze: make the epistemic term matter (learned map uncertainty), learn the prior from the agent's
   own exploration instead of true maps; use the learned walker in the maze (Stage 2 to 3).
4. Audit #4/#5: learned cause likelihoods; key chain from recipes instead of a scripted state machine.
5. Chess: connect HRM to search with learned halting; more games per ladder rung.
6. Bike: learn from rendered video, not simulator states; Stage 2 balance transfer.
7. Delegate suitable items to Codex (`tools/codex_delegate.py`, one at a time) and supervise.

## Housekeeping

- Stale worktree `C:\projects\pai-ceiling` (branch `ceiling`, an aborted run from 2026-09-26) can be
  removed; its partial edits are superseded.
- Archived drafts (not in git): `data/archive/` (old chess drafts, a Kaggle job text, partial language
  files).
