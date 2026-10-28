# Status: where we stopped (updated 2026-10-28, 04:00)

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
| 6 language | earlier build only; causal explanations in progress (below) | 456/456 held-out parses per language | `docs/LANGUAGE_STAGE6.md` |
| 7 bicycle | earlier build | practised policy 20/20 fresh starts; learns from simulator states, not video | `docs/BIKE_STAGE7.md` |

Audit fix #1 is done (arm relations come from perception, commit aa8d259); the other audit items
are open (see the queue).

## Running right now

Nothing. (Walking v3 is released; maze v3 is done; the Codex language task is reviewed.)

## Waiting for the user

- **PR #2** (Stage 6 causal explanations in English and Arabic, built by Codex, reviewed by Claude):
  https://github.com/Youssef-Ahmed38/pixel-active-inference/pull/2 . Merge it.
- The paper draft (`paper/`, IEEEtran in the user's format) is **local only** by the user's choice:
  `paper/` is in `.git/info/exclude`; numbers live in `paper/numbers.tex`; compile on Overleaf from
  `data/paper_overleaf.zip`. Keep it updated as results land; do not commit it.

## Next steps (queue, in order)

1. Language follow-up: in undisturbed episodes the surprise monitor never fires, so 8/10 abstain
   instead of saying "nothing unusual"; treat "no surprise" as evidence for `none`, re-test on new
   seeds (not 66000-66003, 67000-67009).
2. Fall-cause inference for walking (40% fresh: push and slip taken for own gait error); add sole
   forces as evidence. Good task to delegate to Codex.
3. Maze: use the learned walker in the maze (Stage 2 to 3); learn the spatial prior from own maps.
4. Audit #4/#5: learned cause likelihoods; key chain from recipes instead of a scripted state machine.
5. Chess: connect HRM to search with learned halting; more games per ladder rung.
6. Bike: learn from rendered video, not simulator states; Stage 2 balance transfer.

## Housekeeping

- Stale worktree `C:\projects\pai-ceiling` (branch `ceiling`, an aborted run from 2026-09-26) can be
  removed; its partial edits are superseded.
- Archived drafts (not in git): `data/archive/` (old chess drafts, a Kaggle job text, partial language
  files).
