# Status (2026-10-29)

The vision and the agent's design: [ROADMAP.md](ROADMAP.md). Results in detail: the per-stage reports
linked from the [README](../README.md).

## Results (final seeds)

| World | Key result | Report |
|---|---|---|
| 1 Arm | task 91%, cause 88%; learned cause likelihoods 97/100 (exact state), 78/100 (camera); recalibration after a camera shift 90% vs 20% | README, `docs/ARM_CAUSES.md` |
| 2 Walking (G1) | every gate on new seeds; wall practice lowers wall falls for all three training seeds (4, 9, 4 vs 10 of 40); fall causes 92.6% of 544 falls | `docs/LOCOMOTION_STAGE2.md` |
| 3 Maze | branchy mazes: curiosity 84 vs 67 of 100; own head camera and odometry 26/30; persistent misreads 49% vs 6-13%; walking at 0.4 m/s cuts maze falls from 2.60 to 0.46 per 1000 moves | `docs/MAZE.md` |
| 4 Doors and keys | 99.9% of mock locks; key chain without a script 38/40 unseen scenes | `docs/DISCOVERY.md` |
| 5 Chess | engine-anchor Elo 1037 [863, 1134] at 1 s/move | `docs/CHESS_STRENGTH.md` |
| 6 Language | causes named 41/50 per language; 83/100 on a larger set with hedged absence claims; correct Arabic composition | `docs/LANGUAGE_STAGE6.md` |
| 7 Bicycle (G1) | deliveries in new towns 37/40 vs 23/40 blind; perception audit passed; three training seeds of the cornering rule 120/120 routes; residual PPO 40/40 vs 31/40 on unseen corner shapes | `docs/BIKE_G1.md`, `docs/BIKE_TOWN.md` |
| Learned cause level | registered test: hierarchy 48.7% vs flat 44.5% (no advantage); explicit cause level 90.5% | `docs/THINKER_HIERARCHY.md` |

## In progress

- **Arm baselines:** domain-randomised PPO and recurrent PPO against the agent and the agent without
  cause-driven adaptation, same final episodes, three training seeds each (training).
- **GPU walking baseline:** the published G1 recipe (MuJoCo Playground, MJX/Warp physics, Brax PPO,
  200M steps) trained on the laptop GPU at about 31,000 steps/s; next, its evaluation on the exact CPU
  physics with the same gates as the agent's walker.
- **Paper:** rewritten around the perceptual hierarchy; kept outside the repository until submission.

## Next

1. The GPU walking baseline's CPU evaluation and comparison.
2. The arm baselines' final and comparison.
3. The G1's arms producing the bicycle steering; traffic in the towns.
4. Learning the cause structure from perception (the open problem).
