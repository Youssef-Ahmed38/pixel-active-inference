# One-shot recipe reuse

Recipe learned from one successful episode (seed 2000); tested on 20 new layouts per object.

| object | agent | rollouts per step | task success | steps (successful) |
|---|---|---|---|---|
| red | full planner | 512 | 95% | 69 |
| red | small planner | 4 | 10% | 149 |
| red | small + recipe | 4 | 100% | 70 |
| blue | full planner | 512 | 90% | 88 |
| blue | small planner | 4 | 5% | 149 |
| blue | small + recipe | 4 | 100% | 60 |

```
Recipe for on(block, plate), learned from episode -1, reused 40x (40 successes):
  above_object hand at object + (+7, -7, +105) mm, ~11 steps
  at_object    hand at object + (-3, -4, +2) mm, ~50 steps
  grasped      hand at object_start + (-2, -1, +3) mm, ~1 steps
  lifted       hand at object_start + (+9, -16, +89) mm, ~7 steps
  over_target  hand at target + (+7, +2, +177) mm, ~11 steps
  lowered      hand at target + (-3, -1, +62) mm, ~14 steps
  opened       hand at object + (-3, +1, +28) mm, ~2 steps
  released     hand at object + (-1, +5, +71) mm, ~4 steps
  needs: hand open; object not already at the target
```
