# Evaluation suite: pixelai

## Success vs goal distance (clean)

| goal scale | start dist (m) | success | final dist (m) | steps to 2 cm |
|---|---|---|---|---|
| 0.1 | 0.053 | 1.00 | 0.003 | 19 |
| 0.2 | 0.073 | 1.00 | 0.005 | 26 |
| 0.35 | 0.107 | 1.00 | 0.006 | 56 |
| 0.5 | 0.122 | 0.80 | 0.009 | 102 |
| 1.0 | 0.230 | 0.35 | 0.028 | 199 |

## Robustness (goal scale 0.35)

| disturbance | success | final dist (m) | worst deviation after reaching (m) | steps outside after reaching |
|---|---|---|---|---|
| none | 1.00 | 0.006 | 0.022 | 12 |
| push | 1.00 | 0.006 | 0.034 | 32 |
| occlusion | 1.00 | 0.005 | 0.020 | 2 |
| lighting | 1.00 | 0.006 | 0.022 | 6 |
| camera_shift | 0.05 | 0.032 | 0.041 | 244 |
| payload | 1.00 | 0.006 | 0.022 | 12 |
| sensor_noise | 0.95 | 0.006 | 0.023 | 17 |

## Perception (static arm, belief starts 0.3 rad off per joint)

Final belief error: 0.0183 rad (norm over joints), 20 episodes.
