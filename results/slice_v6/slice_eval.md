# Vertical slice: on(red, plate)

100 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 100% | 100% | {'none': 20} |
| push | 100% | 35% | {'push': 7, 'slippery_object': 11, 'none': 2} |
| heavier_object | 95% | 70% | {'heavier_object': 14, 'none': 6} |
| slippery_object | 25% | 25% | {'none': 15, 'slippery_object': 5} |
| camera_shift | 100% | 100% | {'camera_shift': 20} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| none | 20 | 0 | 0 | 0 | 0 | 0 |
| push | 2 | 7 | 0 | 11 | 0 | 0 |
| heavier_object | 6 | 0 | 14 | 0 | 0 | 0 |
| slippery_object | 15 | 0 | 0 | 5 | 0 | 0 |
| camera_shift | 0 | 0 | 0 | 0 | 20 | 0 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 2 g
- camera offset: mean error 0.2 mm
