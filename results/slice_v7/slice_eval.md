# Vertical slice: on(red, plate)

100 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 100% | 100% | {'none': 20} |
| push | 100% | 90% | {'push': 18, 'none': 2} |
| heavier_object | 90% | 90% | {'heavier_object': 18, 'push': 2} |
| slippery_object | 75% | 60% | {'push': 7, 'slippery_object': 12, 'none': 1} |
| camera_shift | 90% | 100% | {'camera_shift': 20} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| none | 20 | 0 | 0 | 0 | 0 | 0 |
| push | 2 | 18 | 0 | 0 | 0 | 0 |
| heavier_object | 0 | 2 | 18 | 0 | 0 | 0 |
| slippery_object | 1 | 7 | 0 | 12 | 0 | 0 |
| camera_shift | 0 | 0 | 0 | 0 | 20 | 0 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 2 g
- camera offset: mean error 0.3 mm
