# Vertical slice: on(red, plate)

40 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 100% | 88% | {'none': 7, 'push': 1} |
| push | 100% | 100% | {'push': 8} |
| heavier_object | 100% | 88% | {'heavier_object': 7, 'push': 1} |
| slippery_object | 50% | 50% | {'slippery_object': 4, 'push': 2, 'unknown': 1, 'none': 1} |
| camera_shift | 88% | 100% | {'camera_shift': 8} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| none | 7 | 1 | 0 | 0 | 0 | 0 |
| push | 0 | 8 | 0 | 0 | 0 | 0 |
| heavier_object | 0 | 1 | 7 | 0 | 0 | 0 |
| slippery_object | 1 | 2 | 0 | 4 | 0 | 1 |
| camera_shift | 0 | 0 | 0 | 0 | 8 | 0 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 2 g
- camera offset: mean error 0.2 mm
