# Vertical slice: on(red, plate)

100 episodes, objects perceived from the camera (DINOv2 + slots), learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 85% | 65% | {'unknown': 5, 'none': 13, 'slippery_object': 1, 'camera_shift': 1} |
| push | 100% | 85% | {'unknown': 2, 'push': 17, 'none': 1} |
| heavier_object | 65% | 55% | {'unknown': 6, 'slippery_object': 1, 'heavier_object': 11, 'none': 2} |
| slippery_object | 30% | 40% | {'slippery_object': 8, 'unknown': 8, 'none': 1, 'camera_shift': 3} |
| camera_shift | 90% | 10% | {'unknown': 3, 'camera_shift': 2, 'none': 14, 'slippery_object': 1} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| none | 13 | 0 | 0 | 1 | 1 | 5 |
| push | 1 | 17 | 0 | 0 | 0 | 2 |
| heavier_object | 2 | 0 | 11 | 1 | 0 | 6 |
| slippery_object | 1 | 0 | 0 | 8 | 3 | 8 |
| camera_shift | 14 | 0 | 0 | 1 | 2 | 3 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 3 g
- camera offset: mean error 21.9 mm
