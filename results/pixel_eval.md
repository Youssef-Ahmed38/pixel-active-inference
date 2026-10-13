# Vertical slice: on(red, plate)

100 episodes, objects perceived from the camera (DINOv2 + slots), learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 85% | 85% | {'none': 17, 'heavier_object': 1, 'camera_shift': 1, 'unknown': 1} |
| push | 85% | 100% | {'push': 20} |
| heavier_object | 85% | 60% | {'push': 2, 'unknown': 4, 'heavier_object': 12, 'none': 2} |
| slippery_object | 35% | 75% | {'slippery_object': 15, 'push': 3, 'none': 1, 'unknown': 1} |
| camera_shift | 85% | 10% | {'camera_shift': 2, 'slippery_object': 5, 'none': 13} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| none | 17 | 0 | 1 | 0 | 1 | 1 |
| push | 0 | 20 | 0 | 0 | 0 | 0 |
| heavier_object | 2 | 2 | 12 | 0 | 0 | 4 |
| slippery_object | 1 | 3 | 0 | 15 | 0 | 1 |
| camera_shift | 13 | 0 | 0 | 5 | 2 | 0 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 6 g
- camera offset: mean error 70.2 mm
