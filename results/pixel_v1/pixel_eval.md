# Vertical slice: on(red, plate)

100 episodes, objects perceived from the camera (DINOv2 + slots), learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 100% | 70% | {'camera_shift': 3, 'none': 14, 'unknown': 2, 'slippery_object': 1} |
| push | 85% | 80% | {'push': 16, 'none': 3, 'slippery_object': 1} |
| heavier_object | 85% | 70% | {'heavier_object': 14, 'slippery_object': 1, 'push': 3, 'unknown': 1, 'camera_shift': 1} |
| slippery_object | 25% | 55% | {'slippery_object': 11, 'push': 1, 'unknown': 5, 'none': 2, 'camera_shift': 1} |
| camera_shift | 90% | 20% | {'camera_shift': 4, 'none': 14, 'slippery_object': 1, 'unknown': 1} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| none | 14 | 0 | 0 | 1 | 3 | 2 |
| push | 3 | 16 | 0 | 1 | 0 | 0 |
| heavier_object | 0 | 3 | 14 | 1 | 1 | 1 |
| slippery_object | 2 | 1 | 0 | 11 | 1 | 5 |
| camera_shift | 14 | 0 | 0 | 1 | 4 | 1 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 3 g
- camera offset: mean error 66.7 mm
