# Vertical slice: on(red, plate)

60 episodes, privileged object state, learned entity world model, MPPI planning, explanations do not change behaviour (ablation).

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| heavier_object | 95% | 95% | {'heavier_object': 19, 'push': 1} |
| slippery_object | 45% | 40% | {'slippery_object': 8, 'push': 9, 'none': 3} |
| camera_shift | 20% | 80% | {'camera_shift': 16, 'push': 4} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | slippery_object | camera_shift | unknown |
|---|---|---|---|---|---|---|
| heavier_object | 0 | 1 | 19 | 0 | 0 | 0 |
| slippery_object | 3 | 9 | 0 | 8 | 0 | 0 |
| camera_shift | 0 | 4 | 0 | 0 | 16 | 0 |

Estimates when the cause was named correctly:

- extra mass: mean absolute error 3 g
- camera offset: mean error 0.2 mm
