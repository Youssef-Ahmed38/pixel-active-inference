# Vertical slice: on(red, plate)

60 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 85% | 100% | {'none': 20} |
| push | 85% | 85% | {'none': 3, 'push': 17} |
| heavier_object | 75% | 95% | {'heavier_object': 19, 'none': 1} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | unknown |
|---|---|---|---|---|
| none | 20 | 0 | 0 | 0 |
| push | 3 | 17 | 0 | 0 |
| heavier_object | 1 | 0 | 19 | 0 |
