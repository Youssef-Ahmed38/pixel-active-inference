# Vertical slice: on(red, plate)

60 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 90% | 95% | {'none': 19, 'push': 1} |
| push | 90% | 90% | {'push': 18, 'none': 2} |
| heavier_object | 75% | 95% | {'heavier_object': 19, 'push': 1} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | unknown |
|---|---|---|---|---|
| none | 19 | 1 | 0 | 0 |
| push | 2 | 18 | 0 | 0 |
| heavier_object | 0 | 1 | 19 | 0 |
