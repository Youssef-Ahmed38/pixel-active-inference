# Vertical slice: on(red, plate)

60 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 100% | 100% | {'none': 20} |
| push | 100% | 95% | {'push': 19, 'none': 1} |
| heavier_object | 95% | 95% | {'heavier_object': 19, 'push': 1} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | unknown |
|---|---|---|---|---|
| none | 20 | 0 | 0 | 0 |
| push | 1 | 19 | 0 | 0 |
| heavier_object | 0 | 1 | 19 | 0 |
