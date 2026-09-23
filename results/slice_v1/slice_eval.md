# Vertical slice: on(red, plate)

60 episodes, privileged object state, learned entity world model, MPPI planning.

| condition | task success | cause accuracy | inferred causes |
|---|---|---|---|
| none | 85% | 95% | {'none': 19, 'push': 1} |
| push | 80% | 95% | {'push': 19, 'none': 1} |
| heavier_object | 65% | 85% | {'heavier_object': 17, 'push': 3} |

Confusion (rows: true cause, columns: inferred):

| true \ inferred | none | push | heavier_object | unknown |
|---|---|---|---|---|
| none | 19 | 1 | 0 | 0 |
| push | 1 | 19 | 0 | 0 |
| heavier_object | 0 | 3 | 17 | 0 |
