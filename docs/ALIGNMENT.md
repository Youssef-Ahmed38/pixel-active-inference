# Arm and maze alignment

This implements the five alignment steps for the existing tabletop arm and maze work. It does
not require the unfinished Job A pixel run or Job C data collection. The arm still uses the
local trained entity world model and the local v7 evidence archive.

| Step | Implementation | Verified scope |
|---|---|---|
| Shared observation and action contract | `pai.cognition.contracts`, arm and maze adapters | Both domains produce entity and relation beliefs, skill commands, outcomes and surprise evidence. Body coordinates stay domain specific. |
| Shared planning objective | `pai.planning.objective.expected_free_energy` | Arm MPPI and maze frontier scoring use one pragmatic, risk and information-gain accounting function. Their world models and optimizers remain domain specific. |
| Shared surprise, cause and memory | `pai.cognition.causes`, `pai.cognition.memory` | Both domains use the same posterior representation and episode schema. Arm retains its calibrated 12-channel likelihood model; maze has rule-based collision and camera likelihoods. |
| Thinker on both domains | `pai.cognition.thinker_data`, `pai.cognition.thinker`, `scripts/cross_domain_thinker.py` | One HRM architecture trains on arm evidence and fine-tunes on maze evidence. The feature schema preserves body, contact and scene meaning and includes a domain bit. |
| Transfer ablation | `scripts/cross_domain_thinker.py` | Same initialization, maze training episodes and held-out maze episodes compare arm-pretrained and scratch HRM. |

The `CognitiveController` runs a shared goal → choose skill → execute → observe → record loop.
`scripts/alignment_eval.py` exercises it with the arm's `on(red, plate)` task and the maze's
hidden-goal task. The arm adapter uses `CameraFramePerception`, the trained world model and
MPPI, including the original 25-step settling check. The maze adapter uses the existing
frontier policy and continuous mobile skill. Advanced maze camera and passage disturbance
handling still runs in `pai.maze.advanced`; the arm's disturbance adaptation still runs in
`pai.agents.slice_agent`. Their domain-specific likelihoods and corrective actions are not
interchangeable.

## Evidence

The shared-loop check in `results/alignment_eval_5x5.json` succeeds on five arm placement seeds
and five maze goal seeds. This is a smoke check, not a disturbance or full-coverage evaluation.
The transfer pilot uses
100 existing arm episodes (60 train, 40 held out), plus 24 generated maze episodes (one train
and seven held out per cause). The three maze causes are no disturbance, camera shift and a
new passage block. Training and test splits are by whole episode. The maze input contains
observed movement stalls, unexpected collisions and localizer frame changes; it does not
contain the maze monitor's named cause reports or ground-truth event labels. Ground truth is
used only as the training target and score.

| Split seed | Arm held-out cause accuracy | Maze transfer, 40 steps | Maze scratch, 40 steps |
|---:|---:|---:|---:|
| 0 | 33/40 | 21/21 | 21/21 |
| 1 | 31/40 | 21/21 | 21/21 |
| 2 | 37/40 | 21/21 | 21/21 |

The corrected training loop retains Adam's state across learning-curve checkpoints and includes
all three examples in each tiny training batch. The old loop restarted Adam and sampled with
replacement, leaving the scratch model at 14/21 after 40 updates. The corrected scratch
model reaches 21/21 in all three splits. Arm pretraining can help at 5–10 updates, but the
advantage is gone by 20–40 updates on this dataset. See
`results/cross_domain_fewshot*.json` for the learning curves and per-cause counts. These
three splits reuse the same small set of generated maze episodes, so they are a pilot transfer
result, not independent replication across new maze families. The maze's visual decoder and
localizer are hand designed. The early-update improvement does not establish transfer of arm
control, learned dynamics or a general visual representation.

Reproduce with:

```powershell
.venv\Scripts\python.exe scripts/alignment_eval.py
.venv\Scripts\python.exe scripts/cross_domain_thinker.py --maze-per-cause 8 --maze-train-per-cause 1 --maze-steps 40
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.venv\Scripts\python.exe -m pytest -q --noconftest tests/test_cross_domain_thinker.py tests/test_cognition.py tests/test_maze.py tests/test_maze_advanced.py
```

The arm evaluation needs `runs/slice_world_model/world_model.pt`. The thinker needs
`results/slice_v7/slice_evidence.npz`. Neither is downloaded by these scripts.
