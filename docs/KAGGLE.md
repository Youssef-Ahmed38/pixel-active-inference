# Running jobs on Kaggle (two accounts)

Every job uses the same notebook, [notebooks/kaggle_run.ipynb](../notebooks/kaggle_run.ipynb). It
clones this repository, installs the package, fetches the robot model, optionally runs a CPU step
(`PRE`), and then runs `SCRIPT` with `ARGS` on every GPU (`torchrun` when there are two).

**Each account is used only by its owner.** You run jobs on your account; your teammate runs
jobs on theirs.

## Notebook settings (every job)

1. Download `notebooks/kaggle_run.ipynb` from the repository and import it
   (**Create → New Notebook → File → Import Notebook**). Re-import after the notebook changes in the repo.
2. Right-hand panel: **Accelerator → GPU T4 x2**, **Internet → On** (needs phone verification).
3. Set `PRE`, `SCRIPT` and `ARGS` in the settings cell as below.
4. **Save Version → Save & Run All**. The job keeps running when the browser is closed. Everything
   under `/kaggle/working` becomes the version's output.
5. Progress lines with an ETA are printed for every long step.

## Job A: object slots (week 2), no inputs needed

```python
PRE    = "python scripts/collect_frames.py --config configs/tabletop.yaml --episodes 300 --workers 4 --out /tmp/frames"
SCRIPT = "scripts/week2_slots.py"
ARGS   = "--config configs/tabletop.yaml --episodes 300 --workers 4 slots.frames_dir=/tmp/frames slots.cache_dir=/tmp/features slots.out_dir=runs/slots"
```

- **Output:** `pai/runs/slots/slots.pt` and `log.jsonl`.
- **Time:** frames 10–20 min on the CPUs, features ~5 min, then slot training on both GPUs.

## Job B: the agent from camera images (week 3), needs two model files

The agent needs the slot model (from job A) and the world model (trained on the laptop). Share them
with a private Kaggle Dataset:

1. On the account that has the files: **Datasets → New Dataset**, upload `slots.pt` (from job A's
   output) and `world_model.pt` (from `runs/slice_world_model/` on the laptop), and name it e.g.
   `pai-models`. Keep it **private**.
2. To let the other account use it: the dataset's **Settings → Sharing → add collaborator**
   (your teammate's Kaggle username).
3. In the notebook that runs job B: **Add Input → Datasets → pai-models**. It appears under
   `/kaggle/input/pai-models/`.

```python
PRE    = ""
SCRIPT = "scripts/slice_eval.py"
ARGS   = ("--config configs/tabletop.yaml --tag pixel --no-gif "
          "--slots /kaggle/input/pai-models/slots.pt slice.world_model=/kaggle/input/pai-models/world_model.pt")
```

- **Output:** `pai/results/pixel_eval.md`, `pixel_eval.json`, `pixel_reports.md`.
- **GPUs:** this job uses one. It is an evaluation, not training, so the second T4 idles; running
  it on the laptop is equally fine.

## Resuming a stopped training job

Training scripts resume from `ckpt_last.pt` in their output folder. In a new session, attach the
previous version's output (**Add Input → Your Work**) and set `RESTORE_FROM` to its `pai` folder.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| GPUs at 0% for a while at the start | the CPU step (`PRE`) is running | wait; progress lines show the ETA |
| `DistStoreError ... wait timeout` | an old notebook without `PRE` | re-import the notebook from the repo |
| Rendering errors mentioning EGL | headless OpenGL is unavailable | send the error text; a CPU fallback can be added |
