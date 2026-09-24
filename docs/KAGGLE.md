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

## Job C: many real episodes as thinker evidence (week 4), needs the world model

The week-4 gate trains the HRM thinker on real agent episodes; `slice_eval.py` gives only 100.
`scripts/collect_evidence.py` runs the same agent, disturbances and teacher on both GPUs (one
process per T4, each on its own shards of 50 episodes) and merges them into one evidence file.
It needs only `world_model.pt`, from the `pai-models` dataset of job B (**Add Input → Datasets →
pai-models**).

```python
PRE    = ""
SCRIPT = "scripts/collect_evidence.py"
ARGS   = ("--config configs/tabletop.yaml --episodes 200 "
          "slice.world_model=/kaggle/input/pai-models/world_model.pt")
```

- **Episodes:** `--episodes` is per condition, so 200 x 5 conditions = 1000 episodes. Env seeds
  10000+ (disjoint from `slice_eval.py`'s 1000+ and the calibration's 5000+).
- **Output:** `pai/runs/evidence/collected_evidence.npz` (the merged file), the shards
  `shard_*.npz`, `calibration.pkl` and `meta.json`. Train on it with
  `python scripts/train_thinker_real.py runs/evidence/collected_evidence.npz` (after downloading).
- **Time:** measured on the laptop (RTX 5060, 1 process): 30 episodes in 7.1 min = 255
  episodes/hour, after 8 calibration episodes. A T4 has not been timed yet; assuming a similar
  per-process rate (the planner is small, the simulator runs on the CPU), 1000 episodes on two
  processes take about 2 h. Each rank prints an ETA after every episode; if it projects well under
  the 12 h limit, `--episodes 400` doubles the data.
- **Resuming:** shards already written are skipped. Attach the stopped version's output and set
  `RESTORE_FROM` to its `pai` folder; the same `ARGS` then runs only the missing shards (a changed
  `--episodes`, `--chunk` or `--conditions` is refused: use a new `--out`).
- **Merging by hand:** `python scripts/collect_evidence.py --merge runs/evidence`. Two accounts
  running the same `ARGS` produce the *same* episodes (same seeds); to split work between accounts,
  add `--episode-seed 20000` to the second one's `ARGS` (10000 + 1000 episodes stays below it) and
  pass both merged files to `train_thinker_real.py`.

## Resuming a stopped training job

Training scripts resume from `ckpt_last.pt` in their output folder. In a new session, attach the
previous version's output (**Add Input → Your Work**) and set `RESTORE_FROM` to its `pai` folder.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| GPUs at 0% for a while at the start | the CPU step (`PRE`) is running | wait; progress lines show the ETA |
| `DistStoreError ... wait timeout` | an old notebook without `PRE` | re-import the notebook from the repo |
| Rendering errors mentioning EGL | headless OpenGL is unavailable | send the error text; a CPU fallback can be added |
