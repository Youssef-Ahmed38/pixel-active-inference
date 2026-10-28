# Brief: the bicycle learned from video and ridden from its own camera (Stage 7)

You are helping build **Pixel Active Inference** (a PhD-application research project): one agent
learns several skills in simulation with the same principles (learned models, observation learning,
practice, learned rather than scripted, honest evaluation). This task is Stage 7, the bicycle.

## 1. Read first

| File | What it is |
|---|---|
| `docs/BIKE_STAGE7.md` | Everything so far: the MuJoCo bicycle and rider, a classical expert, observation learning from expert *motion states* via inverse dynamics, practice (20/20, 20/20, 19/20 over three optimizer seeds), and a **hybrid** vision interface (camera only for frame orientation; the supplied root position tracks the camera and other states come from the simulator): 12/20, later a defined hybrid gate passed |
| `pai/bike/env.py`, `observation_learning.py`, `practice.py`, `vision.py`, `vision_control.py`, `vision_teacher.py` | Code |
| `scripts/bike_*.py` | Evaluation and training scripts |

## 2. The task

1. **Learn from video, not states.** The student learns from rendered video of the expert riding
   (pixels only; the expert's states and controls are used only to audit, never to fit), e.g. by
   learning to estimate the bike's lean, steering and speed from frames, then inferring the
   expert's actions from the estimated motion, as the state-based observation learning does now.
2. **Ride from its own senses.** At test time the policy sees only an on-board camera (mounted on the
   bike or rider, not a camera that tracks the simulator's root position) and the rider's own
   proprioception (e.g. wheel speed, steering angle, an IMU-like lean rate), no simulator states.
3. **Practice** may follow, as in `practice.py`, using only the same senses.
4. **Honest evaluation.** Development seeds for all fitting; final once on new layouts never used for
   the bike (see the seeds in `docs/BIKE_STAGE7.md`), at least 20 starts, against the classical expert
   and the existing hybrid and state-based students on the same starts, with Wilson 95% intervals.
   Report what is learned and what is given (the dynamics model for inverse dynamics is given now;
   say if that remains).

## 3. Rules

- Work in the main checkout on `main`. **Do not create branches, stash, reset or commit**; Claude
  reviews and commits.
- Touch only: `pai/bike/*`, new `scripts/bike_video_*.py`, `tests/test_bike*.py`, a new section in
  `docs/BIKE_STAGE7.md`, new `results/bike_video_*` files, new `docs/media/bike/*` (videos < 3 MB).
  Results files larger than 5 MB: gzip-compressed. **Never record absolute local paths** (user
  folders, caches) in any result file: use project-relative paths or a placeholder.
- **Do not touch** walking, maze, arm, chess, discovery or language files: other work is in progress.
  The laptop GPU is shared with humanoid training; keep batch sizes modest.
- Keep passing: `tests/test_bike*.py`.

## 4. Deliverables

Code and fast unit tests; the new documentation section (design, learned vs given, results table
with intervals, failures, seeds, exact commands, a short video of a ride from the on-board camera);
`results/bike_video_*` with per-episode records. Finish with a short report: files, commands, seeds,
numbers, what is not done.
