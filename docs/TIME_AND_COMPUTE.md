# Realistic time and compute (Claude writes the code; 2 Kaggle accounts + laptop)

Estimated 2026-09-23. GPU-hours are **T4-equivalent** and include a ~2x allowance for failed runs
and retries. Treat every number as a planning range, uncertain by up to 2x. Re-measure with
`scripts/benchmark.py` at the start of each stage and update this file.

## Weekly compute capacity

| Source | Assumption | T4-equiv GPU-h/week |
|---|---|---|
| Kaggle, 2 accounts | ~30 session-h/week each (check the current quota on each account), 2x T4 per session, ~65% useful (setup, failed runs, idle) | ~78 |
| Laptop RTX 5060 (8 GB) | ~40 h/week (nights and idle time); about one T4 for our PyTorch models (measured: 690 img/s on the PixelAI decoder) | ~40 |
| **Total** | | **~118** |

The laptop can't run JAX GPU physics (MJX) natively on Windows; use WSL2 or Kaggle for Stages 2
and 7. The two accounts must be used by their own owners (you and your teammate).

## GPU need per stage

| Stage | Main GPU cost | GPU-h | Weeks of full capacity |
|---|---|---|---|
| 1: arm sprint (8 weeks) | slots 15–40, world-model ensemble 30–80, HRM 10–30, demos 35–95, DINOv2 features 2–6 | 92–251 | 0.8–2.1 |
| 1: evaluation after deadline | ablations and seeds 40–100, model-based RL baseline 70–150, imitation baselines 60–140 | 170–390 | 1.4–3.3 |
| 0: infrastructure | MJX setup and benchmark | 5–15 | <0.2 |
| 2: walk and run | locomotion runs and sweeps 100–300, body model 15–40, fall causes 10–30 | 125–370 | 1.1–3.1 |
| 5: chess | self-play and engine matches 100–350, rule learning 10–30 | 110–380 | 0.9–3.2 |
| 3: maze | exploration and place memory | 50–150 | 0.4–1.3 |
| 4: doors and drawers | manipulation practice (hands are the risk) | 100–300 | 0.8–2.5 |
| 6: language | grounding and a small language model (LLM-generated conversations cost API credit, not GPU) | 30–100 | 0.3–0.8 |
| 7: bike | riding practice and learning from video | 100–300 | 0.8–2.5 |
| **Total** | | **~780–2,260** | **~7–19** |

## Calendar time

| Stage | Calendar weeks | GPU use of capacity |
|---|---|---|
| 1: arm sprint | 8 (fixed by the deadline) | 12–31 GPU-h/week: about 10–25% |
| 1: evaluation | 5–7 | 34–56 GPU-h/week: about 30–50% (the busiest period) |
| 0: infrastructure | 1–2 | negligible |
| 2: walk and run | 6–10 | with chess in parallel, the second-busiest period |
| 5: chess | in parallel with Stage 2 | run self-play mostly on the laptop, Stockfish on the laptop CPU |
| 3: maze | 3–5 | low |
| 4: doors and drawers | 5–10 | medium |
| 6: language | 6–10 | low |
| 7: bike | 8–14 | medium |
| **Everything** | **42–66 weeks from 2026-09-24** | **finish between mid-July and end of December 2027** |

For Stages 0–7 alone (after Stage 1): **29–51 weeks** calendar, needing **520–1,615 GPU-h**
against **~3,400–6,000 GPU-h** of capacity in that time. **Only ~15–27% of the compute gets used.**

## What actually sets the pace

1. **Research iteration, not GPUs.** Each experiment is a cycle: change → train (hours to a
   day) → look at results → decide. Walking or bike riding may need 20–40 such cycles. With
   Claude writing the changes in minutes, a cycle is about a day, and cycles add up to weeks.
2. **Your review time.** You must understand each module to defend it. Budget ~1 hour a day;
   this is the real limit on how fast modules can be accepted.
3. **Claude usage limits.** Long sessions and many parallel agents hit session limits (it
   happened on 2026-09-23). The plan lives in files, so work resumes after a reset.
4. **Kaggle session mechanics.** 12-hour sessions, queueing and restarts make runs lumpy.
   Resumable checkpoints (already built) make this manageable.
5. **Hard research problems.** Walking learned by the agent's own objective, humanoid hands, and
   bike riding carry real risk. Their fallbacks (in [BUILD_PLAN_STAGES_2_7.md](BUILD_PLAN_STAGES_2_7.md))
   keep the schedule from collapsing.

## Consequences for the plan

- **Compute is not the bottleneck.** Two Kaggle accounts plus the laptop are enough for
  everything, with room to spare. No paid GPUs are needed, as long as the quotas stay as assumed.
- **Keep GPUs busy while building.** Queue the next training run before starting the next piece
  of code, so iteration and compute overlap.
- **Two busy periods:** post-deadline evaluation, and Stages 2 + 5 together. Put the chess
  self-play on the laptop then.
- **The realistic finish for everything is mid-July to end of December 2027**, not a few weeks.
  Stage 1 by the application deadline is on track: its compute needs are small.
