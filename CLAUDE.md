# Pixel Active Inference: instructions for a new session

**Start here:** read `docs/STATUS.md` (where we stopped, what is running, the next steps). Do not
re-analyse the whole project; the status file and the docs it links are up to date. If the file
`data/HANDOFF_PRIVATE.md` exists (local only, not in git), read it too: it has the GitHub push
procedure and the Codex thread details.

## The project in five lines

A PhD-application research project: **one agent** that learns many skills in simulation (MuJoCo)
with the same principles, extending PixelAI (Sancaktar et al., 2020): active inference (world
models, expected free energy, surprise, Bayesian hidden-cause inference, curiosity), episodic
memory and recipes, an HRM "thinker", learned rather than scripted, evaluated honestly.
Stages: 1 arm manipulation, 2 humanoid walking/running, 3 maze, 4 doors/drawers/keys, 5 chess,
6 language (English/Arabic), 7 bicycle. Plan: `docs/ROADMAP.md`; results: `README.md`.

## Working rules (from the user)

- **One task at a time.** No parallel multi-agent workflows (credit cost).
- **Training runs on this laptop's GPU** (RTX 5060, 8 GB), never on Kaggle.
- **Delegate implementation to Codex when sensible** with `tools/codex_delegate.py` and supervise it
  (review diff, tests, recompute numbers from raw results, fresh seeds, no ground-truth leakage),
  then commit and open a PR; the user merges PRs.
- **Honest results:** final numbers on fresh seeds never used in development, frozen parameters,
  exact commands; keep failures and negative results; say what is learned and what is given.
- **Media:** record GIF/MP4/PNG of every milestone into `docs/media/<topic>/` (< 3 MB) and the README.
- **Commits:** the author is set in the repo's local git config; never add AI co-author lines; before
  every commit scan the staged diff for personal data (names, emails, hostnames, local user paths);
  commit only specific paths (other agents may have uncommitted files in the folder); never switch
  the main checkout's branch. Pushing needs the account switch in `data/HANDOFF_PRIVATE.md`.

## Map

| What | Where |
|---|---|
| Current state, running jobs, next steps | `docs/STATUS.md` |
| Plan and stage gates | `docs/ROADMAP.md`, `docs/SPRINT.md`, `docs/BUILD_PLAN_STAGES_2_7.md` |
| Independent audit of earlier work (what to fix) | `docs/AUDIT_OUTSIDE_BUILDS.md` |
| Per-stage reports | `docs/LOCOMOTION_STAGE2.md`, `docs/MAZE.md`, `docs/DISCOVERY.md`, `docs/CHESS_*.md`, `docs/LANGUAGE_STAGE6.md`, `docs/BIKE_STAGE7.md` |
| Task briefs for delegation | `docs/briefs/` |
| Python | `.venv\Scripts\python.exe` (Windows; Git Bash available) |
