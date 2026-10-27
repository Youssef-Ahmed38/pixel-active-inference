"""Generate animated GIF demonstration of the Active Inference maze explorer.

Renders three side-by-side panels:
1. Ground truth maze (viewer's perspective)
2. Agent's Bayesian belief map, entropy uncertainty, and chosen EFE target
3. Local sensor field-of-view and real-time EFE metrics

Saves to docs/media/maze/maze_v2_efe.gif (< 3 MB) and sidecar JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.efe_agent import ActiveInferenceExplorer
from pai.maze.env import MazeWorld, VISUAL_COLORS, VisualObservation
from pai.maze.map_model import ObservationModel, SpatialPrior


def render_belief_panel(agent: ActiveInferenceExplorer, world: MazeWorld) -> np.ndarray:
    """Render the agent's probabilistic belief map, entropy, and planned target."""
    size = agent.size
    panel = np.zeros((size, size, 3), dtype=np.uint8)

    for r in range(size):
        for c in range(size):
            b = agent.map.beliefs[r, c]  # [P(wall), P(floor), P(goal)]
            ent = agent.map.cell_entropy((r, c))

            if (r, c) not in agent.map.observed_cells:
                # Unobserved cell: dark slate blue tinted by prior
                panel[r, c] = (25, 30, 45)
            elif b[0] > 0.6:
                # Believed wall
                panel[r, c] = (70, 75, 85)
            elif b[2] > 0.4:
                # Believed goal
                panel[r, c] = (245, 190, 35)
            else:
                # Believed floor: soft beige, slightly dimmed if uncertain
                brightness = max(0.5, 1.0 - 0.3 * ent)
                panel[r, c] = (int(205 * brightness), int(205 * brightness), int(195 * brightness))

    # Mark visited path
    for p in agent.map.visits:
        if 0 <= p[0] < size and 0 <= p[1] < size:
            panel[p] = (140, 205, 160)

    # Mark planned route if available
    if agent.target and agent.target != world.position:
        route = agent.map.route(world.position, agent.target)
        for s in route[:-1]:
            if 0 <= s[0] < size and 0 <= s[1] < size:
                panel[s] = (235, 130, 45)  # orange route line

    # Mark chosen target frontier
    if agent.target and 0 <= agent.target[0] < size and 0 <= agent.target[1] < size:
        panel[agent.target] = (60, 215, 245)  # cyan target marker

    # Mark agent position
    pos = world.position
    if 0 <= pos[0] < size and 0 <= pos[1] < size:
        panel[pos] = (245, 40, 215)  # magenta agent position

    return panel


def render_truth_panel(world: MazeWorld) -> np.ndarray:
    """Render simulator ground truth for the viewer."""
    size = world.size
    panel = np.empty((size, size, 3), dtype=np.uint8)
    panel[:] = VISUAL_COLORS["floor"]
    panel[world.grid.astype(bool)] = VISUAL_COLORS["wall"]

    for p in world.visited:
        panel[p] = (155, 215, 170)

    panel[world.goal] = VISUAL_COLORS["goal"]
    panel[world.position] = (245, 40, 215)
    return panel


def render_local_panel(world: MazeWorld, obs) -> np.ndarray:
    """Render local camera view / local sensory FOV."""
    size = world.size
    local_panel = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw local field of view on black background
    r0, c0 = world.position
    for (r, c), kind in obs.visible.items():
        # Center in panel
        pr = size // 2 + (r - r0)
        pc = size // 2 + (c - c0)
        if 0 <= pr < size and 0 <= pc < size:
            local_panel[pr, pc] = VISUAL_COLORS.get(kind, (150, 150, 150))

    local_panel[size // 2, size // 2] = (245, 40, 215)  # agent center
    return local_panel


def render_frame(world: MazeWorld, agent: ActiveInferenceExplorer, obs, scale: int = 8) -> np.ndarray:
    left = render_truth_panel(world)
    mid = render_belief_panel(agent, world)
    right = render_local_panel(world, obs)

    # Scale up with nearest neighbor
    left_scaled = np.repeat(np.repeat(left, scale, 0), scale, 1)
    mid_scaled = np.repeat(np.repeat(mid, scale, 0), scale, 1)
    right_scaled = np.repeat(np.repeat(right, scale, 0), scale, 1)

    gap = np.full((left_scaled.shape[0], 6, 3), 15, dtype=np.uint8)
    return np.concatenate([left_scaled, gap, mid_scaled, gap, right_scaled], axis=1)


def generate_demo_gif(seed: int = 50000, size: int = 21, budget: int = 220, every: int = 2,
                      models_dir: Path | None = None,
                      out_path: Path = ROOT / "docs" / "media" / "maze" / "maze_v2_efe.gif"):
    world = MazeWorld(seed=seed, size=size)

    obs_model = None
    transfer_prior = None
    if models_dir and (models_dir / "observation_model.json").exists():
        obs_model = ObservationModel.load(models_dir / "observation_model.json")
    if models_dir and (models_dir / "spatial_prior.json").exists():
        transfer_prior = SpatialPrior.load(models_dir / "spatial_prior.json")

    agent = ActiveInferenceExplorer(
        sensor_range=world.sensor_range, size=size, seed=seed,
        obs_model=obs_model, prior=transfer_prior,
        epistemic_weight=1.0, effort_weight=0.5, goal_weight=5.0,
    )

    obs = world.observe()
    frames = []
    efe_logs = []

    for step in range(budget):
        if step % every == 0:
            frames.append(render_frame(world, agent, obs))

        move = agent.act(obs)
        if move is None:
            break

        efe_logs.append({
            "step": step,
            "position": world.position,
            "target": agent.target,
            "efe": agent.last_efe_score,
            "epistemic": agent.last_epistemic,
            "pragmatic": agent.last_pragmatic,
            "effort": agent.last_effort,
        })

        obs = world.step(move)
        if obs.reached_goal:
            # Add final celebratory frame
            frames.append(render_frame(world, agent, obs))
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=0.08, loop=0)

    # First and last frame stills
    imageio.imwrite(out_path.with_name(out_path.stem + "_first.png"), frames[0])
    imageio.imwrite(out_path.with_name(out_path.stem + "_last.png"), frames[-1])

    sidecar = out_path.with_suffix(".json")
    truth = world.truth()
    sidecar_data = {
        "panels": ["Ground Truth (Viewer)", "Probabilistic Belief & Target (EFE)", "Local Field of View"],
        "seed": seed,
        "steps": world.steps,
        "reached_goal": bool(world.position == world.goal),
        "coverage": float(truth["coverage"]),
        "frames": len(frames),
        "efe_history": efe_logs,
    }
    sidecar.write_text(json.dumps(sidecar_data, indent=2))
    print(f"Generated demo GIF: {out_path} ({len(frames)} frames, {out_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--budget", type=int, default=220)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "runs" / "maze_models")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "maze" / "maze_v2_efe.gif")
    args = parser.parse_args()

    generate_demo_gif(
        seed=args.seed, size=args.size, budget=args.budget, every=args.every,
        models_dir=args.models_dir, out_path=args.out,
    )


if __name__ == "__main__":
    main()
