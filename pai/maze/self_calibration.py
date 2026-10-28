"""Learn the maze sensor model from the agent's own experience, without simulator labels.

During practice in training mazes the agent logs every noisy look it takes at every cell. Its body
supplies a few certain labels: a cell it stands on is floor (goal when it reports reaching the
goal) and a cell it bumps into is a wall. A Dawid-Skene EM then fits P(seen | true) and the class
frequencies, treating each cell's type as latent and the body labels as known.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pai.maze.env import MazeWorld
from pai.maze.map_model import KIND_TO_IDX, ObservationModel


def collect_practice(seeds: list[int], make_env, make_agent, steps: int = 160):
    """Run practice episodes; return every look (with its viewpoint) and body labels (no simulator truth)."""
    views: dict[tuple, list[tuple]] = defaultdict(list)   # (seed, cell) -> [(viewpoint, seen index)]
    anchors: dict[tuple, int] = {}
    for s in seeds:
        env = make_env(s)
        agent = make_agent(s)
        obs = env.observe()
        for _ in range(steps):
            here = obs.position
            anchors[(s, here)] = KIND_TO_IDX["goal" if obs.reached_goal else "floor"]
            bumped = None
            move = agent.act(obs)
            if move is None:
                break
            obs = env.step(move)
            if obs.collided:
                bumped = (here[0] + move[0], here[1] + move[1])
                anchors[(s, bumped)] = KIND_TO_IDX["wall"]
            for cell, kind in obs.visible.items():
                if cell != bumped:  # a bump is touch, not a look
                    views[(s, cell)].append((obs.position, KIND_TO_IDX[kind]))
        anchors[(s, obs.position)] = KIND_TO_IDX["goal" if obs.reached_goal else "floor"]
    return dict(views), anchors


def repeat_look_weight(views: dict[tuple, list[tuple]]) -> tuple[float, dict]:
    """How much new evidence a repeat look from the same viewpoint carries, from disagreement rates.

    If repeat looks from one spot disagree as often as looks from different spots, they are
    independent (weight 1); if they never disagree, they only repeat the first reading (weight 0).
    """
    same = [0, 0]
    diff = [0, 0]
    for looks in views.values():
        first_from: dict = {}
        for vantage, seen in looks:
            if vantage in first_from:
                same[0] += seen != first_from[vantage]
                same[1] += 1
            else:
                for other, other_seen in first_from.items():
                    diff[0] += seen != other_seen
                    diff[1] += 1
                first_from[vantage] = seen
    d_same = same[0] / max(same[1], 1)
    d_diff = diff[0] / max(diff[1], 1)
    weight = float(np.clip(d_same / d_diff, 0.0, 1.0)) if d_diff > 0 else 1.0
    return weight, {"repeat_pairs": same[1], "repeat_disagreement": round(d_same, 4),
                    "viewpoint_pairs": diff[1], "viewpoint_disagreement": round(d_diff, 4)}


def weighted_counts(views: dict[tuple, list[tuple]], weight: float) -> dict[tuple, np.ndarray]:
    """Per-cell label counts; the first look from each viewpoint counts 1, repeats count ``weight``."""
    counts = {}
    for key, looks in views.items():
        c = np.zeros(3)
        seen_from = set()
        for vantage, seen in looks:
            c[seen] += weight if vantage in seen_from else 1.0
            seen_from.add(vantage)
        counts[key] = c
    return counts


def dawid_skene(looks: dict[tuple, np.ndarray], anchors: dict[tuple, int],
                init: ObservationModel | None = None, iters: int = 100,
                alpha: float = 1.0) -> tuple[ObservationModel, np.ndarray]:
    """EM for a shared confusion matrix; returns the fitted model and class frequencies."""
    keys = sorted(set(looks) | set(anchors))
    counts = np.array([looks.get(k, np.zeros(3)) for k in keys])        # (n, 3) seen-label counts
    fixed = np.array([anchors.get(k, -1) for k in keys])
    matrix = (init or ObservationModel()).matrix.copy()                    # [seen, true]
    prior = np.full(3, 1 / 3)
    for _ in range(iters):
        log_post = np.log(prior)[None, :] + counts @ np.log(matrix)       # (n, 3) over true type
        log_post -= log_post.max(1, keepdims=True)
        post = np.exp(log_post)
        post /= post.sum(1, keepdims=True)
        post[fixed >= 0] = np.eye(3)[fixed[fixed >= 0]]
        expected = counts.T @ post + alpha                                  # (seen, true)
        matrix = expected / expected.sum(0, keepdims=True)
        prior = (post.sum(0) + alpha) / (post.sum() + 3 * alpha)
    return ObservationModel(matrix=matrix), prior


def self_calibrated_sensor(seeds: list[int], size: int, make_env, make_agent) -> tuple[ObservationModel, dict]:
    """Sensor model and repeat-look weight from practice; ``info['repeat_weight']`` goes to the map."""
    views, anchors = collect_practice(seeds, make_env, make_agent)
    weight, stats = repeat_look_weight(views)
    looks = weighted_counts(views, weight)
    model, prior = dawid_skene(looks, anchors)
    info = {"cells": len(set(looks) | set(anchors)), "looks": int(sum(len(v) for v in views.values())),
            "body_labels": len(anchors), "class_frequencies": prior.round(4).tolist(),
            "repeat_weight": round(weight, 4), **stats}
    return model, info
