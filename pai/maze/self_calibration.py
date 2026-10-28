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
    """Run practice episodes; return per-cell look counts and body labels (no simulator truth)."""
    looks: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))
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
                    looks[(s, cell)][KIND_TO_IDX[kind]] += 1
        anchors[(s, obs.position)] = KIND_TO_IDX["goal" if obs.reached_goal else "floor"]
    return dict(looks), anchors


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
    looks, anchors = collect_practice(seeds, make_env, make_agent)
    model, prior = dawid_skene(looks, anchors)
    info = {"cells": len(set(looks) | set(anchors)), "looks": int(sum(v.sum() for v in looks.values())),
            "body_labels": len(anchors), "class_frequencies": prior.round(4).tolist()}
    return model, info
