"""Evaluation episodes for PixelAI-style tasks.

task="reach":      start at q0, goal given as an image of the arm at q_goal (or as q_goal in
                   joints mode). Success = end-effector within `success_ee_dist` at the end.
task="perception": arm is static (k_a = 0), belief starts at q + offset and must converge to
                   the true configuration from vision + proprioception (PixelAI's perception test).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pai.agents import PixelAIAgent
from pai.envs import PandaEnv, build_disturbances


def _sample_pair(env: PandaEnv, rng, goal_scale=None, min_ee_dist: float = 0.05):
    """Start/goal configurations. goal_scale=None samples both independently from the
    workspace box; a number s puts the goal at q0 + s * U(-half_range, half_range), which
    controls how far the goal is (PixelAI's pixel-error gradient is only informative locally)."""
    half = np.asarray(env.cfg.sample_half_range, float)[list(env.cfg.active_joints)]
    lo, hi = env.home_qpos[env.qpos_idx] - half, env.home_qpos[env.qpos_idx] + half
    while True:
        q0 = env.sample_q(rng)
        qg = env.sample_q(rng) if goal_scale is None else np.clip(q0 + goal_scale * rng.uniform(-half, half), lo, hi)
        if np.linalg.norm(env.ee_pos_at(q0) - env.ee_pos_at(qg)) >= min_ee_dist:
            return q0, qg


def run_episode(env: PandaEnv, agent: PixelAIAgent, cfg, q0, qg, task: str = "reach",
                record: bool = False, perception_offset: float = 0.3, rng=None) -> dict:
    ev = cfg.eval
    obs = env.reset(q0=q0)  # restores nominal camera/lights, so the goal image is undisturbed
    goal_img = env.render_at(qg)
    goal_ee = env.ee_pos_at(qg)
    start_ee_dist = float(np.linalg.norm(obs["ee_pos"] - goal_ee))
    mu0 = None
    if task == "perception":
        rng = rng or np.random.default_rng(0)
        mu0 = obs["q"] + rng.choice([-1, 1], size=obs["q"].shape) * perception_offset
    agent.reset(obs["q"], goal_image=goal_img, goal_q=qg, mu0=mu0)

    trace = {"ee_dist": [], "belief_err": [], "free_energy": [], "e_v": []}
    frames, t_success, t0 = [], None, time.time()
    for t in range(int(ev.steps)):
        want_pred = record and t % 2 == 0
        a, info = agent.step(obs, return_prediction=want_pred)
        if task == "perception":
            a = np.zeros_like(a)
        true_q = env.data.qpos[env.qpos_idx]
        if want_pred:
            frames.append(_frame(obs["image"], goal_img, info.prediction))
        obs = env.step(a)
        d = float(np.linalg.norm(obs["ee_pos"] - goal_ee))
        trace["ee_dist"].append(d)
        trace["belief_err"].append(float(np.linalg.norm(info.mu - true_q)))
        trace["free_energy"].append(info.free_energy)
        trace["e_v"].append(info.e_v)
        if t_success is None and d < ev.success_ee_dist:
            t_success = t
    wall = time.time() - t0
    after = np.asarray(trace["ee_dist"][t_success:]) if t_success is not None else np.array([])
    result = {
        "start_ee_dist": start_ee_dist,
        "final_ee_dist": trace["ee_dist"][-1],
        "final_belief_err": trace["belief_err"][-1],
        "success": trace["ee_dist"][-1] < ev.success_ee_dist,
        "steps_to_success": t_success,
        # Robustness: how far the arm is knocked off the goal after first reaching it, and how
        # many control steps it spends outside the success radius before the episode ends.
        "max_dev_after_success": float(after.max()) if after.size else None,
        "steps_outside_after_success": int((after >= ev.success_ee_dist).sum()) if after.size else None,
        "control_hz": ev.steps / wall,
        "trace": trace,
    }
    if record:
        result["frames"] = frames
    return result


def _frame(obs_img, goal_img, pred):
    size = obs_img.shape[0]
    if pred is not None and pred.shape[0] != size:
        f = size // pred.shape[0]
        pred = np.repeat(np.repeat(pred, f, 0), f, 1)
    panels = [obs_img, goal_img] + ([pred] if pred is not None else [])
    return np.concatenate(panels, axis=1)


def evaluate(cfg, decoder=None, device="cpu", task: str = "reach", tag: str = "pixelai") -> dict:
    ev = cfg.eval
    env = PandaEnv(cfg.env, disturbances=build_disturbances(cfg.env.get("disturbances", [])))
    agent_cfg = cfg.agent
    if task == "perception":
        agent_cfg = type(cfg.agent)({**cfg.agent, "k_a": 0.0})
    agent = PixelAIAgent(agent_cfg, decoder, device)
    rng = np.random.default_rng(cfg.seed)
    episodes = []
    for i in range(int(ev.episodes)):
        q0, qg = _sample_pair(env, rng, ev.get("goal_scale"), ev.get("min_goal_ee_dist", 0.05))
        if task == "perception":
            qg = q0  # the goal prior then agrees with the true pose and does not bias perception
        episodes.append(run_episode(env, agent, cfg, q0, qg, task, record=(i == 0 and ev.save_gif), rng=rng))
    env.close()

    out_dir = Path(ev.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = episodes[0].pop("frames", None)
    if frames:
        import imageio

        imageio.mimsave(out_dir / f"{tag}_{task}.gif", frames, duration=0.04, loop=0)

    summary = {
        "tag": tag,
        "task": task,
        "episodes": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "start_ee_dist_median": float(np.median([e["start_ee_dist"] for e in episodes])),
        "final_ee_dist_mean": float(np.mean([e["final_ee_dist"] for e in episodes])),
        "final_ee_dist_median": float(np.median([e["final_ee_dist"] for e in episodes])),
        "final_belief_err_mean": float(np.mean([e["final_belief_err"] for e in episodes])),
        "steps_to_success_median": _median_or_none([e["steps_to_success"] for e in episodes]),
        "max_dev_after_success_median": _median_or_none([e["max_dev_after_success"] for e in episodes]),
        "steps_outside_after_success_median": _median_or_none([e["steps_outside_after_success"] for e in episodes]),
        "control_hz_mean": float(np.mean([e["control_hz"] for e in episodes])),
        "disturbances": cfg.env.get("disturbances", []),
    }
    (out_dir / f"{tag}_{task}.json").write_text(json.dumps({"summary": summary, "episodes": episodes}, indent=1))
    return summary


def _median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None
