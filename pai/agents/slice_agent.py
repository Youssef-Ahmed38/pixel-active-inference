"""The vertical slice: one task through every level, on privileged object state.

    L2 goal        on(obj, target) as an ordered list of subgoal preferences
    L1 planning    MPPI in the learned entity world model, scored by expected free energy
    L0 body        the chosen end-effector action, held for `action_repeat` control steps
    surprise       prediction vs outcome, standardised by the model's own uncertainty
    causes         Bayesian comparison of none / push / heavier_object when surprise spikes
    memory         every episode stored with its expectations, surprises, causes and outcome

Everything above L0 sees only entity tokens. `perceive` chooses where they come from: the
simulator's exact object state (default) or the camera through DINOv2 and object slots
(pai.perception.slot_tokens.SlotPerception). Nothing else changes between the two.
"""

from __future__ import annotations

import numpy as np
import torch

from pai.causes.inference import Calibration, StepEvidence, SurpriseMonitor, combined_sigma
from pai.envs import build_disturbances
from pai.goals.relations import RelationalGoal
from pai.memory.episodic import EpisodeRecord
from pai.memory.overlay import annotate
from pai.planning.mppi import MPPIPlanner
from pai.world.entities import FORCE, POS, encode, entity_names

HOLDING_SUBGOALS = range(3, 6)  # lifted, over_target, lowered: the object should be in the hand


def run_episode(env, wm, cfg, episode: int, seed: int, disturbances: list[dict] | None = None,
                relation: str = "on", obj: str = "red", target: str = "plate", device="cpu",
                frames: list | None = None, calibration: Calibration | None = None,
                raw: list | None = None, perceive=None) -> EpisodeRecord:
    """raw, if given, collects (residual, model_sigma) per step, used to calibrate surprise."""
    sc = cfg.slice
    env.disturbances = build_disturbances(disturbances or [])
    obs = env.reset(seed=seed)
    names = entity_names(env)
    goal = RelationalGoal(relation, obj, target, names, env.objects)
    goal_text = f"{relation}({obj}, {target})"
    planner = MPPIPlanner(wm, horizon=sc.horizon, samples=sc.samples, iterations=sc.iterations,
                          temperature=sc.temperature, epistemic_weight=sc.epistemic_weight, device=device)
    calibration = calibration or Calibration()
    monitor = SurpriseMonitor(env.control_dt * sc.action_repeat, calibration=calibration)
    fallbacks = 0
    sense = perceive if perceive is not None else (lambda obs, prev: encode(env, obs, prev))
    if perceive is not None and hasattr(perceive, "reset"):
        perceive.reset()
    prev_ee = obs["ee_pos"].copy()
    tokens = sense(obs, prev_ee)
    subgoal_times, surprise = {}, []
    traj_tokens, traj_actions, phases = [tokens.copy()], [], {}
    k = 0
    for k in range(sc.max_plan_steps):
        x = torch.as_tensor(tokens, device=device)
        idx_before = goal.index
        if goal.update(x):
            break
        if goal.fell_back:
            fallbacks += 1
            idx_before = 0
        for i in range(idx_before, goal.index):
            subgoal_times[goal.subgoals[i].name] = k
        a, _ = planner.act(x, goal.current.cost, grip=goal.current.grip)
        first, _ = phases.get(goal.current.name, (k, k))
        phases[goal.current.name] = (first, k + 1)
        with torch.no_grad():
            pred = wm.predict(x[None], a[None])
        a_np = a.cpu().numpy()
        belief = monitor.reports[-1].posterior if monitor.reports else None
        level = monitor.history[-1].surprise if monitor.history else 0.0
        for _ in range(sc.action_repeat):
            prev_ee = obs["ee_pos"].copy()
            obs = env.step(np.r_[a_np[:3], 0.0, a_np[3]])
            if frames is not None and env.t % 4 == 0:
                frames.append(annotate(obs["image"], goal_text, goal.current.name, level, belief, env.t * env.control_dt))
        new_tokens = sense(obs, prev_ee)
        residual = np.r_[new_tokens[0, POS] - (tokens[0, POS] + pred.delta[0, 0, :3].cpu().numpy()),
                         new_tokens[0, FORCE] - pred.force[0].cpu().numpy()]
        model_sigma = np.sqrt(np.r_[(pred.aleatoric + pred.epistemic)[0, 0, :3].cpu().numpy(),
                                    pred.force_var[0].cpu().numpy()])
        step_name = goal.current.name if goal.current else "done"
        if raw is not None:
            raw.append((residual, model_sigma, step_name))
        ev = StepEvidence(k, residual, combined_sigma(model_sigma, calibration, step_name), a_np,
                          holding=goal.index in HOLDING_SUBGOALS)
        monitor.add(ev)
        surprise.append(round(ev.surprise, 3))
        tokens = new_tokens
        traj_tokens.append(tokens.copy())
        traj_actions.append(a_np.copy())
    for i in range(goal.index):
        subgoal_times.setdefault(goal.subgoals[i].name, k)
    for _ in range(sc.settle_steps):  # let the object come to rest before judging success
        obs = env.step(np.array([0, 0, 0, 0, 1.0]))
    success = (relation, obj, target) in obs["predicates"]

    verdict = monitor.episode_verdict()
    truth = [e for e in env.events.events if e["type"] == "disturbance_start"]
    return EpisodeRecord(
        episode=episode, goal=f"{relation}({obj}, {target})", success=bool(success), steps=k,
        subgoal_times=subgoal_times, surprise=surprise,
        inferred_cause=verdict.best if verdict else "none",
        cause_posterior=verdict.posterior if verdict else None,
        cause_params=verdict.params if verdict else None,
        true_disturbances=truth,
        fallbacks=fallbacks,
        surprise_threshold=calibration.threshold,
        tokens=np.stack(traj_tokens),
        actions=np.stack(traj_actions) if traj_actions else np.zeros((0, 4), np.float32),
        phases=phases,
        events=[e for e in env.events.events if e["type"] in ("grasp", "release", "slip", "relation_true",
                                                                 "relation_false", "disturbance_start",
                                                                 "disturbance_end")],
    )
