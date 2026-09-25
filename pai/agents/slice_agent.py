"""The vertical slice: one task through every level, on privileged object state.

    L2 goal        on(obj, target) as an ordered list of subgoal preferences
    L1 planning    MPPI in the learned entity world model, scored by expected free energy
    L0 body        the chosen end-effector action, held for `action_repeat` control steps
    surprise       prediction vs outcome, standardised by the model's own uncertainty
    causes         Bayesian comparison of hidden causes (pai.causes.inference) when surprise spikes
    adaptation     a confident explanation changes what the agent does next: it recalibrates its
                   camera, expects the extra weight, or carries a slippery object more gently
    memory         every episode stored with its expectations, surprises, causes and outcome

Everything above L0 sees only entity tokens. `perceive` chooses where they come from: the
simulator's object state seen through the calibrated camera pose (default,
pai.perception.camera_frame) or the camera image through DINOv2 and object slots
(pai.perception.slot_tokens.SlotPerception). Nothing else changes between the two.
"""

from __future__ import annotations

import numpy as np
import torch

from pai.causes.inference import N_CH, Calibration, StepEvidence, SurpriseMonitor, combined_sigma
from pai.envs import build_disturbances
from pai.goals.relations import RelationalGoal
from pai.memory.episodic import EpisodeRecord
from pai.memory.overlay import annotate
from pai.memory.recipes import proposal
from pai.perception.camera_frame import CameraFramePerception
from pai.planning.mppi import MPPIPlanner
from pai.world.entities import FORCE, POS, entity_names
from pai.cognition.causes import select_arm_cause

HOLDING_SUBGOALS = range(3, 6)  # lifted, over_target, lowered: the object should be in the hand
ADAPT_CONFIDENCE = 0.9          # posterior needed before an explanation changes behaviour
GENTLE_CARRY = 0.5              # speed factor while carrying an object believed to be slippery


def run_episode(env, wm, cfg, episode: int, seed: int, disturbances: list[dict] | None = None,
                relation: str = "on", obj: str = "red", target: str = "plate", device="cpu",
                frames: list | None = None, calibration: Calibration | None = None,
                raw: list | None = None, perceive=None, adapt: bool = True,
                evidence: list | None = None, recipe=None, cognitive_memory=None,
                online_thinker=None) -> EpisodeRecord:
    """raw, if given, collects (residual, model_sigma, step name) per step, used to calibrate
    surprise. evidence, if given, collects (z, holding, subgoal index) per step for the thinker.
    adapt: act on confident explanations (off for the ablation).
    recipe: a recalled pai.memory.recipes.Recipe; its waypoints become an extra planner candidate."""
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
    sense = perceive if perceive is not None else CameraFramePerception(env)
    if hasattr(sense, "reset"):
        sense.reset()
    others = [i for i in range(1, len(names)) if i != goal.i_o]  # "the scene": objects not being handled
    hold_bias = np.zeros(N_CH)   # expected extra weight and sag, learned from an explanation
    carry_speed, adaptations, camera_fixed = 1.0, [], set()
    speed_limit = planner.high[:3].clone()
    prev_ee = obs["ee_pos"].copy()
    tokens = sense(obs, prev_ee)
    start_obj = tokens[goal.i_o, POS].copy()  # recipes refer to where the object started
    plan_dt = env.control_dt * sc.action_repeat
    subgoal_times, surprise = {}, []
    traj_tokens, traj_actions, phases = [tokens.copy()], [], {}
    thinker_events = []
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
        holding = goal.index in HOLDING_SUBGOALS
        factor = carry_speed if holding else 1.0
        planner.high[:3], planner.low[:3] = speed_limit * factor, -speed_limit * factor
        prop = None
        if recipe is not None:
            p_np = proposal(recipe, goal.current.name, tokens, start_obj, goal.i_t, sc.horizon, plan_dt,
                            max_speed=float(speed_limit[0]) * factor, grip=goal.current.grip, i_o=goal.i_o)
            prop = None if p_np is None else torch.as_tensor(p_np, device=device)
        a, _ = planner.act(x, goal.current.cost, grip=goal.current.grip, proposal=prop)
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
        err = new_tokens[:, POS] - (tokens[:, POS] + pred.delta[0, :, :3].cpu().numpy())   # (N, 3)
        var = (pred.aleatoric + pred.epistemic)[0, :, :3].cpu().numpy()
        residual = np.r_[err[0], new_tokens[0, FORCE] - pred.force[0].cpu().numpy(),
                         err[goal.i_o], err[others].mean(0)]
        model_sigma = np.sqrt(np.r_[var[0], pred.force_var[0].cpu().numpy(),
                                    var[goal.i_o], var[others].sum(0) / len(others) ** 2])
        step_name = goal.current.name if goal.current else "done"
        if raw is not None:
            raw.append((residual, model_sigma, step_name))
        sigma = combined_sigma(model_sigma, calibration, step_name)
        raw_ev = StepEvidence(k, residual, sigma, a_np, holding=holding, step=step_name)
        ev = StepEvidence(k, residual - hold_bias, sigma, a_np, holding=holding, step=step_name) if holding else raw_ev
        report = monitor.add(ev, raw_ev)
        surprise.append(round(ev.surprise, 3))
        if evidence is not None:
            evidence.append((residual / ev.sigma, holding, goal.index))
        thinker_report = (online_thinker.observe(k, residual / ev.sigma, holding,
                                                trigger=report is not None)
                          if online_thinker is not None else None)
        if thinker_report is not None:
            thinker_events.append({"type": "thinker_cause", **thinker_report})
        cause, confident = select_arm_cause(report, thinker_report, ADAPT_CONFIDENCE)
        if adapt and confident:
            prm = report.params[cause]
            if (cause == "camera_shift" and hasattr(sense, "correction") and getattr(sense, "allow_recalibration", True)
                    and prm["onset"] not in camera_fixed):
                camera_fixed.add(prm["onset"])  # one jump, one correction, however often it is re-explained
                jump = np.asarray(prm["seen_jump"], np.float32)
                sense.correction -= jump             # recalibrate: undo the apparent jump of the world
                new_tokens[1:, POS] -= jump
                adaptations.append({"t": k, "cause": cause, "action": "recalibrated the camera",
                                    "correction_mm": (-1000 * jump).round(1).tolist()})
            elif cause == "heavier_object":
                # fitted to raw evidence, so the estimate is the whole extra weight, not an increment
                hold_bias[5] = prm["extra_weight_N"]
                hold_bias[[2, 8]] = -prm["sag_mm"] / 1000
                adaptations.append({"t": k, "cause": cause, "action": "expect the extra weight",
                                    "extra_mass_kg": prm["extra_mass_kg"]})
            elif cause == "slippery_object" and carry_speed == 1.0:
                carry_speed = GENTLE_CARRY
                adaptations.append({"t": k, "cause": cause, "action": f"carry at {GENTLE_CARRY:.0%} speed"})
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
    record = EpisodeRecord(
        episode=episode, goal=f"{relation}({obj}, {target})", success=bool(success), steps=k,
        subgoal_times=subgoal_times, surprise=surprise,
        inferred_cause=verdict.best if verdict else "none",
        cause_posterior=verdict.posterior if verdict else None,
        cause_params=verdict.params if verdict else None,
        true_disturbances=truth,
        fallbacks=fallbacks,
        adaptations=adaptations,
        surprise_threshold=calibration.threshold,
        tokens=np.stack(traj_tokens),
        actions=np.stack(traj_actions) if traj_actions else np.zeros((0, 4), np.float32),
        phases=phases,
        events=[e for e in env.events.events if e["type"] in ("grasp", "release", "slip", "relation_true",
                                                                 "relation_false", "disturbance_start",
                                                                 "disturbance_end")] + thinker_events,
    )
    if cognitive_memory is not None:
        from pai.cognition.adapters import from_arm_record
        cognitive_memory.add(from_arm_record(record, names))
    return record
