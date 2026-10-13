"""Visual/mobile maze episode, disturbances, and explanations scored against event truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pai.maze.agent import Explorer
from pai.maze.env import Cell, MazeWorld, Observation, VisualObservation
from pai.maze.mobility import ContinuousMobile
from pai.maze.visual import VisualExplorer
from pai.cognition.causes import CausePosterior
from pai.cognition.contracts import CognitiveEpisode, GoalRelation, SkillCommand, SkillOutcome, SurpriseEvidence
from pai.cognition.adapters import maze_state


@dataclass
class MazeCauseMonitor:
    reports: list[dict] = field(default_factory=list)

    def observe_collision(self, step: int, target: Cell, previously_free: bool, collided: bool) -> None:
        if collided and previously_free:
            posterior = CausePosterior.from_log_evidence(
                {"none": -8.0, "passage_blocked": 0.0, "unknown": -4.0})
            if not posterior.should_adapt():
                return
            self.reports.append({"step": step, "cause": "passage_blocked", "target": target,
                                 "posterior": posterior.probabilities,
                                 "action": "mark the passage closed and replan"})

    def observe_camera(self, step: int, before: Cell, after: Cell) -> None:
        if before != after:
            posterior = CausePosterior.from_log_evidence(
                {"none": -8.0, "camera_shift": 0.0, "unknown": -4.0})
            if not posterior.should_adapt():
                return
            self.reports.append({"step": step, "cause": "camera_shift", "offset": after,
                                 "posterior": posterior.probabilities,
                                 "action": "recalibrate the image-to-map offset"})


def run_advanced_episode(world: MazeWorld, *, visual: bool = True, mobile: bool = True,
                         budget: int = 600, block_at: int | None = None,
                         shift_at: int | None = None, odom_noise: float = 0.18,
                         seed: int = 0, frame_hook: Callable | None = None,
                         cognitive_memory=None, online_thinker=None,
                         visual_classifier=None, walk: bool = False,
                         fall_at: int | None = None,
                         causal_probes: bool = True) -> dict:
    """Run one episode; agents see only Observation or VisualObservation.

    The runner may use truth to select a connected passage for injection and to
    score results, but does not pass truth to the policy or cause monitor.
    """
    agent = (VisualExplorer(world.size, world.sensor_range, seed,
                            classifier=visual_classifier,
                            enable_body_recovery=causal_probes) if visual
             else Explorer(world.sensor_range, world.size, seed))
    if walk:
        from pai.maze.walking import WalkingMobile
        body = WalkingMobile(world, fall_at=fall_at)
    else:
        body = ContinuousMobile(world) if mobile else None
    causes = MazeCauseMonitor()
    cognitive = CognitiveEpisode("maze", seed, GoalRelation("cover", "maze"), False, 0)
    obs: Observation | VisualObservation = (world.visual_observe(odom_noise=odom_noise)
                                            if visual else world.observe())
    first_goal_step = 0 if obs.reached_goal else None
    errors, raw_errors = [], []
    raw_odometry = np.asarray(world.start, dtype=float)
    did_block = did_shift = False
    thinker_actions = []
    for _ in range(budget):
        previous_offset = agent.localizer.camera_offset if visual else (0, 0)
        move = agent.act(obs)
        if visual:
            causes.observe_camera(world.steps, previous_offset, agent.localizer.camera_offset)
            if cognitive_memory is not None:
                offset_delta = np.asarray(agent.localizer.camera_offset) - np.asarray(previous_offset)
                cognitive.evidence.append(SurpriseEvidence(world.steps, ("visual_frame_shift",),
                                                            (float(np.linalg.norm(offset_delta)),), (0.1,)))
        if visual:
            raw_odometry += np.asarray(obs.odom_delta)
            errors.append(abs(agent.localizer.position[0] - world.position[0])
                          + abs(agent.localizer.position[1] - world.position[1]))
            raw_errors.append(float(np.linalg.norm(raw_odometry - world.position, ord=1)))
        if cognitive_memory is not None and world.steps % 10 == 0:
            if visual:
                cognitive.beliefs.append(maze_state(agent.localizer.position, agent.localizer.map_labels,
                                                     world.steps, len(agent.localizer.map_labels)))
            else:
                cognitive.beliefs.append(maze_state(obs.position, obs.visible, world.steps,
                                                     len(agent.memory.cells)))
        if frame_hook is not None:
            frame_hook(world, agent, obs, causes)
        if move is None:
            break
        if cognitive_memory is not None:
            cognitive.commands.append(SkillCommand("move", parameters=tuple(map(float, move))))
        target = world.position[0] + move[0], world.position[1] + move[1]
        previously_free = agent.memory.cells.get((agent.localizer.position[0] + move[0],
                                                   agent.localizer.position[1] + move[1])
                                                  if visual else target) in ("floor", "goal")
        if block_at is not None and not did_block and world.steps >= block_at:
            did_block = world.block(target)
        body_result = body.move(move) if body else None
        outcome = body_result.observation if body_result else world.step(move)
        fell = bool(body_result is not None and getattr(body_result, "fell", False))
        if cognitive_memory is not None:
            cognitive.outcomes.append(SkillOutcome(cognitive.commands[-1], not outcome.collided,
                                                   outcome.collided, 1.0))
            cognitive.evidence.append(SurpriseEvidence(world.steps, ("motion_stall",),
                                                        (float(outcome.collided),), (0.1,)))
            cognitive.evidence.append(SurpriseEvidence(world.steps, ("unexpected_collision",),
                                                        (float(outcome.collided and previously_free),), (0.1,)))
        if shift_at is not None and not did_shift and world.steps >= shift_at:
            world.shift_camera((0, 1))
            did_shift = True
        obs = world.visual_observe(outcome.collided, odom_noise, body_fell=fell) if visual else outcome
        causes.observe_collision(world.steps, target, previously_free, outcome.collided and not fell)
        if online_thinker is not None:
            report = online_thinker.observe(
                world.steps, unexpected_collision=bool(outcome.collided and previously_free and not fell),
                frame_shift=bool(visual and previous_offset != agent.localizer.camera_offset))
            if report is not None and report["actionable"]:
                if report["cause"] == "passage_blocked" and outcome.collided and previously_free:
                    believed_target = (agent.localizer.position[0] + move[0],
                                       agent.localizer.position[1] + move[1]) if visual else target
                    agent.memory.cells[believed_target] = "wall"
                    for adjacent in agent.memory.edges.pop(believed_target, set()):
                        agent.memory.edges[adjacent].discard(believed_target)
                    agent.planner.target = None
                    thinker_actions.append({"step": world.steps, "cause": report["cause"],
                                            "action": "invalidate blocked route"})
                elif report["cause"] == "camera_shift" and visual:
                    thinker_actions.append({"step": world.steps, "cause": report["cause"],
                                            "action": "accept visual frame recalibration"})
        if outcome.reached_goal and first_goal_step is None:
            first_goal_step = world.steps
    truth = world.truth()
    truth_causes = [e["type"] for e in world.events if e["type"] in ("passage_blocked", "camera_shift")]
    reported = [e["cause"] for e in causes.reports]
    if cognitive_memory is not None:
        cognitive.success = first_goal_step is not None and truth["visited"] == truth["reachable"]
        cognitive.steps = world.steps
        cognitive.events = list(world.events)
        cognitive.adaptations = list(causes.reports)
        cognitive.adaptations.extend(thinker_actions)
        cognitive.cause_posterior = (causes.reports[-1]["posterior"] if causes.reports else {"none": 1.0})
        cognitive_memory.add(cognitive)
    return {"steps": world.steps, "visited": truth["visited"], "reachable": truth["reachable"],
            "coverage": truth["coverage"], "goal_found": first_goal_step is not None,
            "goal_step": first_goal_step, "collisions": sum(e["type"] == "collision" for e in world.events),
            "mean_pose_error": float(np.mean(errors)) if errors else 0.0,
            "mean_raw_odom_error": float(np.mean(raw_errors)) if raw_errors else 0.0,
            "loop_closures": agent.localizer.loop_closures if visual else 0,
            "camera_corrections": agent.localizer.camera_corrections if visual else 0,
            "body_recovery": agent.body_recovery if visual else False,
            "body_visited": len(agent.body_visited) if visual else 0,
            "body_attempted": len(agent.body_attempted) if visual else 0,
            "causal_probes": len(agent.probe_reports) if visual else 0,
            "truth_causes": truth_causes, "cause_reports": causes.reports,
            "cause_recall": {cause: cause in reported for cause in set(truth_causes)},
            "body_time": body.time if body else float(world.steps),
            "falls": body.falls if walk else 0,
            "recoveries": body.recoveries if walk else 0,
            "thinker_reports": online_thinker.reports if online_thinker is not None else [],
            "thinker_actions": thinker_actions}
