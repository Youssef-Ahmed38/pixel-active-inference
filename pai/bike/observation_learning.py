"""State-observation precursor to watching video: infer controls, then imitate.

The student sees sensor features and controls recovered by inverse dynamics.
Expert controls are recorded solely to audit that inference. This does not yet
extract poses from video, learn the dynamics model, or establish Stage 2 transfer.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from pai.bike.env import BicycleEnv, BalanceController
from pai.cognition.contracts import SkillCommand


def infer_action(env, qpos, qvel, acceleration):
    data = mujoco.MjData(env.model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.qacc[:] = acceleration
    mujoco.mj_inverse(env.model, data)
    rear = env.joints["rear_spin"].dofadr[0]
    steer = env.joints["steering"]
    lean = env.joints["torso_lean"]
    # Invert the known drive torque and steering/lean servo transmissions.
    action = np.array([data.qfrc_inverse[rear],
        qpos[steer.qposadr[0]] + (data.qfrc_inverse[steer.dofadr[0]] + 12 * qvel[steer.dofadr[0]]) / 100,
        qpos[lean.qposadr[0]] + (data.qfrc_inverse[lean.dofadr[0]] + 15 * qvel[lean.dofadr[0]]) / 100])
    return np.clip(action, env.model.actuator_ctrlrange[:, 0], env.model.actuator_ctrlrange[:, 1])


def collect_demonstrations(seeds, output, video=None, diverse_starts=False):
    sensations, inferred, audit_controls, qpositions, qvelocities, episode_ids = [], [], [], [], [], []
    summaries = []
    boundaries = []
    renderer = writer = None
    try:
        for seed in seeds:
            env = BicycleEnv()
            if diverse_starts:
                rng = np.random.default_rng(seed)
                speed, roll = float(rng.uniform(2.5, 4.5)), float(rng.uniform(-.1, .1))
                env.reset(seed, speed, roll)
            else:
                speed, roll = 4.0, None
                env.reset(seed)
            expert = BalanceController()
            if video is not None and writer is None:
                import imageio.v2 as imageio
                renderer = mujoco.Renderer(env.model, height=360, width=640)
                video = Path(video)
                video.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(video), format="FFMPEG", fps=25, macro_block_size=1)
            frame = 0
            while not env.done:
                action = expert.action(env.sensation())
                env.data.ctrl[:] = action
                for substep in range(env.substeps):
                    sensation = env.sensation().copy()
                    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
                    mujoco.mj_step(env.model, env.data)
                    acceleration = (env.data.qvel - qvel) / env.model.opt.timestep
                    recovered = infer_action(env, qpos, qvel, acceleration)
                    sensations.append(sensation)
                    inferred.append(recovered)
                    audit_controls.append(action.copy())
                    qpositions.append(qpos)
                    qvelocities.append(qvel)
                    episode_ids.append(seed)
                    boundaries.append(substep == 0)
                env.steps += 1
                if writer is not None and frame % 2 == 0:
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = env.data.qpos[:3]
                    camera.distance = 3
                    camera.azimuth = 120
                    camera.elevation = -15
                    renderer.update_scene(env.data, camera=camera)
                    writer.append_data(renderer.render())
                frame += 1
            summaries.append({"seed": seed, "success": not env.fallen and env.travelled >= env.distance,
                              "distance": env.travelled, "seconds": env.data.time,
                              "start_speed": speed, "start_roll": roll})
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sensations=sensations, inferred_actions=inferred,
                        audit_expert_actions=audit_controls, qpos=qpositions, qvel=qvelocities,
                        episode_ids=episode_ids, control_boundaries=boundaries)
    errors = np.abs(np.asarray(inferred) - np.asarray(audit_controls))
    return {"episodes": summaries, "samples": len(sensations),
            "inverse_action_mae": errors.mean(axis=0).tolist(),
            "channels": ["rear_torque_Nm", "steering_target_rad", "lean_target_rad"]}


@dataclass
class ObservedRidingPolicy:
    weights: np.ndarray
    training: str = "unspecified initialization"

    @classmethod
    def fit(cls, path, ridge=.01):
        if ridge <= 0:
            raise ValueError("ridge regularization must be positive")
        with np.load(path) as data:
            x = np.asarray(data["sensations"], dtype=float)
            y = np.asarray(data["inferred_actions"], dtype=float)
            # The policy chooses once per control interval. Intermediate states
            # occur after its action and are not additional decision examples.
            if "control_boundaries" in data:
                boundaries = np.asarray(data["control_boundaries"], dtype=bool)
                x, y = x[boundaries], y[boundaries]
        if x.ndim != 2 or x.shape[1] != 6 or y.shape != (len(x), 3) or len(x) == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("finite nonempty demonstrations required")
        x = np.column_stack([x, np.ones(len(x))])
        weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
        return cls(weights, "inferred inverse-dynamics actions; expert control labels unused")

    def action(self, sensation):
        return np.append(sensation, 1.0) @ self.weights

    def __call__(self, state, adapter):
        return None if adapter.done else SkillCommand("ride", parameters=tuple(map(float, self.action(adapter.sensation()))))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "weights": self.weights.tolist(),
            "training": self.training}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported riding checkpoint")
        return cls(np.asarray(data["weights"], dtype=float), data.get("training", "unspecified initialization"))
