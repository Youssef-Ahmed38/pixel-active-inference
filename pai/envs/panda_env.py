"""Franka Panda environment with pixel observations and joint-velocity control.

The Menagerie Panda uses joint position servos, so a velocity action `a` (rad/s) is
integrated into a position target: target <- clip(target + a * control_dt). This is the
same interface PixelAI used on the real Panda (velocity commands).

Observations (dict):
    image  (H, W, 3) uint8  camera image, after image-space disturbances
    q      (n_active,)      measured joint angles of the active joints (after sensor noise)
    dq     (n_active,)      joint velocities
    ee_pos (3,)             end-effector position (privileged, for evaluation only)
"""

from __future__ import annotations

import mujoco
import numpy as np

from pai.envs.disturbances import Disturbance, build_disturbances
from pai.envs.scene import ARM_JOINTS, LIGHT_NAMES, build_model

EE_SITE_BODY = "hand"


class PandaEnv:
    def __init__(self, cfg, disturbances: list[Disturbance] | None = None):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, cfg.image_size, cfg.image_size)

        self.active_joints = [ARM_JOINTS[i] for i in cfg.active_joints]
        self.qpos_idx = np.array([self.model.joint(j).qposadr[0] for j in self.active_joints])
        self.qvel_idx = np.array([self.model.joint(j).dofadr[0] for j in self.active_joints])
        self.act_idx = np.array([self.model.actuator(f"actuator{i + 1}").id for i in cfg.active_joints])
        self.joint_range = np.array([self.model.joint(j).range for j in self.active_joints])
        self.home_qpos = self.model.key("home").qpos.copy()
        self.home_ctrl = self.model.key("home").ctrl.copy()
        self.ee_body = self.model.body(EE_SITE_BODY).id
        self.cam_id = self.model.camera(cfg.camera.name).id
        self.light_ids = [self.model.light(n).id for n in LIGHT_NAMES]

        self.n_substeps = max(1, round(cfg.control_dt / self.model.opt.timestep))
        self.control_dt = self.n_substeps * self.model.opt.timestep
        self.max_vel = float(cfg.max_joint_vel)

        # Pristine copies of everything a disturbance may modify, restored on reset().
        self._nominal = {
            "light_diffuse": self.model.light_diffuse.copy(),
            "cam_pos": self.model.cam_pos.copy(),
            "cam_quat": self.model.cam_quat.copy(),
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "geom_friction": self.model.geom_friction.copy(),
        }
        self.disturbances = disturbances if disturbances is not None else build_disturbances(cfg.get("disturbances", []))
        self.rng = np.random.default_rng()
        self.t = 0
        self.reset()  # start at the home pose so render_at() is valid before the first reset

    @property
    def n_active(self) -> int:
        return len(self.active_joints)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, q0: np.ndarray | None = None, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mass_changed = not np.array_equal(self.model.body_mass, self._nominal["body_mass"])
        for key, value in self._nominal.items():
            getattr(self.model, key)[:] = value
        if mass_changed:  # a payload changed masses: recompute the derived constants for the original ones
            mujoco.mj_setConst(self.model, mujoco.MjData(self.model))  # scratch data: it overwrites what it gets
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.ctrl[:] = self.home_ctrl
        if q0 is not None:
            q0 = np.clip(q0, self.joint_range[:, 0], self.joint_range[:, 1])
            self.data.qpos[self.qpos_idx] = q0
            self.data.ctrl[self.act_idx] = q0
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        for d in self.disturbances:
            d.reset(self)
        return self._observe()

    def step(self, action: np.ndarray) -> dict:
        action = np.clip(np.asarray(action, float), -self.max_vel, self.max_vel)
        target = self.data.ctrl[self.act_idx] + action * self.control_dt
        self.data.ctrl[self.act_idx] = np.clip(target, self.joint_range[:, 0], self.joint_range[:, 1])
        for d in self.disturbances:
            d.before_step(self)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        for d in self.disturbances:
            d.after_step(self)
        self.t += 1
        return self._observe()

    def close(self) -> None:
        self.renderer.close()

    # ------------------------------------------------------------------ sensing
    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render()

    def render_at(self, q: np.ndarray) -> np.ndarray:
        """Render the arm at joint configuration q without changing the simulation state."""
        saved = self.data.qpos.copy()
        self.data.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, self.data)
        img = self.render()
        self.data.qpos[:] = saved
        mujoco.mj_kinematics(self.model, self.data)
        return img

    def ee_pos_at(self, q: np.ndarray) -> np.ndarray:
        saved = self.data.qpos.copy()
        self.data.qpos[self.qpos_idx] = q
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.xpos[self.ee_body].copy()
        self.data.qpos[:] = saved
        mujoco.mj_kinematics(self.model, self.data)
        return pos

    def _observe(self) -> dict:
        obs = {
            "image": self.render(),
            "q": self.data.qpos[self.qpos_idx].copy(),
            "dq": self.data.qvel[self.qvel_idx].copy(),
            "ee_pos": self.data.xpos[self.ee_body].copy(),
        }
        for d in self.disturbances:
            obs = d.on_observation(self, obs)
        return obs

    # ------------------------------------------------------------------ sampling
    def sample_q(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Random configuration in a box around the home pose (keeps the arm in view)."""
        rng = rng or self.rng
        home = self.home_qpos[self.qpos_idx]
        half = np.asarray(self.cfg.sample_half_range, float)[list(self.cfg.active_joints)]
        q = home + rng.uniform(-half, half)
        return np.clip(q, self.joint_range[:, 0] + 1e-3, self.joint_range[:, 1] - 1e-3)
