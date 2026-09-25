"""Tabletop manipulation scene: Panda + table + blocks, a plate and a bowl.

Built on the Menagerie Panda with MjSpec. Adds what Phase 1 needs on top of PandaEnv:
- objects with free joints, randomised placement
- end-effector control (Cartesian deltas + yaw + gripper) through damped least-squares IK,
  alongside joint-velocity control
- a compliant mode: gravity compensation plus low joint stiffness, so pushes actually move the
  arm and recovering is the agent's job rather than the stiff servos'
- privileged state (object poses), relation predicates and a ground-truth event log, which
  are the answer key for scoring the agent's explanations
- segmentation masks per object, used to supervise object slots at first

Observation dict: image, q, dq, gripper (finger opening, m), ee_pos, wrist_force (N, world frame),
plus privileged
`state` (object name -> pose) and `predicates` (set of relation tuples).
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from pai.envs.disturbances import Disturbance, build_disturbances
from pai.envs.events import EventLog
from pai.envs.predicates import ObjectSpec, compute_predicates
from pai.envs.scene import ARM_JOINTS, PANDA_XML, lookat_xyaxes

ARM_BODIES = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]
TABLE_TOP_Z = 0.0
GRIPPER_OPEN_CTRL = 255.0
BOWL_WALL_SEGMENTS = 14


def _add_objects(spec: mujoco.MjSpec, cfg) -> dict[str, ObjectSpec]:
    wb = spec.worldbody
    objects: dict[str, ObjectSpec] = {}
    geom = mujoco.mjtGeom

    for b in cfg.blocks:
        h = float(b.size) / 2
        body = wb.add_body(name=b.name, pos=[0.5, 0.0, h])
        body.add_freejoint()
        body.add_geom(name=f"{b.name}_geom", type=geom.mjGEOM_BOX, size=[h, h, h], rgba=list(b.rgba),
                      mass=float(b.mass), friction=[float(cfg.object_friction), 0.01, 0.001], condim=4)
        objects[b.name] = ObjectSpec(b.name, "block", radius=h, height=2 * h)

    p = cfg.plate
    body = wb.add_body(name="plate", pos=[0.5, 0.2, p.height / 2])
    body.add_freejoint()
    # A 12 mm cylinder is fragile in MuJoCo. A block that lands tilted touches it with a single contact
    # point, pivots through the plate and gets the plate ejected: 8 of 9 placement failures in
    # evaluation v3/v4 (diagnosis: replaying the recorded falls). A 3 mm contact margin detects the
    # contact before touching and stops all replayed falls; the price is that objects rest ~3 mm above
    # the plate. The stiffer solref additionally limits how deep a fast block sinks in.
    body.add_geom(name="plate_geom", type=geom.mjGEOM_CYLINDER, size=[p.radius, p.height / 2, 0], rgba=list(p.rgba),
                  mass=float(p.mass), friction=[float(cfg.object_friction), 0.01, 0.001], condim=4,
                  solref=[0.005, 1.0], margin=0.003)
    objects["plate"] = ObjectSpec("plate", "plate", radius=float(p.radius), height=float(p.height))

    bw = cfg.bowl
    body = wb.add_body(name="bowl", pos=[0.5, -0.2, 0.0])
    body.add_freejoint()
    base_h = 0.005
    body.add_geom(name="bowl_base", type=geom.mjGEOM_CYLINDER, size=[bw.radius, base_h, 0], pos=[0, 0, base_h],
                  rgba=list(bw.rgba), mass=float(bw.mass) * 0.4)
    wall_len = 2 * np.pi * bw.radius / BOWL_WALL_SEGMENTS * 0.6
    for i in range(BOWL_WALL_SEGMENTS):
        ang = 2 * np.pi * i / BOWL_WALL_SEGMENTS
        body.add_geom(name=f"bowl_wall{i}", type=geom.mjGEOM_BOX,
                      size=[0.004, wall_len, bw.height / 2],
                      pos=[bw.radius * np.cos(ang), bw.radius * np.sin(ang), bw.height / 2],
                      quat=[np.cos(ang / 2), 0, 0, np.sin(ang / 2)], rgba=list(bw.rgba),
                      mass=float(bw.mass) * 0.6 / BOWL_WALL_SEGMENTS)
    objects["bowl"] = ObjectSpec("bowl", "bowl", radius=float(bw.radius), height=float(bw.height))
    return objects


def build_tabletop_model(cfg, extra: Callable[[mujoco.MjSpec], None] | None = None,
                         ) -> tuple[mujoco.MjModel, dict[str, ObjectSpec]]:
    """extra, if given, adds more bodies to the spec just before compiling (e.g. articulated fixtures)."""
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    wb = spec.worldbody
    geom = mujoco.mjtGeom

    # Table top is the plane z = 0; the robot base stands on it. A visible slab plus a floor below.
    wb.add_geom(name="table", type=geom.mjGEOM_BOX, size=[0.6, 0.6, 0.02], pos=[0.45, 0.0, -0.02],
                rgba=list(cfg.table_rgba))
    wb.add_geom(name="floor", type=geom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[0, 0, -0.75], rgba=[0.12, 0.14, 0.18, 1])
    wb.add_light(name="key", pos=[1.0, 0.8, 2.2], dir=[-0.4, -0.3, -1.0], diffuse=[0.7, 0.7, 0.7])
    wb.add_light(name="fill", pos=[-0.8, -1.0, 1.8], dir=[0.4, 0.5, -1.0], diffuse=[0.35, 0.35, 0.4])
    for cam in cfg.cameras:
        wb.add_camera(name=cam.name, pos=list(cam.pos), xyaxes=lookat_xyaxes(cam.pos, cam.target), fovy=cam.fovy)

    for name in ARM_BODIES:  # gravity compensation: needed for the compliant mode, harmless for stiff servos
        spec.body(name).gravcomp = 1.0
    spec.body("hand").add_site(name="grip", pos=[0, 0, 0.1034])  # between the fingertips
    # Wrist force/torque sensing (the force part is used): what a held object weighs, and pushes.
    spec.body("hand").add_site(name="wrist", pos=[0, 0, 0])
    spec.add_sensor(name="wrist_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="wrist")
    # The Menagerie gripper servo closes with only ~2 N on a 4 cm block (force = -100 * finger gap),
    # far below a real Panda hand (~70 N), and cannot hold anything heavier than a toy block. Scale
    # its stiffness and damping by cfg.gripper_strength (10 -> ~20 N on a block).
    grip = spec.actuator("actuator8")
    k = float(cfg.get("gripper_strength", 1.0))
    grip.gainprm[0] *= k
    grip.biasprm[1] *= k
    grip.biasprm[2] *= k

    objects = _add_objects(spec, cfg)
    if extra is not None:
        extra(spec)
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    model = spec.compile()
    model.vis.quality.offsamples = int(cfg.render.msaa)
    model.vis.quality.shadowsize = int(cfg.render.shadow_size)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.image_size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.image_size)
    return model, objects


class TabletopEnv:
    def __init__(self, cfg, disturbances: list[Disturbance] | None = None):
        self.cfg = cfg
        self.model, self.objects = self._build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, cfg.image_size, cfg.image_size)
        m = self.model

        self.qpos_idx = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.qvel_idx = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act_idx = np.arange(7)
        self.grip_act = m.actuator("actuator8").id
        self.finger_qpos = np.array([m.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.joint_range = np.array([m.joint(j).range for j in ARM_JOINTS])
        self.home_q = np.array(cfg.home_q, float)
        self.grip_site = m.site("grip").id
        self.wrist_site = m.site("wrist").id
        self.force_adr = int(m.sensor_adr[m.sensor("wrist_force").id])
        self.cam_id = m.camera(cfg.cameras[0].name).id
        self.obj_body = {n: m.body(n).id for n in self.objects}
        self.obj_qpos = {n: m.joint(m.body(n).jntadr[0]).qposadr[0] for n in self.objects}
        self.obj_qvel = {n: m.joint(m.body(n).jntadr[0]).dofadr[0] for n in self.objects}

        self.n_substeps = max(1, round(cfg.control_dt / m.opt.timestep))
        self.control_dt = self.n_substeps * m.opt.timestep
        self._nominal = {
            "light_diffuse": m.light_diffuse.copy(),
            "cam_pos": m.cam_pos.copy(),
            "cam_quat": m.cam_quat.copy(),
            "body_mass": m.body_mass.copy(),
            "body_inertia": m.body_inertia.copy(),
            "geom_friction": m.geom_friction.copy(),
            "geom_priority": m.geom_priority.copy(),
            "actuator_gainprm": m.actuator_gainprm.copy(),
            "actuator_biasprm": m.actuator_biasprm.copy(),
        }
        self.disturbances = disturbances if disturbances is not None else build_disturbances(cfg.get("disturbances", []))
        self.rng = np.random.default_rng()
        self.events = EventLog(self)
        self.t = 0
        self.reset()

    def _build_model(self, cfg) -> tuple[mujoco.MjModel, dict[str, ObjectSpec]]:
        """Subclasses override this to add bodies (see build_tabletop_model's `extra`)."""
        return build_tabletop_model(cfg)

    # ------------------------------------------------------------------ compliance
    def set_stiffness(self, scale: float) -> None:
        """Scale the arm's joint-servo stiffness (1 = Menagerie default, ~0.05 = compliant).
        Damping scales with sqrt(scale) so the joints stay roughly equally damped."""
        g, b = self._nominal["actuator_gainprm"], self._nominal["actuator_biasprm"]
        kp, kd = g[:7, 0] * scale, -b[:7, 2] * np.sqrt(scale)
        self.model.actuator_gainprm[:7, 0] = kp
        self.model.actuator_biasprm[:7, 1] = -kp
        self.model.actuator_biasprm[:7, 2] = -kd

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None, layout: dict[str, np.ndarray] | None = None) -> dict:
        """layout optionally fixes object xy positions (name -> [x, y]); others are sampled."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mass_changed = not np.array_equal(self.model.body_mass, self._nominal["body_mass"])
        for key, value in self._nominal.items():
            getattr(self.model, key)[:] = value
        if mass_changed:  # a payload changed masses: recompute the derived constants for the original ones
            mujoco.mj_setConst(self.model, mujoco.MjData(self.model))  # scratch data: it overwrites what it gets
        self.set_stiffness(float(self.cfg.stiffness))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_idx] = self.home_q
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.act_idx] = self.home_q
        self.data.ctrl[self.grip_act] = GRIPPER_OPEN_CTRL
        self._place_objects(layout or {})
        mujoco.mj_forward(self.model, self.data)
        self.ee_target_pos = self.data.site_xpos[self.grip_site].copy()
        self.ee_target_yaw = 0.0
        self._down_rot = self.data.site_xmat[self.grip_site].reshape(3, 3).copy()
        self.t = 0
        for d in self.disturbances:
            d.reset(self)
        self._settle(int(self.cfg.settle_steps))
        self.events.reset()
        return self._observe()

    def _place_objects(self, layout: dict) -> None:
        lo, hi = np.array(self.cfg.spawn_low), np.array(self.cfg.spawn_high)
        placed: list[tuple[np.ndarray, float]] = []
        for name, spec in self.objects.items():
            for _ in range(200):
                xy = np.asarray(layout[name], float) if name in layout else self.rng.uniform(lo, hi)
                if name in layout or all(np.linalg.norm(xy - p) > spec.radius + r + 0.02 for p, r in placed):
                    break
            placed.append((xy, spec.radius))
            adr = self.obj_qpos[name]
            z = spec.height / 2 if spec.kind != "bowl" else 0.0
            yaw = self.rng.uniform(-np.pi, np.pi) if spec.kind == "block" else 0.0
            self.data.qpos[adr : adr + 3] = [xy[0], xy[1], TABLE_TOP_Z + z + 0.002]
            self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def _settle(self, n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray) -> dict:
        """Action layout depends on cfg.control:
        ee:              [dx, dy, dz, dyaw, grip]   (m/s, rad/s, grip in [-1 close, 1 open])
        joint_velocity:  [7 joint velocities, grip]
        """
        action = np.asarray(action, float)
        if self.cfg.control == "ee":
            self._ee_control(action[:4])
        else:
            vel = np.clip(action[:7], -self.cfg.max_joint_vel, self.cfg.max_joint_vel)
            target = self.data.ctrl[self.act_idx] + vel * self.control_dt
            self.data.ctrl[self.act_idx] = np.clip(target, self.joint_range[:, 0], self.joint_range[:, 1])
        self.data.ctrl[self.grip_act] = GRIPPER_OPEN_CTRL * (np.clip(action[-1], -1, 1) + 1) / 2

        for d in self.disturbances:
            d.before_step(self)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        for d in self.disturbances:
            d.after_step(self)
        self.t += 1
        obs = self._observe()
        self.events.update(obs)
        return obs

    def _ee_control(self, a: np.ndarray) -> None:
        """Integrate the Cartesian target and solve damped least-squares IK towards it."""
        c = self.cfg
        dp = np.clip(a[:3], -c.max_ee_vel, c.max_ee_vel) * self.control_dt
        lo, hi = np.array(c.ee_low), np.array(c.ee_high)
        self.ee_target_pos = np.clip(self.ee_target_pos + dp, lo, hi)
        self.ee_target_yaw = float(np.clip(self.ee_target_yaw + np.clip(a[3], -1, 1) * c.max_yaw_vel * self.control_dt,
                                           -np.pi / 2, np.pi / 2))
        cy, sy = np.cos(self.ee_target_yaw), np.sin(self.ee_target_yaw)
        rot_target = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ self._down_rot

        m, d = self.model, self.data
        # Solve IK on the commanded joint targets (not the measured pose) so external pushes do not
        # move the setpoint: the compliant servo then pulls the arm back.
        q_cmd = d.ctrl[self.act_idx].copy()
        saved = d.qpos.copy()
        d.qpos[self.qpos_idx] = q_cmd
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        pos = d.site_xpos[self.grip_site].copy()
        rot = d.site_xmat[self.grip_site].reshape(3, 3)
        e_pos = self.ee_target_pos - pos
        e_rot = 0.5 * sum(np.cross(rot[:, i], rot_target[:, i]) for i in range(3))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.grip_site)
        d.qpos[:] = saved
        mujoco.mj_kinematics(m, d)

        J = np.vstack([jp[:, self.qvel_idx], jr[:, self.qvel_idx]])
        err = np.concatenate([e_pos, e_rot])
        lam = float(c.ik_damping)
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
        # Null-space pull towards the home posture keeps the elbow in a sensible place.
        null = np.eye(7) - np.linalg.pinv(J) @ J
        dq += null @ (float(c.ik_nullspace_gain) * (self.home_q - q_cmd))
        dq = np.clip(dq, -c.max_joint_vel * self.control_dt, c.max_joint_vel * self.control_dt)
        self.data.ctrl[self.act_idx] = np.clip(q_cmd + dq, self.joint_range[:, 0], self.joint_range[:, 1])

    def close(self) -> None:
        self.renderer.close()

    # ------------------------------------------------------------------ sensing
    def render(self, camera: str | int | None = None) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.cam_id if camera is None else camera)
        return self.renderer.render()

    def render_segmentation(self, camera: str | int | None = None) -> np.ndarray:
        """(H, W) int labels: 0 background/table, 1 robot, 2 + i for the i-th object in self.objects."""
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(self.data, camera=self.cam_id if camera is None else camera)
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        geom_ids, obj_types = seg[..., 0], seg[..., 1]
        # one table lookup per pixel (label of each geom id, index 0 = no geom): ~100x faster than
        # per-label masks, which made segmentation the bottleneck of frame collection
        if getattr(self, "_geom_label", None) is None:
            body_label = np.zeros(self.model.nbody, np.int32)
            body_label[[self.model.body(n).id for n in ARM_BODIES + ["link0"]]] = 1
            for i, name in enumerate(self.objects):
                body_label[self.obj_body[name]] = 2 + i
            self._geom_label = np.r_[0, body_label[self.model.geom_bodyid]].astype(np.int32)
        # int(): comparing an array with the enum object itself goes element by element in Python
        idx = np.where((obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_ids >= 0), geom_ids + 1, 0)
        return self._geom_label[idx]

    def wrist_force(self) -> np.ndarray:
        """Force measured between the arm and the hand, in the world frame (N). With nothing held it
        is the hand's own weight (~8 N up); a held object adds its weight; a push adds its force."""
        local = self.data.sensordata[self.force_adr : self.force_adr + 3]
        return self.data.site_xmat[self.wrist_site].reshape(3, 3) @ local

    def object_state(self) -> dict[str, dict]:
        out = {}
        for name in self.objects:
            adr, vadr = self.obj_qpos[name], self.obj_qvel[name]
            out[name] = {
                "pos": self.data.qpos[adr : adr + 3].copy(),
                "quat": self.data.qpos[adr + 3 : adr + 7].copy(),
                "vel": self.data.qvel[vadr : vadr + 3].copy(),
            }
        return out

    def _observe(self) -> dict:
        state = self.object_state()
        obs = {
            "image": self.render() if self.cfg.render_images else None,
            "q": self.data.qpos[self.qpos_idx].copy(),
            "dq": self.data.qvel[self.qvel_idx].copy(),
            "gripper": float(self.data.qpos[self.finger_qpos].sum()),
            "ee_pos": self.data.site_xpos[self.grip_site].copy(),
            "wrist_force": self.wrist_force(),
            "state": state,
            "predicates": compute_predicates(self, state),
        }
        for d in self.disturbances:
            obs = d.on_observation(self, obs)
        return obs
