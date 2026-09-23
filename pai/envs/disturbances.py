"""Disturbances used by the robustness suite.

Each disturbance is active on control steps [start, start + duration) (duration < 0 means
"until the end of the episode"). Physical disturbances act through the model/data in
before_step; sensory ones rewrite the observation in on_observation. The env restores all
model parameters on reset, so disturbances only need to (re)apply themselves.

Config entries look like:
    - {type: push, start: 40, duration: 10, body: link5, force: [0, 30, 0]}
    - {type: occlusion, start: 0, duration: -1, box: [0.3, 0.3, 0.7, 0.6]}
"""

from __future__ import annotations

import numpy as np


class Disturbance:
    def __init__(self, start: int = 0, duration: int = -1, **kwargs):
        self.start = int(start)
        self.duration = int(duration)
        self.params = kwargs

    def active(self, t: int) -> bool:
        return t >= self.start and (self.duration < 0 or t < self.start + self.duration)

    def reset(self, env) -> None:
        pass

    def before_step(self, env) -> None:
        pass

    def after_step(self, env) -> None:
        pass

    def on_observation(self, env, obs: dict) -> dict:
        return obs


class Push(Disturbance):
    """External Cartesian force (N) on a body, e.g. someone shoving the elbow."""

    def before_step(self, env):
        body = env.model.body(self.params.get("body", "link5")).id
        env.data.xfrc_applied[body, :3] = self.params["force"] if self.active(env.t) else 0.0


class Occlusion(Disturbance):
    """Paints a rectangle over the image. box = [x0, y0, x1, y1] in fractions of the image."""

    def on_observation(self, env, obs):
        if self.active(env.t) and obs.get("image") is not None:
            h, w = obs["image"].shape[:2]
            x0, y0, x1, y1 = self.params.get("box", [0.35, 0.3, 0.7, 0.65])
            img = obs["image"].copy()
            img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = self.params.get("color", [20, 20, 20])
            obs["image"] = img
        return obs


class Lighting(Disturbance):
    """Scales all light intensities (scale < 1 darkens) and optionally tints them."""

    def before_step(self, env):
        if self.active(env.t):
            tint = np.asarray(self.params.get("tint", [1.0, 1.0, 1.0]))
            base = env._nominal["light_diffuse"]
            env.model.light_diffuse[:] = np.clip(base * self.params.get("scale", 0.4) * tint, 0, 1)


class CameraShift(Disturbance):
    """Moves the camera by `offset` metres (a miscalibrated / bumped camera)."""

    def before_step(self, env):
        if self.active(env.t):
            env.model.cam_pos[env.cam_id] = env._nominal["cam_pos"][env.cam_id] + np.asarray(self.params["offset"])


class Payload(Disturbance):  # applied once when it becomes active; env.reset() restores the model
    """Adds mass (kg) to a body: an unknown load in the hand, or an object heavier than it looks.

    The rotational inertia is scaled by the same factor (same shape, denser material). Changing the
    mass alone gives a heavy body with a light body's inertia, which makes the simulation unstable."""

    def reset(self, env):
        self._applied = False

    def before_step(self, env):
        if self.active(env.t) and not getattr(self, "_applied", False):
            import mujoco

            body = env.model.body(self.params.get("body", "hand")).id
            m0 = env._nominal["body_mass"][body]
            m1 = m0 + self.params.get("mass", 2.0)
            env.model.body_mass[body] = m1
            env.model.body_inertia[body] = env._nominal["body_inertia"][body] * (m1 / m0)
            # Derived constants (subtree masses, joint inverse weights) must follow the new mass,
            # otherwise the solver works with stale values and the simulation becomes unstable.
            # mj_setConst overwrites the MjData it is given, so it gets a scratch copy.
            mujoco.mj_setConst(env.model, mujoco.MjData(env.model))
            self._applied = True


class Friction(Disturbance):
    """Scales the sliding friction of every geom of a body (e.g. a slippery or sticky object)."""

    def before_step(self, env):
        if self.active(env.t):
            body = env.model.body(self.params["body"]).id
            geoms = np.flatnonzero(env.model.geom_bodyid == body)
            env.model.geom_friction[geoms, 0] = env._nominal["geom_friction"][geoms, 0] * self.params.get("scale", 0.2)


class SensorNoise(Disturbance):
    """Gaussian noise on proprioception (rad) and pixels (in 0..255 units)."""

    def on_observation(self, env, obs):
        if self.active(env.t):
            obs["q"] = obs["q"] + env.rng.normal(0, self.params.get("q_std", 0.01), obs["q"].shape)
            img_std = self.params.get("image_std", 0.0)
            if img_std > 0 and obs.get("image") is not None:
                noisy = obs["image"].astype(np.float32) + env.rng.normal(0, img_std, obs["image"].shape)
                obs["image"] = np.clip(noisy, 0, 255).astype(np.uint8)
        return obs


REGISTRY = {
    "push": Push,
    "occlusion": Occlusion,
    "lighting": Lighting,
    "camera_shift": CameraShift,
    "payload": Payload,
    "friction": Friction,
    "sensor_noise": SensorNoise,
}


def build_disturbances(specs) -> list[Disturbance]:
    out = []
    for spec in specs or []:
        spec = dict(spec)
        kind = spec.pop("type")
        d = REGISTRY[kind](**spec)
        d.kind = kind  # recorded by the event log
        out.append(d)
    return out
