"""Simulator-only RGB teaching records; not imported by image inference."""
import math
import mujoco
import numpy as np
from pai.bike.env import BicycleEnv, BalanceController
from pai.cognition.contracts import SkillCommand


def collect_motion(seed, controller=None):
    rng = np.random.default_rng(seed)
    env = BicycleEnv()
    env.reset(seed, speed=rng.uniform(2.5, 4.5), roll=rng.uniform(-.1, .1))
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3, 120, -15
    expert = controller or BalanceController()
    images, labels = [], []
    with mujoco.Renderer(env.model, height=96, width=96) as renderer:
        while not env.done:
            if env.steps % 2 == 0:
                camera.lookat[:] = env.data.qpos[:3]
                renderer.update_scene(env.data, camera=camera)
                images.append(renderer.render().copy())
                w, x, y, z = env.data.sensor("orientation").data
                labels.append([math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                               math.asin(np.clip(2 * (w * y - z * x), -1, 1)),
                               math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])
            env.execute(SkillCommand("ride", parameters=tuple(map(float, expert.action(env.sensation())))))
    return np.asarray(images), np.asarray(labels), {
        "seed": seed, "frames": len(images), "success": not env.fallen and env.travelled >= env.distance,
        "distance": env.travelled, "fallen": env.fallen}
