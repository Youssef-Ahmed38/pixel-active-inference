"""Builds the MuJoCo scene around the MuJoCo Menagerie Franka Panda using MjSpec."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from pai.config import REPO_ROOT

MENAGERIE_DIR = Path(os.environ.get("PAI_MENAGERIE", REPO_ROOT / "third_party" / "mujoco_menagerie"))
PANDA_XML = MENAGERIE_DIR / "franka_emika_panda" / "panda.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
LIGHT_NAMES = ("key", "fill")


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """MuJoCo camera `xyaxes` for a camera at `pos` looking at `target` (cameras look along -z)."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x, *y]


def build_model(cfg) -> mujoco.MjModel:
    """cfg: the `env` section of the config (camera, lights, render quality)."""
    if not PANDA_XML.exists():
        raise FileNotFoundError(
            f"{PANDA_XML} not found. Run `python scripts/fetch_assets.py` or set PAI_MENAGERIE."
        )
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    wb = spec.worldbody

    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], rgba=[0.12, 0.14, 0.18, 1])
    wb.add_light(name="key", pos=[1.0, 0.8, 2.2], dir=[-0.4, -0.3, -1.0], diffuse=[0.75, 0.75, 0.75])
    wb.add_light(name="fill", pos=[-0.8, -1.0, 1.8], dir=[0.4, 0.5, -1.0], diffuse=[0.35, 0.35, 0.4])

    cam = cfg.camera
    wb.add_camera(name=cam.name, pos=list(cam.pos), xyaxes=lookat_xyaxes(cam.pos, cam.target), fovy=cam.fovy)

    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    model = spec.compile()
    model.vis.quality.offsamples = int(cfg.render.msaa)
    model.vis.quality.shadowsize = int(cfg.render.shadow_size)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.image_size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.image_size)
    return model
