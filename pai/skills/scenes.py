"""Small test beds for the skills, built on DoorSceneEnv fixture adders (the fixtures themselves are not
changed): a fixture held shut by an equality constraint, and a peg with a slot and a solid block.

- `locked(adder)`: the same fixture with a joint-equality lock on its DoF (as the fixture library's
  "locked" cause does), active from the start. Pulling it must stall.
- `with_peg(adder)`: adds a free peg (the held object for insert; its tip site "peg_tip" points along
  the peg), a plate with a square slot ("slot" site at its top, axis down) and a solid block
  ("block_top"), both beside the fixture within reach of every body. A mocap "holder" welded to the peg
  keeps it in place while a hand closes on it (a key taken off a hook), so the grasp does not depend
  on catching a floating object; `hand_peg(env)` puts it in the open hand and `let_go_holder(env)`
  releases the weld.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from pai.envs.embodiments import _axes, quat_from_mat
from pai.skills.core import Target

_BOX = mujoco.mjtGeom.mjGEOM_BOX
PEG_HALF = 0.12          # peg half length (m): long enough that the tip clears every hand's fingers
PEG_RADIUS = 0.012       # as the test fixtures' bar handles, which every hand's synergy is tuned for
SLOT_HALF = 0.021        # half width of the square slot (9 mm clearance around the peg)
SLOT_POS = (0.25, -0.15, 0.5)   # top-centre of the slot plate (world, the test fixtures stand at x=0.5)
BLOCK_POS = (0.25, 0.15, 0.5)   # top-centre of the solid block

PEG = Target("peg", site="peg_tip", axis_local=(0.0, 0.0, -1.0), kind="object")
SLOT = Target("slot", site="slot", axis_local=(0.0, 0.0, -1.0), approach_local=(0.0, 0.0, -1.0))
BLOCK = Target("block", site="block_top", axis_local=(0.0, 0.0, -1.0), approach_local=(0.0, 0.0, -1.0))


def locked(adder: Callable) -> Callable:
    """Fixture adder -> the same fixture held shut by an (active) joint-equality lock."""

    def add(spec: mujoco.MjSpec, pose=((0.5, 0.0, 0.0), 0.0), rng=None):
        info = adder(spec, pose, rng)
        joint = info.joint if hasattr(info, "joint") else info["joint"]
        spec.add_equality(name=f"{joint}_held", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                          name1=joint, active=True, data=[0.0] * 11, solref=[0.005, 1.0])
        return info

    return add


def with_peg(adder: Callable) -> Callable:
    """Fixture adder -> the same fixture plus a peg, a slot plate and a solid block."""

    def add(spec: mujoco.MjSpec, pose=((0.5, 0.0, 0.0), 0.0), rng=None):
        info = adder(spec, pose, rng)
        wb = spec.worldbody
        park = [0.0, 0.8, 0.05]
        peg = wb.add_body(name="peg", pos=park, gravcomp=1.0)
        peg.add_freejoint(name="peg_free")
        peg.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PEG_RADIUS, PEG_HALF, 0], mass=0.3,
                     rgba=[0.85, 0.7, 0.2, 1], friction=[1.5, 0.02, 0.001], condim=4)
        peg.add_site(name="peg_tip", pos=[0, 0, -PEG_HALF - PEG_RADIUS], size=[0.006] * 3)
        wb.add_body(name="peg_holder", mocap=True, pos=park)
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name="peg_hold",
                          name1="peg", name2="peg_holder", data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        (hx, hy, top), hw, t = SLOT_POS, SLOT_HALF, 0.02
        rim = 0.03
        for k, (dx, dy, sx, sy) in enumerate([(hw + rim, 0, rim, hw + 2 * rim), (-hw - rim, 0, rim, hw + 2 * rim),
                                              (0, hw + rim, hw, rim), (0, -hw - rim, hw, rim)]):
            wb.add_geom(name=f"slot_{k}", type=_BOX, size=[sx, sy, t], pos=[hx + dx, hy + dy, top - t],
                        rgba=[0.4, 0.5, 0.6, 1], solref=[0.004, 1])
        wb.add_site(name="slot", pos=[hx, hy, top], size=[0.006] * 3)
        bx, by, btop = BLOCK_POS
        wb.add_geom(name="block", type=_BOX, size=[0.06, 0.06, t], pos=[bx, by, btop - t], rgba=[0.6, 0.4, 0.4, 1],
                    solref=[0.004, 1])
        wb.add_site(name="block_top", pos=[bx, by, btop], size=[0.006] * 3)
        return info

    return add


def hand_peg(env) -> None:
    """Put the peg at the open hand's grasp centre along palm y (as a bar handle), tip down, and hold it
    there with the holder weld until let_go_holder()."""
    m, d = env.model, env.data
    R = d.site_xmat[env.emb.palm_site].reshape(3, 3)
    up = R[:, 1] * (np.sign(R[2, 1]) or 1.0)       # peg +z up: the tip (-z end) points down
    j = m.joint("peg_free")
    q, v = j.qposadr[0], j.dofadr[0]
    d.qpos[q:q + 3] = d.site_xpos[env.emb.palm_site]
    d.qpos[q + 3:q + 7] = quat_from_mat(_axes(R[:, 0], up))
    d.qvel[v:v + 6] = 0
    mid = m.body_mocapid[m.body("peg_holder").id]
    d.mocap_pos[mid], d.mocap_quat[mid] = d.qpos[q:q + 3], d.qpos[q + 3:q + 7]
    d.eq_active[m.equality("peg_hold").id] = 1
    mujoco.mj_forward(m, d)


def let_go_holder(env) -> None:
    env.data.eq_active[env.model.equality("peg_hold").id] = 0
