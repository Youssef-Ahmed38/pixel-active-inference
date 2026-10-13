"""Kinematic clearance of a body pose: does the hand (and, for an arm, the arm) fit there?

A grasp is not only a palm pose at the part: the rest of the body has to fit around it, at the grasp
and all the way through what the grasp is for (the turn of a knob or a key, the swing of a door). Which
of the grasps a part allows fits depends on the body (a Shadow forearm, a Panda hand 20 cm wide, a
Robotiq gripper) and on what is around the part (the latch jamb, a shelf, the knob under the lock).

`Clearance(env)` answers this on a scratch copy of the simulation state, never touching the live one:

    cost = Clearance(env).cost(palm_pos, palm_rot, closure=0.6, ignore=(knob_body,), joints={door_q: 0.4})

puts the body there kinematically (floating hands: the wrist; arms: damped least-squares IK from the
current posture, with restarts; `reach_error` is what the arm could not reach), sets the finger synergy
to `closure`, optionally sets other joints (a door opened, a knob turned), and returns the summed
penetration of the body's geoms into the world within a `margin` (m), plus the reach error. Bodies in
`ignore` (the part being grasped, a key in the hand) do not count. It is privileged (it reads the model),
like the oracle that uses it to choose among grasps; no learning agent calls it.
"""

from __future__ import annotations

from contextlib import contextmanager

import mujoco
import numpy as np

from pai.envs.embodiments import ArmHand, FloatingHand, quat_from_mat, rot_error


class Clearance:
    def __init__(self, env, margin: float = 0.01, ik_restarts: int = 6, ik_iters: int = 250):
        self.env, self.emb, self.m = env, env.emb, env.model
        self.d = mujoco.MjData(self.m)
        self.margin = margin
        self.ik_restarts, self.ik_iters = ik_restarts, ik_iters
        emb = self.emb
        self.own = np.array([emb.own_body[self.m.geom_bodyid[g]] for g in range(self.m.ngeom)], bool)
        self.own_geoms = np.flatnonzero(self.own)
        self.reach_error = 0.0
        self.rot_error = 0.0
        self.last_q = None

    @contextmanager
    def _scratch(self):
        """The embodiment's methods on the scratch data (they read self.data)."""
        live = self.emb.data
        self.emb.data = self.d
        try:
            yield self.d
        finally:
            self.emb.data = live

    def _sync(self) -> None:
        live = self.env.data
        self.d.qpos[:] = live.qpos
        self.d.qvel[:] = 0
        self.d.mocap_pos[:] = live.mocap_pos
        self.d.mocap_quat[:] = live.mocap_quat
        self.d.eq_active[:] = live.eq_active

    def _place(self, pos, R, closure, q0=None) -> None:
        emb, d = self.emb, self.d
        if isinstance(emb, FloatingHand):
            d.qpos[emb.free_qadr:emb.free_qadr + 3] = pos
            d.qpos[emb.free_qadr + 3:emb.free_qadr + 7] = quat_from_mat(R)
            self.reach_error = self.rot_error = 0.0
        elif isinstance(emb, ArmHand):
            q = self.solve_ik(pos, R, q0)
            d.qpos[emb.qadr] = q
        if closure is not None:
            emb._preset_fingers(float(closure))

    def solve_ik(self, pos, R, q0=None) -> np.ndarray:
        """Arm joints for the palm pose (sets reach_error / rot_error). From q0 alone if given (a sweep
        continues from its previous pose), else the best of the current posture, home and a few restarts."""
        emb, d = self.emb, self.d
        rng = np.random.default_rng(0)
        if q0 is not None:
            starts = [np.asarray(q0, float)]
        else:
            starts = [self.env.data.qpos[emb.qadr].copy(), emb.home_q.copy()]
            starts += [rng.uniform(emb.jrange[:, 0], emb.jrange[:, 1]) for _ in range(self.ik_restarts)]
        best = None
        for q in starts:
            for _ in range(self.ik_iters):
                q = np.clip(q + np.clip(emb._ik_step(q, pos, R, emb.home_q), -0.1, 0.1), emb.jrange[:, 0],
                            emb.jrange[:, 1])
            p, Rq = emb._fk(q)
            e, re = float(np.linalg.norm(p - pos)), float(np.linalg.norm(rot_error(Rq, R)))
            if best is None or e + 0.05 * re < best[1] + 0.05 * best[2]:
                best = (q.copy(), e, re)
            if e < 0.003 and re < 0.03:
                break
        self.last_q, self.reach_error, self.rot_error = best
        return best[0]

    def penetration(self, ignore=()) -> float:
        """Summed penetration (within margin) of the body's geoms into anything not in `ignore` (body ids,
        with their subtrees)."""
        m, d = self.m, self.d
        ign = set()
        for b in ignore:
            for c in range(m.nbody):
                x = c
                while True:
                    if x == b:
                        ign.add(c)
                        break
                    if x == 0:
                        break
                    x = m.body_parentid[x]
        saved = m.geom_margin[self.own_geoms].copy()
        m.geom_margin[self.own_geoms] = self.margin
        try:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
        finally:
            m.geom_margin[self.own_geoms] = saved
        pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            o1, o2 = self.own[c.geom1], self.own[c.geom2]
            if o1 == o2:
                continue
            other = c.geom2 if o1 else c.geom1
            if m.geom_bodyid[other] in ign:
                continue
            pen += max(0.0, self.margin - c.dist)
        return pen

    def cost(self, pos, R, closure: float | None = 0.6, ignore=(), joints: dict | None = None,
             reach_weight: float = 10.0, q0=None) -> float:
        """Penetration + reach_weight * reach error of the body placed at (pos, R). joints: {qpos adr: value};
        q0: arm posture to start the IK from (see solve_ik)."""
        self._sync()
        with self._scratch():
            for adr, v in (joints or {}).items():
                self.d.qpos[adr] = v
            self._place(np.asarray(pos, float), np.asarray(R, float), closure, q0)
            pen = self.penetration(ignore)
        return pen + reach_weight * (self.reach_error + 0.05 * self.rot_error)

    def sweep_cost(self, poses, **kw) -> float:
        """Summed cost over a sequence of (pos, R, joints) poses (an arm's IK continues along the sweep)."""
        total, q0 = 0.0, None
        for pos, R, joints in poses:
            total += self.cost(pos, R, joints=joints, q0=q0, **kw)
            q0 = self.last_q if isinstance(self.emb, ArmHand) else None
        return total
