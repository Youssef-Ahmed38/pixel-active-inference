"""Scripted ORACLE for the home doors: proves each variant is physically solvable. Never for learning.

It reads the privileged ground truth (env.info: lock state, where the matching key is) and runs a fixed
plan through the common embodiment action (palm velocity + angular velocity + grasp):

    [open the drawer] -> grasp the matching key -> lift -> align with the keyhole -> insert -> turn a
    quarter (unlock) -> let go                                        (if the key deadbolt is thrown)
    grasp the thumb-turn -> turn it back                               (if the thumb deadbolt is thrown)
    grasp the knob / lever (or press the push bar) -> turn -> pull / push the door open, the hand
    following the door round its arc with a small lead

Skills are generators: each yields actions and receives the next observation, so plans compose with
`yield from`. Palm commands are P-controllers on the embodiment's commanded palm target (privileged,
like everything here), so contact never winds the command up.
Poses held relative to a moving part (a turned knob, the swinging door, a key in the hand) are
recorded when the skill starts and re-applied to the part's current pose, so one skill serves every
embodiment and door geometry.
"""

from __future__ import annotations

import numpy as np

from pai.envs.embodiments import ArmHand, rot_error, rotvec_to_mat
from pai.skills.clearance import Clearance
from pai.skills.grasp_table import GRASPS

KP, KR = 4.0, 3.0
TURN_SPEED = 0.8        # rad/s for knobs, thumb-turns and keys
DOOR_LEAD = 0.12        # rad the door target runs ahead of the door
DOOR_GOAL = 0.9         # rad (door_opened is logged at 0.66)
DRAWER_GOAL = 0.85      # opening fraction
ROLLS = (0.0, np.pi / 4, -np.pi / 4)   # approach rolls about a bar tried for a grasp
KEY_ROLLS = (0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2)   # key bow: from above to along the shaft
KEY_TRIES = 2          # grasps tried on a key before giving up
TRANSIT_CLEAR = 0.42   # m in front of the door's hinge line where the hand changes height and turns
KEY_PUSH = 0.034        # m past the keyhole mouth the key tip is pushed (full depth 0.03)


class Failed(Exception):
    pass


class HomeDoorOracle:
    def __init__(self, env, max_steps: int = 3000):
        self.env, self.emb, self.rt = env, env.emb, env.runtime
        self.dt = env.control_dt
        self.max_steps = max_steps
        self.stage = "start"
        self.result = ""
        self._gen = None
        self.t = 0
        self.force_key: str | None = None   # use this key instead of the matching one (e.g. to show a wrong key)
        self.close_level = -1.0             # the grasp command a "closed" hand gets (per grasp, see grasp_params)
        # per part class: how this body grasps it (pai.skills.grasp_table); a class without an entry is grasped
        # the way that fits best (Clearance)
        self.grasp_params: dict = dict(GRASPS.get(self.emb.name, {}))

    # ------------------------------------------------------------------ driver
    def done(self) -> bool:
        return self.stage in ("done", "failed")

    def act(self, obs) -> np.ndarray:
        self.t += 1
        if self.done():
            action = None
        else:
            try:
                if self._gen is None:
                    self._gen = self._plan(obs)
                    action = next(self._gen)
                else:
                    action = self._gen.send(obs)
            except StopIteration:
                self.stage = "done"
                action = None
            except Failed as e:
                self.stage, self.result = "failed", str(e)
                action = None
        if action is None:
            return np.r_[np.zeros(6), 1.0]
        if self.t > self.max_steps and not self.done():
            self.stage, self.result = "failed", f"timeout in {self.stage}"
        return action

    # ------------------------------------------------------------------ primitives
    def _cmd(self, obs, pos, R, grasp):
        # Steer the commanded palm target, not the measured palm: a hand blocked by contact (a key at the end
        # of its travel, a stiff door) then pushes with a bounded force instead of winding the target up.
        e = self.emb
        if grasp < 0:
            grasp = self.close_level
        return np.r_[KP * (pos - e.target_pos), KR * rot_error(e.target_rot, R), grasp]

    def _goto(self, obs, pos, R, grasp, tol=0.012, rtol=0.12, steps=250, hold=0):
        """Move the palm to (pos, R). Returns the last observation."""
        for k in range(steps):
            err = np.linalg.norm(pos - obs["palm_pos"])
            rerr = np.linalg.norm(rot_error(obs["palm_rot"], R))
            if err < tol and rerr < rtol:
                break
            obs = yield self._cmd(obs, pos, R, grasp)
        for _ in range(hold):
            obs = yield self._cmd(obs, pos, R, grasp)
        return obs

    def _hold(self, obs, grasp, steps, pos=None, R=None):
        pos = obs["palm_pos"].copy() if pos is None else pos
        R = obs["palm_rot"].copy() if R is None else R
        for _ in range(steps):
            obs = yield self._cmd(obs, pos, R, grasp)
        return obs

    def _wall(self, obs):
        """(outward horizontal normal of the door wall, a point on the hinge line): the transit frame."""
        if getattr(self, "_wall_frame", None) is None:
            a = obs["articulations"]
            d = a["door"]
            n = rotvec_to_mat(-d["joint_axis"] * d["q"]) @ a["handle"]["normal"]   # as when the door is shut
            n[2] = 0.0
            self._wall_frame = (n / np.linalg.norm(n), a["door"]["joint_anchor"].copy())
        return self._wall_frame

    def _transit(self, obs, pos, R, grasp, clear=TRANSIT_CLEAR):
        """Go to (pos, R) through free space: back off from the wall to `clear` (m in front of the hinge
        line) if the hand is closer, change height and orientation out there, then move in. A straight
        move would sweep the hand up through a shelf or down across a table top."""
        n, w0 = self._wall(obs)
        u = lambda p: float(np.dot(p - w0, n))   # noqa: E731
        cur = obs["palm_pos"].copy()
        if np.linalg.norm(pos - cur) < 0.06:
            return obs
        us = max(u(cur), u(pos), clear)
        if u(cur) < us - 0.02:
            obs = yield from self._goto(obs, cur + (us - u(cur)) * n, obs["palm_rot"].copy(), grasp, tol=0.03,
                                        rtol=0.3, steps=150)
        mid = pos + (us - u(pos)) * n
        obs = yield from self._goto(obs, mid, R, grasp, tol=0.04, rtol=0.2, steps=250)
        return obs

    def _grasp(self, obs, pos_fn, axis, approach, standoff=0.1, close_steps=40, R=None, offset=None):
        """Pre-grasp in front of a bar-like handle, straight in, close in place. R: the palm orientation
        (default: the body's grasp_rot for the bar axis and approach); offset: of the grasp centre from the
        part, in the palm frame."""
        if R is None:
            R = self.emb.grasp_rot(axis, approach, obs["palm_rot"])
        if offset is not None and np.any(offset):
            f, off = pos_fn, R @ np.asarray(offset, float)
            pos_fn = lambda o: f(o) + off   # noqa: E731
        z = R[:, 2]
        # turn into the grasp orientation well clear of the handle (a turning hand sweeps things off tables)
        obs = yield from self._transit(obs, pos_fn(obs) - (standoff + 0.12) * z, R, 1.0)
        obs = yield from self._goto(obs, pos_fn(obs) - (standoff + 0.12) * z, R, 1.0, tol=0.04, rtol=0.2, steps=250)
        obs = yield from self._goto(obs, pos_fn(obs) - standoff * z, R, 1.0, tol=0.02, rtol=0.15, steps=200)
        obs = yield from self._goto(obs, pos_fn(obs), R, 1.0, tol=0.006, steps=150)
        obs = yield from self._hold(obs, -1.0, close_steps)
        return obs

    def _release(self, obs, back: np.ndarray, dist=0.1, up=0.0):
        obs = yield from self._hold(obs, 1.0, 25)
        p = obs["palm_pos"] + dist * back
        obs = yield from self._goto(obs, p, obs["palm_rot"].copy(), 1.0, tol=0.02, steps=120)
        if up:
            obs = yield from self._goto(obs, obs["palm_pos"] + [0, 0, up], obs["palm_rot"].copy(), 1.0, tol=0.02, steps=120)
        return obs

    def _rotate(self, obs, part: str, angle: float, done_fn, grasp=-1.0, extra=0.0, steps=250):
        """Turn a grasped part about its joint axis by up to `angle`, keeping the grasp's pose relative to it."""
        a = obs["articulations"][part]
        axis, c = a["joint_axis"].copy(), a["joint_anchor"].copy()
        p0, R0 = obs["palm_pos"].copy(), obs["palm_rot"].copy()
        alpha = 0.0
        sign = np.sign(angle)
        for k in range(steps):
            if done_fn(obs):
                break
            alpha = sign * min(abs(alpha) + TURN_SPEED * self.dt, abs(angle) + extra)
            Rr = rotvec_to_mat(axis * alpha)
            obs = yield self._cmd(obs, c + Rr @ (p0 - c), Rr @ R0, grasp)
        else:
            raise Failed(f"turn {part}: not done after {steps} steps")
        return obs

    def _follow_door(self, obs, grasp=-1.0, goal=DOOR_GOAL, steps=700):
        """Carry the palm round the door's arc with a lead; ends when the door is open (or it stalls)."""
        a = obs["articulations"]["door"]
        axis, c = a["joint_axis"].copy(), a["joint_anchor"].copy()
        q0 = a["q"]
        p0, R0 = obs["palm_pos"].copy(), obs["palm_rot"].copy()
        best, best_k = q0, 0
        for k in range(steps):
            q = obs["articulations"]["door"]["q"]
            if q >= goal:
                return obs
            if q > best + 0.01:
                best, best_k = q, k
            if k - best_k > 80:
                raise Failed(f"door stalled at {q:.2f} rad")
            Rr = rotvec_to_mat(axis * (min(q + DOOR_LEAD, goal + 0.1) - q0))
            obs = yield self._cmd(obs, c + Rr @ (p0 - c), Rr @ R0, grasp)
        raise Failed("door: timeout")

    # ------------------------------------------------------------------ grasp choice
    @property
    def clearance(self) -> Clearance:
        if getattr(self, "_clearance", None) is None:
            self._clearance = Clearance(self.env)
        return self._clearance

    def _grasp_cands(self, obs, axis, approach, symmetric=False, rolls=(0.0,)) -> list:
        """Palm orientations that grasp a bar (axis) from `approach`: both ways round the bar, the approach
        rolled about the bar by `rolls` (rad), and for a round part (symmetric about the approach: a knob) the
        bar axis anywhere across it. The first is the body's default (grasp_rot)."""
        ap = np.asarray(approach, float) / np.linalg.norm(approach)
        axes = [np.asarray(axis, float)]
        if symmetric:
            axes += [rotvec_to_mat(ap * phi) @ axes[0] for phi in np.arange(1, 6) * np.pi / 6]
        out = []
        for a in axes:
            for psi in rolls:
                z = rotvec_to_mat(a / np.linalg.norm(a) * psi) @ ap
                R1 = self.emb.grasp_rot(a, z, obs["palm_rot"])
                out += [R1, self.emb.grasp_rot(a, z, R1 @ np.diag([-1.0, -1.0, 1.0]))]
        return out

    def _choose(self, pos, cands, ignore=(), sweep=None, standoff=0.1, closure=0.6):
        """The candidate orientation whose body fits best (Clearance) at the pre-grasp (open hand), at the
        grasp and along `sweep(pos, R) -> [(pos, R, joints)]` (what the grasp is for). Ties keep the order."""
        best, best_c = None, np.inf
        clr = self.clearance
        for R in cands:
            poses = [(pos - standoff * R[:, 2], R, {}, 0.0), (pos, R, {}, closure)]
            poses += [(p, Rr, j, closure) for p, Rr, j in (sweep(pos, R) if sweep else [])]
            c, q0 = 0.0, None
            for p, Rr, j, cl in poses:
                c += clr.cost(p, Rr, closure=cl, ignore=ignore, joints=j, q0=q0)
                q0 = clr.last_q if isinstance(self.emb, ArmHand) else None
            if c < best_c - 1e-3:
                best, best_c = R, c
        self.last_choice_cost = best_c
        return best

    @staticmethod
    def _grasp_R(axis, approach, roll=0.0, flip=0, spin=0.0) -> np.ndarray:
        """Palm orientation for a bar `axis` (turned by `spin` about the approach: a knob's grasp across its
        face), approached along `approach` rolled by `roll` about the bar; palm y = +bar axis (flip 0) or -."""
        ap = np.asarray(approach, float) / np.linalg.norm(approach)
        a = rotvec_to_mat(ap * spin) @ np.asarray(axis, float)
        a /= np.linalg.norm(a)
        z = rotvec_to_mat(a * roll) @ ap
        z = z - np.dot(z, a) * a
        z /= np.linalg.norm(z)
        y = a if not flip else -a
        return np.column_stack([np.cross(y, z), y, z])

    def _part_grasp(self, obs, cls, pos, axis, approach, symmetric=False, ignore=(), sweep=None, rolls=None):
        """(palm orientation, grasp-centre offset) for grasping a part of class `cls`, and set the closing
        command: this body's entry in grasp_params, else the candidate that fits best (Clearance)."""
        p = self.grasp_params.get(cls)
        if p:
            self.close_level = float(p.get("grip", -1.0))
            return (self._grasp_R(axis, approach, p.get("roll", 0.0), p.get("flip", 0), p.get("spin", 0.0)),
                    np.asarray(p.get("off", (0.0, 0.0, 0.0)), float))
        self.close_level = -1.0
        cands = self._grasp_cands(obs, axis, approach, symmetric=symmetric,
                                  rolls=(0.0,) if symmetric else (ROLLS if rolls is None else rolls))
        return self._choose(pos, cands, ignore=ignore, sweep=sweep), np.zeros(3)

    @staticmethod
    def _about(axis, c, ang, pos, R):
        Rr = rotvec_to_mat(np.asarray(axis, float) * ang)
        return c + Rr @ (pos - c), Rr @ R

    # ------------------------------------------------------------------ skills
    def _open_drawer(self, obs):
        self.stage = "drawer"
        a = obs["articulations"]["drawer"]
        rt = self.rt

        def sweep(pos, R):
            return [(pos + a["pull_dir"] * 0.2, R, {rt.drawer["q"]: 0.2})]
        R, off = self._part_grasp(obs, "drawer", a["handle_pos"], a["handle_axis"], -a["normal"],
                                  ignore=(self._body_of(rt.drawer["jnt"]),), sweep=sweep)
        obs = yield from self._grasp(obs, lambda o: o["articulations"]["drawer"]["handle_pos"], a["handle_axis"], -a["normal"],
                                     R=R, offset=off)
        d = a["pull_dir"].copy()
        p0, R0 = obs["palm_pos"].copy(), obs["palm_rot"].copy()
        for k in range(400):
            op = obs["articulations"]["drawer"]["opening"]
            if op >= DRAWER_GOAL:
                break
            s = (op + 0.12) * 0.25
            obs = yield self._cmd(obs, p0 + d * s, R0, -1.0)
        else:
            raise Failed("drawer did not open")
        obs = yield from self._release(obs, a["normal"], 0.08, up=0.15)
        return obs

    def _fetch_and_turn_key(self, obs):
        info = self.env.info
        name = self.force_key or next(n for n, k in info["keys"].items() if k["matches"])
        if info["keys"][name]["slot"][0] == "other_room":
            raise Failed("the key is in the other room (unsolvable)")
        if info["keys"][name]["slot"][0] == "drawer":
            obs = yield from self._open_drawer(obs)
        self.stage = "grasp_key"
        kid = next(k for k in self.rt.keys if k["name"] == name)
        d = self.env.data

        def key_pose():
            return d.xpos[kid["body"]].copy(), d.xmat[kid["body"]].reshape(3, 3).copy()

        grip = lambda o: d.site_xpos[kid["grip"]].copy()   # noqa: E731  (privileged: also inside a drawer)
        in_drawer = self.env.info["keys"][name]["slot"][0] == "drawer"
        kl = obs["articulations"]["key_lock"]
        x, mouth = kl["keyhole_axis"], kl["keyhole_pos"]
        jax = kl["joint_axis"]
        plug_q = self.rt.kl["plug_q"]
        clr = self.clearance

        def slot_frames():
            """The two key poses that fit the keyhole (the blade fits either way round)."""
            return [np.column_stack([x, s * kl["slot_dir"], np.cross(x, s * kl["slot_dir"])]) for s in (1, -1)]

        def insert_cost(rel_p, rel_R, closure=0.6):
            """Clearance of the body holding the key (palm pose rel_p, rel_R in the key frame) in the lock and
            through the quarter turn, for the better of the two ways round; and that way (the key pose)."""
            best = (np.inf, None)
            for Rd in slot_frames():
                c, q0 = 0.0, None
                for th in (0.0, np.pi / 4, np.pi / 2):
                    p, R = self._about(jax, mouth, th, mouth + 0.028 * x + Rd @ rel_p, Rd @ rel_R)
                    c += clr.cost(p, R, closure=closure, ignore=(kid["body"],), joints={plug_q: th}, q0=q0)
                    q0 = clr.last_q if isinstance(self.emb, ArmHand) else None
                if c < best[0] - 1e-3:
                    best = (c, Rd)
            return best

        cls = "key_drawer" if in_drawer else "key"
        pk0, Rk = key_pose()
        g0 = grip(obs)
        table = self.grasp_params.get(cls)
        if table:
            self.close_level = float(table.get("grip", -1.0))
            ranked = [(self._grasp_R(Rk[:, 1], [0, 0, -1.0], table.get("roll", 0.0), table.get("flip", 0)),
                       np.asarray(table.get("off", (0.0, 0.0, 0.0)), float))]
        else:
            # every way round the bow, from above to along the shaft (towards the tip or from it); ranked by how
            # well the body fits at the bow, lifted, and holding the key in the lock through the turn
            self.close_level = -1.0
            scored = []
            for R in self._grasp_cands(obs, Rk[:, 1], [0, 0, -1.0], rolls=KEY_ROLLS):
                z = R[:, 2]
                c = clr.cost(g0 - 0.12 * z, R, closure=0.0, ignore=(kid["body"],))
                c += clr.cost(g0, R, closure=0.6, ignore=(kid["body"],))
                c += clr.cost(g0 + [0, 0, 0.2], R, closure=0.6, ignore=(kid["body"],))
                c += insert_cost(Rk.T @ (g0 - pk0), Rk.T @ R)[0]
                scored.append((c, len(scored), R))
            ranked = [(R, np.zeros(3)) for _, _, R in sorted(scored, key=lambda t: t[:2])[:KEY_TRIES]]
        held = False
        for R, off in ranked:
            z0 = key_pose()[0][2]
            obs = yield from self._grasp(obs, grip, Rk[:, 1], R[:, 2], standoff=0.12, R=R, offset=off)
            for lift in ([0, 0, 0.1], [0, 0, 0.1]):
                obs = yield from self._goto(obs, obs["palm_pos"] + lift, obs["palm_rot"].copy(), -1.0, tol=0.02,
                                            steps=150)
            held = key_pose()[0][2] - z0 > 0.1 and np.linalg.norm(grip(obs) - obs["palm_pos"]) < 0.06
            if held:
                break
            obs = yield from self._release(obs, -R[:, 2], 0.1, up=0.05)
            if self.rt._left_slot(kid) and not in_drawer:
                break                         # knocked off its cradle: another try would not find it there
        if not held:
            raise Failed("could not pick up the key")
        pk, Rk = key_pose()
        self.stage = "insert_key"
        # the way round that fits the body best through the turn (a body swinging down hits the handle under
        # the lock, one on the latch side the jamb), as the key sits in the hand now
        _, Rd = insert_cost(Rk.T @ (obs["palm_pos"] - pk), Rk.T @ obs["palm_rot"], closure=None)
        if Rd is None:
            Rd = slot_frames()[0]

        def aligned_cmd(obs, tip_goal):
            """Palm command that puts the key (as it sits in the hand now) at tip_goal with pose Rd."""
            pk, Rk_now = key_pose()
            rel_R = Rk_now.T @ obs["palm_rot"]
            Rp = Rd @ rel_R
            pos = self.emb.target_pos + (tip_goal - pk) + (Rd - Rk_now) @ (Rk_now.T @ (obs["palm_pos"] - pk))
            return self._cmd(obs, pos, Rp, -1.0), pk, Rk_now

        def palm_for(obs, tip_goal):
            """Palm pose that puts the key (as it sits in the hand now) at tip_goal with pose Rd."""
            pk, Rk_now = key_pose()
            return tip_goal + Rd @ (Rk_now.T @ (obs["palm_pos"] - pk)), Rd @ (Rk_now.T @ obs["palm_rot"])

        def in_hand(obs):
            return np.linalg.norm(grip(obs) - obs["palm_pos"]) < 0.06

        # carry it out in front of the wall first (a straight move would drag it across the shelf edge)
        obs = yield from self._transit(obs, *palm_for(obs, mouth - 0.12 * x), -1.0)
        for goal_depth, tol, steps in ((-0.12, 0.02, 400), (-0.03, 0.004, 300)):
            for k in range(steps):
                a, pk, Rk_now = aligned_cmd(obs, mouth + goal_depth * x)
                off = pk - (mouth + goal_depth * x)
                if (np.linalg.norm(off) < tol and np.dot(Rk_now[:, 0], x) > 0.995
                        and abs(np.dot(Rk_now[:, 1], Rd[:, 1])) > 0.99):
                    break
                obs = yield a
                if not in_hand(obs):
                    raise Failed("dropped the key")
        for k in range(300):   # slide in along the axis, correcting the key pose as measured
            if self.rt.inserted:
                break
            pk, _ = key_pose()
            depth = float(np.dot(pk - mouth, x))
            a, _, _ = aligned_cmd(obs, mouth + min(depth + 0.015, KEY_PUSH) * x)
            obs = yield a
        else:
            raise Failed("key did not go in")
        self.stage = "turn_key"
        bolt = self.rt.bolt_q["key"]
        obs = yield from self._rotate(obs, "key_lock", np.pi / 2, lambda o: self.env.data.qpos[bolt] < 0.007,
                                      extra=0.25, steps=300)
        obs = yield from self._release(obs, -x, 0.1)
        return obs

    def _body_of(self, jnt: int) -> int:
        return int(self.env.model.jnt_bodyid[jnt])

    def _unlock_thumb(self, obs):
        self.stage = "thumb"
        a = obs["articulations"]["thumb_turn"]
        rt = self.rt
        turn = -a["q"] - 0.2

        def sweep(pos, R):
            return [(*self._about(a["joint_axis"], a["joint_anchor"], f * turn, pos, R),
                     {rt.thumb["q"]: a["q"] + f * turn}) for f in (0.5, 1.0)]
        R, off = self._part_grasp(obs, "thumb", a["handle_pos"], a["handle_axis"], -a["normal"],
                                  ignore=(self._body_of(rt.thumb["jnt"]),), sweep=sweep)
        obs = yield from self._grasp(obs, lambda o: o["articulations"]["thumb_turn"]["handle_pos"], a["handle_axis"],
                                     -a["normal"], R=R, offset=off)
        obs = yield from self._rotate(obs, "thumb_turn", turn, lambda o: o["articulations"]["thumb_turn"]["q"] < 0.08)
        obs = yield from self._release(obs, a["normal"], 0.1)
        return obs

    def _open_door(self, obs):
        self.stage = "handle"
        h = obs["articulations"]["handle"]
        kind = h["kind"]
        rt = self.rt
        site = lambda o: o["articulations"]["handle"]["handle_pos"]   # noqa: E731
        door = obs["articulations"]["door"]
        if kind == "push_bar":
            R = self.emb.grasp_rot(h["handle_axis"], -h["normal"], obs["palm_rot"])
            z = R[:, 2]
            obs = yield from self._transit(obs, site(obs) - 0.1 * z, R, -1.0)
            obs = yield from self._goto(obs, site(obs) - 0.1 * z, R, -1.0, tol=0.02, steps=300)
            obs = yield from self._goto(obs, site(obs) + 0.035 * z, R, -1.0, tol=0.01, steps=150)
        else:
            thr = rt.rec.handle_threshold
            angle = 0.8 if kind.startswith("knob") else 0.55
            turned = min(thr + 0.15, angle)

            def sweep(pos, R):
                p1, R1 = self._about(h["joint_axis"], h["joint_anchor"], turned, pos, R)
                out = [(p1, R1, {rt.handle_q: turned})]
                for th in (0.25, 0.5):
                    out.append((*self._about(door["joint_axis"], door["joint_anchor"], th, p1, R1),
                                {rt.handle_q: turned, rt.door["q"]: th}))
                return out
            R, off = self._part_grasp(obs, kind, h["handle_pos"], h["handle_axis"], -h["normal"],
                                      symmetric=kind.startswith("knob"), ignore=(self._body_of(rt.handle_jnt),),
                                      sweep=sweep)
            obs = yield from self._grasp(obs, site, h["handle_axis"], -h["normal"], R=R, offset=off)
            obs = yield from self._rotate(obs, "handle", angle, lambda o: o["articulations"]["handle"]["q"] > thr + 0.08,
                                          extra=0.3)
        self.stage = "door"
        obs = yield from self._follow_door(obs)
        return obs

    def _plan(self, obs):
        info = self.env.info
        if info["key_locked"]:
            obs = yield from self._fetch_and_turn_key(obs)
        if info["deadbolt_locked"]:
            obs = yield from self._unlock_thumb(obs)
        obs = yield from self._open_door(obs)
        self.stage = "done"


def run_oracle(env, seed: int = 0, max_steps: int = 3000, frames: list | None = None, camera: str = "front",
               frame_every: int = 5, frame_size: int = 320, **options) -> dict:
    """Reset (options: lock_state, key_place) and run the oracle; returns the outcome and the events."""
    obs = env.reset(seed=seed, **options)
    oracle = HomeDoorOracle(env, max_steps)
    stable = True
    for t in range(max_steps + 40):
        obs = env.step(oracle.act(obs))
        if frames is not None and t % frame_every == 0:
            frames.append(env.render(camera, frame_size))
        if not env.stable():
            stable = False
            break
        if oracle.done():
            break
    opened = any(e["type"] == "door_opened" for e in env.runtime.events)
    return {"opened": opened, "stage": oracle.stage, "why": oracle.result, "stable": stable, "steps": env.t,
            "door_q": float(obs["articulations"]["door"]["q"]), "info": env.info,
            "events": [e["type"] for e in env.runtime.events]}
