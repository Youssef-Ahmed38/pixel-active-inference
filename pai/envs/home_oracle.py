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

from pai.envs.embodiments import rot_error, rotvec_to_mat

KP, KR = 4.0, 3.0
TURN_SPEED = 0.8        # rad/s for knobs, thumb-turns and keys
DOOR_LEAD = 0.12        # rad the door target runs ahead of the door
DOOR_GOAL = 0.9         # rad (door_opened is logged at 0.66)
DRAWER_GOAL = 0.85      # opening fraction
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

    def _grasp(self, obs, pos_fn, axis, approach, standoff=0.1, close_steps=40):
        """Pre-grasp in front of a bar-like handle, straight in, close in place."""
        R = self.emb.grasp_rot(axis, approach, obs["palm_rot"])
        z = R[:, 2]
        # turn into the grasp orientation well clear of the handle (a turning hand sweeps things off tables)
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

    # ------------------------------------------------------------------ skills
    def _open_drawer(self, obs):
        self.stage = "drawer"
        a = obs["articulations"]["drawer"]
        obs = yield from self._grasp(obs, lambda o: o["articulations"]["drawer"]["handle_pos"], a["handle_axis"], -a["normal"])
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

        _, Rk = key_pose()
        # take the bow from above (it sits 7 cm above its cradle) and lift; if the key is not in the hand
        # afterwards, put it back down and try from the front (along the shaft) instead
        grip = lambda o: d.site_xpos[kid["grip"]].copy()   # noqa: E731  (privileged: also inside a drawer)
        for attempt, approach in enumerate((np.array([0, 0, -1.0]), Rk[:, 0].copy())):
            if attempt and self.env.info["keys"][name]["slot"][0] == "drawer":
                break                         # the drawer front is in the way of a front grasp
            z0 = key_pose()[0][2]
            obs = yield from self._grasp(obs, grip, key_pose()[1][:, 1], approach, standoff=0.12)
            for lift in ([0, 0, 0.1], [0, 0, 0.1]):
                obs = yield from self._goto(obs, obs["palm_pos"] + lift, obs["palm_rot"].copy(), -1.0, tol=0.02,
                                            steps=150)
            held = key_pose()[0][2] - z0 > 0.1 and np.linalg.norm(grip(obs) - obs["palm_pos"]) < 0.06
            if held:
                break
            obs = yield from self._release(obs, -approach, 0.1, up=0.05)
        else:
            raise Failed("could not pick up the key")
        if not held:
            raise Failed("could not pick up the key")
        pk, Rk = key_pose()
        # align in front of the keyhole (the blade fits either way round: take the nearer)
        self.stage = "insert_key"
        kl = obs["articulations"]["key_lock"]
        x, mouth = kl["keyhole_axis"], kl["keyhole_pos"]
        # The blade fits either way round. Pick the way that keeps the hand's body (behind the palm, -z) on
        # the hinge side: the lock is next to the latch jamb, which stands proud of the door face. The
        # quarter turn then also swings the bow towards the hinge side, away from jamb and thumb-turn.
        hinge = obs["articulations"]["door"]["joint_anchor"] - mouth
        hinge[2] = 0.0
        hinge /= np.linalg.norm(hinge)
        cands = []
        for s in (1, -1):
            y = s * kl["slot_dir"]
            Rd = np.column_stack([x, y, np.cross(x, y)])
            Rp = Rd @ (Rk.T @ obs["palm_rot"])
            cands.append((float(np.dot(-Rp[:, 2], hinge)), Rd))
        _, Rd = max(cands, key=lambda c: c[0])

        def aligned_cmd(obs, tip_goal):
            """Palm command that puts the key (as it sits in the hand now) at tip_goal with pose Rd."""
            pk, Rk_now = key_pose()
            rel_R = Rk_now.T @ obs["palm_rot"]
            Rp = Rd @ rel_R
            pos = self.emb.target_pos + (tip_goal - pk) + (Rd - Rk_now) @ (Rk_now.T @ (obs["palm_pos"] - pk))
            return self._cmd(obs, pos, Rp, -1.0), pk, Rk_now

        for goal_depth, tol, steps in ((-0.12, 0.02, 400), (-0.03, 0.004, 300)):
            for k in range(steps):
                a, pk, Rk_now = aligned_cmd(obs, mouth + goal_depth * x)
                off = pk - (mouth + goal_depth * x)
                if (np.linalg.norm(off) < tol and np.dot(Rk_now[:, 0], x) > 0.995
                        and abs(np.dot(Rk_now[:, 1], Rd[:, 1])) > 0.99):
                    break
                obs = yield a
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
        obs = yield from self._rotate(obs, "key_lock", np.pi / 2, lambda o: self.env.data.qpos[bolt] < 0.002,
                                      extra=0.25, steps=300)
        obs = yield from self._release(obs, -x, 0.1)
        return obs

    def _unlock_thumb(self, obs):
        self.stage = "thumb"
        a = obs["articulations"]["thumb_turn"]
        obs = yield from self._grasp(obs, lambda o: o["articulations"]["thumb_turn"]["handle_pos"], a["handle_axis"], -a["normal"])
        obs = yield from self._rotate(obs, "thumb_turn", -a["q"] - 0.2, lambda o: o["articulations"]["thumb_turn"]["q"] < 0.08)
        obs = yield from self._release(obs, a["normal"], 0.1)
        return obs

    def _open_door(self, obs):
        self.stage = "handle"
        h = obs["articulations"]["handle"]
        kind = h["kind"]
        site = lambda o: o["articulations"]["handle"]["handle_pos"]   # noqa: E731
        if kind == "push_bar":
            R = self.emb.grasp_rot(h["handle_axis"], -h["normal"], obs["palm_rot"])
            z = R[:, 2]
            obs = yield from self._goto(obs, site(obs) - 0.1 * z, R, -1.0, tol=0.02, steps=300)
            obs = yield from self._goto(obs, site(obs) + 0.035 * z, R, -1.0, tol=0.01, steps=150)
        else:
            obs = yield from self._grasp(obs, site, h["handle_axis"], -h["normal"])
            thr = self.rt.rec.handle_threshold
            angle = 0.8 if kind.startswith("knob") else 0.55
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
