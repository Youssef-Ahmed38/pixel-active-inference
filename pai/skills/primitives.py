"""Generic motor skills: small closed-loop controllers over the common embodiment action.

Each skill is `fn(ctx, target, **params) -> SkillResult` (see core.py). It reads only the common
observation (palm pose, finger forces, opening) and the target's frame, and commands only palm
velocities and the grasp; the same code drives the Panda, the floating hands and the G1.

- reach(target, approach)      palm to the target, oriented for a grasp (Embodiment.grasp_rot): first to a
                               standoff along the approach direction, then straight in
- grasp()                      close the hand in place; success = at least two fingers feel contact
- release()                    open the hand in place; forgets the held object
- turn(axis, angle)            rotate the palm about an axis through the grasped point (a knob, a key, a
                               lever), at a fixed rate
- push_pull(direction, distance, max_force)
                               move the palm along a direction at a fixed speed; with a target the
                               direction and the grip rotate with the target's frame, so a grasped
                               door handle is carried round its arc. Stops at the force limit
- insert(target, held, axis)   bring the tip of a held object to a site, aligned with the axis, and push
                               it in along the axis
- retreat()                    open the hand and back away along the approach direction

Every skill that commands motion runs a StallMonitor: when the commanded motion does not happen (the
drawer is held shut, the key does not go in, the knob does not turn), it stops early and reports
`stalled`. What stalled means for the world is not decided here; that is the agent's to learn.
"""

from __future__ import annotations

import numpy as np

from pai.envs.embodiments import rot_error, rotvec_to_mat
from pai.skills.core import Recorder, SkillContext, SkillResult, StallMonitor, Target, _unit


def _hold_pose(ctx: SkillContext):
    return ctx.obs["palm_pos"].copy(), ctx.obs["palm_rot"].copy()


def _align_rotvec(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Smallest rotation vector taking direction a to direction u."""
    c = np.cross(a, u)
    s, cosang = np.linalg.norm(c), float(np.dot(a, u))
    if s < 1e-9:
        if cosang > 0:
            return np.zeros(3)
        perp = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        return _unit(perp) * np.pi
    return c / s * np.arctan2(s, cosang)


# ---------------------------------------------------------------------------------------------- reach
def reach(ctx: SkillContext, target: Target, approach=None, standoff: float = 0.1, offset=(0.0, 0.0, 0.0),
          tol: float = 0.01, rot_tol: float = 0.2, max_steps: int = 300) -> SkillResult:
    """Palm (grasp centre) to `target` (+ `offset` in its frame), facing along `approach` (default: the
    target's own approach direction). The grasp command is left as it is (open to reach a part, closed
    when carrying something)."""
    rec = Recorder(ctx, "reach", target)
    m, d = ctx.model, ctx.data
    p, Rt, axis, appr = target.world(m, d)
    appr = _unit(appr if approach is None else approach)
    R = ctx.emb.grasp_rot(axis, appr, ctx.obs["palm_rot"])
    off_local = np.asarray(offset, float)
    Rt0_inv_R = Rt.T @ R                        # the grasp orientation follows the target if it moves
    phase = "pregrasp" if standoff > 0 else "approach"
    mon = StallMonitor()
    err = rerr = np.inf
    stalled, k_phase = False, 0
    for _ in range(max_steps):
        p, Rt = target.frame(m, d)
        Rg = Rt @ Rt0_inv_R
        goal = p + Rt @ off_local - (standoff * Rg[:, 2] if phase == "pregrasp" else 0.0)
        err = float(np.linalg.norm(goal - ctx.obs["palm_pos"]))
        rerr = float(np.linalg.norm(rot_error(ctx.obs["palm_rot"], Rg)))
        if phase == "pregrasp" and ((err < 0.02 and rerr < 0.15) or k_phase > max_steps // 2):
            phase, k_phase, mon = "approach", 0, StallMonitor()
            continue
        if phase == "approach" and err < 0.6 * tol and rerr < rot_tol:
            break
        v, w = ctx.servo(goal, Rg)
        rec.step(v, w)
        k_phase += 1
        new_err = float(np.linalg.norm(goal - ctx.obs["palm_pos"]))
        if mon.update(np.linalg.norm(v) * ctx.dt, err - new_err):
            if phase == "pregrasp":       # the standoff is only a waypoint: go straight in from here
                phase, k_phase, mon = "approach", 0, StallMonitor()
                continue
            stalled = True
            break
        if rec.unstable:
            break
    p, Rt = target.frame(m, d)
    Rg = Rt @ Rt0_inv_R
    goal = p + Rt @ off_local
    err = float(np.linalg.norm(goal - ctx.obs["palm_pos"]))
    rerr = float(np.linalg.norm(rot_error(ctx.obs["palm_rot"], Rg)))
    ok = phase == "approach" and err < tol and rerr < rot_tol
    reason = "ok" if ok else ("stalled" if stalled else "timeout")
    return rec.result(ok, stalled, reason, pos_err=err, rot_err=rerr, phase=phase)


# ---------------------------------------------------------------------------------------------- grasp / release
def grasp(ctx: SkillContext, target: Target | None = None, steps: int = 40, min_fingers: int = 2,
          force_threshold: float = 0.3) -> SkillResult:
    """Close the hand where it is. Success: at least min(min_fingers, n_fingers) fingers feel contact.
    If the target is an object, it becomes the held object."""
    rec = Recorder(ctx, "grasp", target)
    pos, R = _hold_pose(ctx)
    c0 = float(np.mean(ctx.obs["finger_closure"]))
    for _ in range(steps):
        v, w = ctx.servo(pos, R)
        rec.step(v, w, grip=-1.0)
        if rec.unstable:
            break
    f = ctx.obs["finger_force"]
    n_touch = int(np.sum(f > force_threshold))
    ok = n_touch >= min(min_fingers, ctx.emb.n_fingers)
    closure = float(np.mean(ctx.obs["finger_closure"]))
    if ok and target is not None and target.kind == "object":
        ctx.held = target
    return rec.result(ok, False, "ok" if ok else "no_contact", fingers_in_contact=n_touch,
                      closure=closure, closure_change=closure - c0, grip_force=float(f.sum()))


def _open_compliant(ctx: SkillContext, rec: Recorder, steps: int, yield_gain: float, back_off: float,
                    ramp: int = 15) -> None:
    """Open the hand gradually (the grasp command ramps to open over `ramp` steps) while the palm backs off
    `back_off` along its normal and gives way to the contact force (admittance): fingers or a thumb
    snapping open against a panel would otherwise flick a free drawer or door shut."""
    pos, R = _hold_pose(ctx)
    g0 = ctx.grip
    n = max(steps, ramp)
    for k in range(n):
        pos = pos - R[:, 2] * back_off / n + np.clip(yield_gain * ctx.hand_force(), -0.004, 0.004)
        v, w = ctx.servo(pos, R)
        rec.step(v, w, grip=g0 + (1.0 - g0) * min(1.0, (k + 1) / ramp))
        if rec.unstable:
            break


def release(ctx: SkillContext, target: Target | None = None, steps: int = 25, yield_gain: float = 2e-4,
            back_off: float = 0.02) -> SkillResult:
    """Open the hand, backing off a little and yielding to contact (see _open_compliant); the held object
    is let go."""
    rec = Recorder(ctx, "release", target)
    _open_compliant(ctx, rec, steps, yield_gain, back_off)
    ctx.held = None
    opening = float(ctx.obs["opening"])
    ok = opening > 0.7
    return rec.result(ok, False, "ok" if ok else "blocked", opening=opening,
                      grip_force=float(np.sum(ctx.obs["finger_force"])))


# ---------------------------------------------------------------------------------------------- turn
def turn(ctx: SkillContext, target: Target | None = None, axis=None, angle: float = 0.5, speed: float = 0.8,
         pivot=None, tol: float = 0.1, settle_steps: int = 25) -> SkillResult:
    """Rotate the palm by `angle` (rad, right-handed) about `axis` through `pivot`. Defaults: the target's
    axis (a knob's, a key's), else the palm normal; the pivot is the palm's grasp centre, i.e. the
    grasped point. The palm pose follows the rigid rotation, so a held knob or key is turned in place."""
    rec = Recorder(ctx, "turn", target)
    m, d = ctx.model, ctx.data
    p0, R0 = _hold_pose(ctx)
    if axis is None:
        axis = target.world(m, d)[2] if target is not None else R0[:, 2]
    u = _unit(axis)
    c = p0.copy() if pivot is None else np.asarray(pivot, float)
    sign = 1.0 if angle >= 0 else -1.0
    mon = StallMonitor(min_cmd=0.1)
    th_cmd, achieved, stalled = 0.0, 0.0, False
    n_ramp = int(np.ceil(abs(angle) / (speed * ctx.dt)))
    for k in range(n_ramp + settle_steps):
        ramping = k < n_ramp
        th_cmd = sign * min(abs(angle), (k + 1) * speed * ctx.dt)
        Q = rotvec_to_mat(u * th_cmd)
        v, w = ctx.servo(c + Q @ (p0 - c), Q @ R0)
        if ramping:          # feed-forward of the rotation rate and of the pivot's arc
            w = w + sign * speed * u
            v = v + sign * speed * np.cross(u, ctx.obs["palm_pos"] - c)
        rec.step(v, w)
        prev = achieved
        achieved = float(np.dot(rot_error(R0, ctx.obs["palm_rot"]), u))
        if ramping and mon.update(speed * ctx.dt, sign * (achieved - prev)):
            stalled = True
            break
        if not ramping and abs(achieved - angle) < 0.5 * tol:
            break
        if rec.unstable:
            break
    err = abs(achieved - angle)
    ok = err < tol and not stalled
    return rec.result(ok, stalled, "ok" if ok else ("stalled" if stalled else "timeout"),
                      angle=achieved, angle_err=err,
                      pivot_drift=float(np.linalg.norm(ctx.obs["palm_pos"] - (c + rotvec_to_mat(u * achieved) @ (p0 - c)))))


# ---------------------------------------------------------------------------------------------- push / pull
def _direction(ctx: SkillContext, target: Target | None, direction) -> np.ndarray:
    if isinstance(direction, str):
        appr = target.world(ctx.model, ctx.data)[3] if target is not None else ctx.obs["palm_rot"][:, 2]
        if direction not in ("pull", "push"):
            raise ValueError(f"direction must be 'pull', 'push' or a vector, not {direction!r}")
        return -appr if direction == "pull" else appr
    return _unit(direction)


def push_pull(ctx: SkillContext, target: Target | None = None, direction="pull", distance: float = 0.1,
              speed: float = 0.06, max_force: float = 80.0, follow: bool = True,
              max_steps: int | None = None, stall_window: int = 50) -> SkillResult:
    """Move the palm `distance` along `direction` ("pull" = against the target's approach, "push" = with
    it, or a world vector) at `speed`, keeping the grasp orientation. With a target and `follow`, the
    direction and the palm pose relative to the target are held in the target's frame (a door handle
    swings: the pull turns with it). Stops when the resisting force (net contact force on the hand
    against the motion) stays above `max_force`, or when the motion stalls."""
    rec = Recorder(ctx, "push_pull", target)
    m, d = ctx.model, ctx.data
    p0, R0 = _hold_pose(ctx)
    dir0 = _direction(ctx, target, direction)
    use_frame = target is not None and follow
    if use_frame:
        pt0, Rt0 = target.frame(m, d)
        dir_l, p_rel, R_rel = Rt0.T @ dir0, Rt0.T @ (p0 - pt0), Rt0.T @ R0
    max_steps = max_steps or int(distance / (speed * ctx.dt) * 2) + 50
    mon = StallMonitor(window=stall_window)
    s, palm_s, stalled, over, reason, resist_max = 0.0, 0.0, False, 0, "timeout", 0.0
    for _ in range(max_steps):
        if use_frame:
            pt, Rt = target.frame(m, d)
            dvec, goal, Rg = Rt @ dir_l, pt + Rt @ p_rel, Rt @ R_rel
        else:
            dvec, goal, Rg = dir0, p0 + dir0 * s, R0
        v, w = ctx.servo(goal, Rg)
        v = v - np.dot(v, dvec) * dvec + speed * dvec
        before = ctx.obs["palm_pos"].copy()
        t_before = target.frame(m, d)[0] if target is not None else None
        rec.step(v, w)
        palm_s += float(np.dot(ctx.obs["palm_pos"] - before, dvec))
        # progress is the target's motion when there is one: a hand slipping off a locked handle moves,
        # the handle does not
        moving = target.frame(m, d)[0] - t_before if target is not None else ctx.obs["palm_pos"] - before
        ds = float(np.dot(moving, dvec))
        s += ds
        resist = float(-np.dot(ctx.hand_force(), dvec))
        resist_max = max(resist_max, resist)
        over = over + 1 if resist > max_force else 0
        if s >= distance:
            reason = "ok"
            break
        if mon.update(speed * ctx.dt, ds):
            stalled, reason = True, "stalled"
            break
        if over >= 3:
            reason, stalled = "force_limit", mon.stopped()
            break
        if rec.unstable:
            break
    moved = float(np.linalg.norm(target.frame(m, d)[0] - pt0)) if use_frame else None
    errors = dict(progress=s, progress_err=max(0.0, distance - s), palm_progress=palm_s, resist_max=resist_max)
    if moved is not None:
        errors["target_moved"] = moved
    return rec.result(reason == "ok", stalled, reason, **errors)


# ---------------------------------------------------------------------------------------------- insert
def insert(ctx: SkillContext, target: Target, held: Target | None = None, axis=None, depth: float = 0.03,
           standoff: float = 0.04, speed: float = 0.04, tol: float = 0.008, max_force: float = 60.0,
           align_steps: int = 250) -> SkillResult:
    """Bring the tip of the held object (its frame; its axis = the direction it goes in) to `standoff`
    before the target site along `axis` (default: the target's axis, pointing into it), aligned, then push
    it `depth` past the site. Servoes the measured tip, so a grasp that is not centred is corrected."""
    rec = Recorder(ctx, "insert", target)
    held = held or ctx.held
    if held is None:
        return rec.result(False, False, "nothing_held")
    m, d = ctx.model, ctx.data

    def tip():
        p, _, a, _ = held.world(m, d)
        return p, a

    # 1. align: tip to the pre-insertion point, tip axis along the insertion axis
    aligned, a_err, p_err = False, np.inf, np.inf
    for _ in range(align_steps):
        site, _, t_axis, _ = target.world(m, d)
        u = _unit(axis) if axis is not None else t_axis
        tp, ta = tip()
        goal = site - standoff * u
        p_err, a_err = float(np.linalg.norm(goal - tp)), float(np.linalg.norm(_align_rotvec(ta, u)))
        if p_err < tol and a_err < 0.1:
            aligned = True
            break
        v = ctx.kp * (goal - tp)
        n = np.linalg.norm(v)
        v = v * ctx.emb.max_lin_vel / n if n > ctx.emb.max_lin_vel else v
        rec.step(v, ctx.kr * _align_rotvec(ta, u))
        if rec.unstable:
            break
    if not aligned:
        return rec.result(False, False, "not_aligned", pos_err=p_err, axis_err=a_err, depth=0.0)

    # 2. push along the axis, correcting the tip back onto the line
    site, _, t_axis, _ = target.world(m, d)
    u = _unit(axis) if axis is not None else t_axis
    mon = StallMonitor(min_cmd=0.015)
    stalled, reason, over = False, "timeout", 0
    for _ in range(int((standoff + depth) / (speed * ctx.dt) * 2) + 50):
        tp, ta = tip()
        rel = tp - site
        lateral = rel - np.dot(rel, u) * u
        v = -ctx.kp * lateral + speed * u
        rec.step(v, ctx.kr * _align_rotvec(ta, u))
        tp2, _ = tip()
        ds = float(np.dot(tp2 - tp, u))
        over = over + 1 if -np.dot(ctx.hand_force(), u) > max_force else 0
        if np.dot(tp2 - site, u) >= depth:
            reason = "ok"
            break
        if mon.update(speed * ctx.dt, ds):
            stalled, reason = True, "stalled"
            break
        if over >= 3:
            reason, stalled = "force_limit", mon.stopped()
            break
        if rec.unstable:
            break
    tp, ta = tip()
    rel = tp - site
    lat = float(np.linalg.norm(rel - np.dot(rel, u) * u))
    dep = float(np.dot(rel, u))
    ok = reason == "ok" and lat < 2 * tol
    return rec.result(ok, stalled, reason if ok or reason != "ok" else "off_axis", depth=dep, lateral_err=lat,
                      axis_err=float(np.linalg.norm(_align_rotvec(ta, u))))


# ---------------------------------------------------------------------------------------------- retreat
def retreat(ctx: SkillContext, target: Target | None = None, distance: float = 0.12, speed: float = 0.15,
            open_hand: bool = True, open_steps: int = 20) -> SkillResult:
    """Open the hand (optionally) and back away `distance` against the approach direction (the target's,
    else the palm normal), keeping the orientation."""
    rec = Recorder(ctx, "retreat", target)
    p0, R0 = _hold_pose(ctx)
    if open_hand:
        _open_compliant(ctx, rec, open_steps, 2e-4, 0.02)
        ctx.held = None
    appr = target.world(ctx.model, ctx.data)[3] if target is not None else R0[:, 2]
    dvec = -_unit(appr)
    p0 = ctx.obs["palm_pos"].copy()
    mon = StallMonitor()
    s, stalled, reason = 0.0, False, "timeout"
    for _ in range(int(distance / (speed * ctx.dt) * 2) + 40):
        v, w = ctx.servo(p0 + dvec * min(distance, s + 0.05), R0)
        v = v - np.dot(v, dvec) * dvec + speed * dvec
        before = ctx.obs["palm_pos"].copy()
        rec.step(v, w)
        ds = float(np.dot(ctx.obs["palm_pos"] - before, dvec))
        s += ds
        if s >= distance:
            reason = "ok"
            break
        if mon.update(speed * ctx.dt, ds):
            stalled, reason = True, "stalled"
            break
        if rec.unstable:
            break
    contact = float(np.linalg.norm(ctx.hand_force()))
    return rec.result(reason == "ok", stalled, reason, progress=s, contact_force=contact)
