"""Articulated fixtures (drawers, doors, sliding doors, lids, flaps) that can be added to any MjSpec.

Embodiment-independent: `add_fixture` builds a fixture at a pose under any parent body and returns
a small `Fixture` record (joint, body and handle-site names, type parameters). The Panda tabletop
scene (pai.envs.articulated) uses it; a humanoid or dexterous-hand scene calls the same functions.
Nothing here knows about the robot: which grasp or motion opens a fixture is the body's problem.

Fixture frame: origin at the bottom centre of the front plane, +x points out of the front towards
the user, +z is up, +y is the user's right. Every moving part opens with positive joint position
from 0 (closed), so the opening fraction q / range is comparable across families.

Families (each with named types in FIXTURE_TYPES, some held out for the test split):
- drawer        one drawer in a small cabinet; bar, knob, hbar (horizontal bar near the top) or
                flush (a low fin pull at the top edge) handle
- drawer_stack  2-3 stacked drawers, one of them the target; handles centred or staggered
- cabinet_door  hinged cabinet door, hinge left or right; bar, knob or lever handle
- room_door     full-size wall door at standing height (for a humanoid), lever/knob/bar
- sliding_door  a panel sliding sideways in front of a fixed one; bar or flush handle
- lid           box lid hinged at the back (horizontal axis); knob or tab handle
- flap          drop-down front flap hinged at the bottom (horizontal axis); knob or bar handle

Parameters per type: size, handle shape/height/offset/standoff, joint damping/friction ("smooth" or
"stiff" feel), spring return (self-closing) and lever direction. Hidden causes are applied per
episode by FixtureRuntime, never visible in the geometry:
- locked   an equality constraint holds the joint closed
- stuck    friction as if ~STUCK_FORCE newtons were needed at the handle
- blocked  a hidden obstacle: the joint range ends at BLOCKED_FRAC of its travel
- latched  (lever handles only) the joint is held closed unless the lever is turned past its
           threshold; once the part is ajar the latch no longer holds it
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

T = 0.012              # carcass panel thickness (m)
TF = 0.018             # front / door panel thickness (m)
GAP = 0.002            # clearance between fronts (m)
CLOSED_FRAC = 0.05     # opening below this counts as closed
OPEN_FRAC = 0.8        # opening above this counts as open
STUCK_FORCE = 60.0     # N at the handle that a stuck joint's friction corresponds to
STUCK_DAMPING = 5.0    # damping multiplier for stuck joints
BLOCKED_FRAC = 0.3     # a blocked joint stops at this fraction of its travel
LATCH_CATCH = {"hinge": 0.03, "slide": 0.004}  # the latch catches only this close to closed (rad / m)
CAUSES = ("none", "locked", "stuck", "blocked", "latched")
LEVER_RANGE = 0.7      # lever travel (rad)

WOOD = [0.62, 0.46, 0.3, 1.0]
METAL = [0.78, 0.78, 0.8, 1.0]
HANDLE_DENSITY = 2700.0
WOOD_DENSITY = 500.0
_geom = mujoco.mjtGeom


# ---------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class FixtureSpec:
    """A concrete fixture type: a registry type name, its family and sampled parameters."""
    type: str
    family: str
    params: dict


@dataclass
class Part:
    """One moving part (drawer, door, lid, ...). Names refer to elements in the compiled model."""
    name: str
    kind: str                      # slide | hinge
    joint: str
    body: str
    handle_site: str               # grasp point; frame x = outward face normal, z = handle long axis
    handle: str                    # bar | knob | hbar | lever | flush | tab
    range: tuple[float, float]
    lock_eq: str
    lever_joint: str | None = None
    lever_body: str | None = None
    latch_threshold: float = 0.3   # lever angle that releases the latch (rad)


@dataclass
class Fixture:
    name: str
    spec: FixtureSpec
    root: str
    parts: list[Part]
    keepout: tuple[float, float, float, float]  # local (xmin, xmax, ymin, ymax) incl. the swept area
    target: str                                 # the part a task should open
    bodies: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------- helpers
def _frame_quat(normal, up) -> list[float]:
    """Quaternion of the frame with x = normal and z = up (orthogonalised)."""
    x = np.asarray(normal, float)
    x = x / np.linalg.norm(x)
    z = np.asarray(up, float) - np.dot(up, x) * x
    z = z / np.linalg.norm(z)
    mat = np.column_stack([x, np.cross(z, x), z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q.tolist()


def _box(body, name: str, half, pos, rgba, quat=None, density: float = WOOD_DENSITY, **kw):
    return body.add_geom(name=name, type=_geom.mjGEOM_BOX, size=list(half), pos=list(pos),
                         quat=quat if quat is not None else [1, 0, 0, 0], rgba=list(rgba), density=density, **kw)


def _add_handle(body, prefix: str, shape: str, pos, normal, up, p: dict) -> str:
    """Adds a rigid handle on a face of `body`; `pos` is the attachment point on the face surface.
    Returns the grasp site's name (frame x = normal, z = the handle's long axis)."""
    n, u = np.asarray(normal, float), np.asarray(up, float)
    pos = np.asarray(pos, float)
    q = _frame_quat(n, u)
    s = float(p.get("standoff", 0.05))
    site = f"{prefix}_handle"
    if shape in ("bar", "hbar"):
        length = float(p.get("handle_len", 0.05))
        width = float(p.get("handle_w", 0.022)) if shape == "bar" else 0.015
        centre = pos + n * s
        _box(body, f"{prefix}_bar", [0.0075, width / 2, length / 2], centre, METAL, q, HANDLE_DENSITY)
        for k, sign in enumerate((-1, 1)):  # two posts back to the face
            _box(body, f"{prefix}_post{k}", [s / 2, width * 0.3, 0.005], pos + n * s / 2 + u * sign * (length / 2 - 0.006),
                 METAL, q, HANDLE_DENSITY)
        body.add_site(name=site, pos=centre.tolist(), quat=q)
    elif shape == "knob":
        r = float(p.get("knob_r", 0.018))
        body.add_geom(name=f"{prefix}_stem", type=_geom.mjGEOM_CAPSULE, size=[0.007, 0, 0],
                      fromto=[*pos, *(pos + n * (s - 0.01))], rgba=METAL, density=HANDLE_DENSITY)
        # A round knob with a flat-sided head (cylinder along the normal): pinched from the side it
        # gives line contacts; a sphere gives point contacts that roll and creep under load.
        body.add_geom(name=f"{prefix}_knob", type=_geom.mjGEOM_CYLINDER, size=[r, 0, 0],
                      fromto=[*(pos + n * (s - 0.011)), *(pos + n * (s + 0.011))], rgba=METAL, density=HANDLE_DENSITY)
        body.add_site(name=site, pos=(pos + n * s).tolist(), quat=q)
    elif shape == "flush":  # a low fin pull: 2 cm deep, 1.2 cm thick
        length = float(p.get("handle_len", 0.055))
        _box(body, f"{prefix}_fin", [0.01, 0.006, length / 2], pos + n * 0.01, METAL, q, HANDLE_DENSITY)
        body.add_site(name=site, pos=(pos + n * 0.01).tolist(), quat=q)
    elif shape == "tab":  # an upright block tab (on a lid), pinched across its thickness
        _box(body, f"{prefix}_tab", [0.02, 0.009, 0.015], pos + n * 0.02, METAL, q, HANDLE_DENSITY)
        body.add_site(name=site, pos=(pos + n * 0.022).tolist(), quat=q)
    else:
        raise ValueError(f"unknown handle shape {shape!r}")
    return site


def _add_lever(body, prefix: str, pos, normal, arm_dir, p: dict) -> tuple[str, str, str]:
    """A lever handle on its own spring-returned hinge (axis = face normal). The arm points along
    arm_dir; positive angle turns its free end down (lever_dir "down") or up ("up").
    Returns (grasp site, joint name, lever body name)."""
    n, a = np.asarray(normal, float), np.asarray(arm_dir, float)
    pos = np.asarray(pos, float)
    s, length = float(p.get("standoff", 0.05)), float(p.get("lever_len", 0.08))
    down = np.cross([0, 0, 1.0], a)  # rotating a about this axis moves its tip down
    axis = down if p.get("lever_dir", "down") == "down" else -down
    body.add_geom(name=f"{prefix}_rose", type=_geom.mjGEOM_CYLINDER, size=[0.02, 0.002, 0],
                  fromto=[*pos, *(pos + n * 0.004)], rgba=METAL, density=HANDLE_DENSITY)
    lever = body.add_body(name=f"{prefix}_lever", pos=(pos + n * s).tolist())
    joint = f"{prefix}_lever_joint"
    lever.add_joint(name=joint, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis.tolist(), range=[0, LEVER_RANGE],
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=[float(p.get("lever_spring", 0.4)), 0, 0],
                    damping=[0.01, 0, 0], armature=0.0005)
    lever.add_geom(name=f"{prefix}_hub", type=_geom.mjGEOM_CYLINDER, size=[0.011, 0, 0],
                   fromto=[*(-n * (s - 0.005)), *(n * 0.007)], rgba=METAL, density=HANDLE_DENSITY)
    q = _frame_quat(n, a)
    _box(lever, f"{prefix}_arm", [0.006, 0.008, length / 2], a * length / 2, METAL, q, HANDLE_DENSITY)
    site = f"{prefix}_handle"
    lever.add_site(name=site, pos=(a * length * float(p.get("lever_grasp", 0.6))).tolist(), quat=q)
    return site, joint, f"{prefix}_lever"


def _joint(body, name: str, kind: str, axis, rng_hi: float, p: dict, pos=(0, 0, 0)) -> None:
    body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE if kind == "slide" else mujoco.mjtJoint.mjJNT_HINGE,
                   axis=list(axis), pos=list(pos), range=[0, rng_hi], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                   damping=[float(p["damping"]), 0, 0], frictionloss=float(p["frictionloss"]),
                   stiffness=[float(p.get("stiffness", 0.0)), 0, 0], springref=0.0,
                   armature=0.01 if kind == "slide" else 0.005,
                   # Dry friction is a soft constraint in MuJoCo and creeps under a steady load; a
                   # harder impedance keeps a stuck joint stuck (creep < 1 mm/s at a third of it).
                   solimp_friction=[0.99, 0.999, 0.001, 0.5, 2.0])


def _lock(spec: mujoco.MjSpec, joint: str) -> str:
    """Joint-equality lock holding the joint at its closed position; toggled per episode."""
    name = f"{joint}_lock"
    spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=joint,
                      active=False, data=[0.0] * 11, solref=[0.005, 1.0])
    return name


def _carcass(root, name: str, W: float, D: float, H: float, z0: float, rgba, open_top: bool = False,
             dividers: int = 0) -> None:
    """Open-fronted box x in [-D, 0], y in [-W/2, W/2], z in [z0, z0 + H], on a set-back plinth."""
    _box(root, f"{name}_bottom", [D / 2, W / 2, T / 2], [-D / 2, 0, z0 + T / 2], rgba)
    if not open_top:
        _box(root, f"{name}_top", [D / 2, W / 2, T / 2], [-D / 2, 0, z0 + H - T / 2], rgba)
    _box(root, f"{name}_back", [T / 2, W / 2, H / 2], [-D + T / 2, 0, z0 + H / 2], rgba)
    for k, sign in enumerate((-1, 1)):
        _box(root, f"{name}_side{k}", [D / 2, T / 2, H / 2], [-D / 2, sign * (W / 2 - T / 2), z0 + H / 2], rgba)
    for i in range(1, dividers + 1):
        _box(root, f"{name}_div{i}", [D / 2, W / 2 - T, T / 2], [-D / 2, 0, z0 + H * i / (dividers + 1)], rgba)
    if z0 > 0.006:  # plinth, lifted 1 mm off the ground so it never touches static ground geoms
        _box(root, f"{name}_plinth", [D / 2 - 0.015, W / 2 - 0.015, (z0 - 0.001) / 2],
             [-D / 2 - 0.005, 0, 0.001 + (z0 - 0.001) / 2], [0.25, 0.22, 0.2, 1])


# ---------------------------------------------------------------------------------------- families
def _drawer_box(spec, root, pname: str, p: dict, W: float, z_lo: float, z_hi: float, handle: str,
                handle_z: float | None = None) -> Part:
    """A drawer whose overlay front spans z_lo..z_hi; its box is inside the carcass behind it."""
    d = float(p["drawer_depth"])
    body = root.add_body(name=pname, pos=[0, 0, 0])
    joint = f"{pname}_slide"
    travel = 0.8 * d
    _joint(body, joint, "slide", [1, 0, 0], travel, p)
    fh, fw = z_hi - z_lo - GAP, W - GAP
    zc = (z_lo + z_hi) / 2
    _box(body, f"{pname}_front", [TF / 2, fw / 2, fh / 2], [TF / 2, 0, zc], p.get("front_rgba", WOOD))
    iw, bh = W - 2 * T - 0.008, z_hi - z_lo - T - 0.02
    zb = z_lo + T / 2 + 0.004
    rgba = [0.8, 0.72, 0.58, 1]
    _box(body, f"{pname}_floor", [d / 2, iw / 2, 0.004], [-d / 2, 0, zb + 0.004], rgba)
    for k, sign in enumerate((-1, 1)):
        _box(body, f"{pname}_wall{k}", [d / 2, 0.004, bh / 2], [-d / 2, sign * (iw / 2 - 0.004), zb + bh / 2], rgba)
    _box(body, f"{pname}_rear", [0.004, iw / 2, bh / 2], [-d + 0.004, 0, zb + bh / 2], rgba)
    hz = zc + float(p.get("handle_dz", 0.0)) if handle_z is None else handle_z
    if handle == "flush":
        hz = z_hi - 0.008 - float(p.get("handle_len", 0.055)) / 2
    up = [0, 1.0, 0] if handle == "hbar" else [0, 0, 1.0]
    site = _add_handle(body, pname, handle, [TF, float(p.get("handle_dy", 0.0)), hz], [1, 0, 0], up, p)
    return Part(pname, "slide", joint, pname, site, handle, (0.0, travel), _lock(spec, joint))


def _build_drawer(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, D, H, z0 = p["width"], p["depth"], p["height"], p["plinth"]
    _carcass(root, name, W, D, H, z0, p.get("carcass_rgba", WOOD))
    hz = z0 + H - 0.024 if p["handle"] == "hbar" else None
    part = _drawer_box(spec, root, name, p, W, z0, z0 + H, p["handle"], hz)
    return [part], (-D - 0.01, TF + part.range[1] + 0.12, -W / 2 - 0.03, W / 2 + 0.03), part.name


def _build_drawer_stack(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, D, H, z0, n = p["width"], p["depth"], p["height"], p["plinth"], int(p["n_drawers"])
    _carcass(root, name, W, D, H, z0, p.get("carcass_rgba", WOOD), dividers=n - 1)
    parts = []
    for i in range(n):
        q = dict(p)
        if p.get("handle_layout") == "staggered":  # alternate left / right, so handles are not stacked
            q["handle_dy"] = (1 if i % 2 else -1) * W / 3
        parts.append(_drawer_box(spec, root, f"{name}{i}", q, W, z0 + H * i / n, z0 + H * (i + 1) / n, p["handle"]))
    travel = parts[0].range[1]
    return parts, (-D - 0.01, TF + travel + 0.12, -W / 2 - 0.03, W / 2 + 0.03), parts[int(p["target"])].name


def _build_hinged_door(spec, root, name: str, p: dict, W: float, H: float, z0: float, thick: float) -> Part:
    """Door covering the front (x in [0, thick]) with a vertical hinge at its left or right edge."""
    hs = -1.0 if p["hinge"] == "left" else 1.0  # hinge on the user's left (-y) or right (+y)
    door = root.add_body(name=name, pos=[0, hs * W / 2, z0 + H / 2])
    joint = f"{name}_hinge"
    _joint(door, joint, "hinge", [0, 0, hs], float(p.get("door_range", np.pi / 2)), p)
    _box(door, f"{name}_panel", [thick / 2, W / 2 - GAP, H / 2 - GAP], [thick / 2, -hs * W / 2, 0],
         p.get("front_rgba", WOOD))
    hy = -hs * (W - float(p["handle_offset"]))  # measured from the free edge
    handle = p["handle"]
    if handle == "lever":
        hz = H / 2 - float(p.get("lever_top", 0.022)) if "lever_top" in p else float(p.get("handle_dz", 0.0))
        site, lj, lb = _add_lever(door, name, [thick, hy, hz], [1, 0, 0], [0, hs, 0], p)
        return Part(name, "hinge", joint, name, site, handle, (0.0, float(p.get("door_range", np.pi / 2))),
                    _lock(spec, joint), lever_joint=lj, lever_body=lb,
                    latch_threshold=float(p.get("latch_threshold", 0.3)))
    site = _add_handle(door, name, handle, [thick, hy, float(p.get("handle_dz", 0.0))], [1, 0, 0], [0, 0, 1], p)
    return Part(name, "hinge", joint, name, site, handle, (0.0, float(p.get("door_range", np.pi / 2))),
                _lock(spec, joint))


def _build_cabinet_door(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, D, H, z0 = p["width"], p["depth"], p["height"], p["plinth"]
    _carcass(root, name + "_body", W, D, H, z0, p.get("carcass_rgba", WOOD))
    part = _build_hinged_door(spec, root, name, p, W, H, z0, TF)
    return [part], (-D - 0.01, W + 0.08, -W / 2 - W * 0.4, W / 2 + W * 0.4), part.name


def _build_room_door(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, H = p["width"], p["height"]
    wall, rgba = 0.12, p.get("carcass_rgba", [0.85, 0.84, 0.8, 1])
    side = 0.35
    for k, sign in enumerate((-1, 1)):
        _box(root, f"{name}_wall{k}", [wall / 2, side / 2, (H + 0.3) / 2],
             [-wall / 2, sign * (W / 2 + side / 2), (H + 0.3) / 2], rgba)
    _box(root, f"{name}_header", [wall / 2, W / 2, 0.15], [-wall / 2, 0, H + 0.15], rgba)
    part = _build_hinged_door(spec, root, name, p, W, H - 0.01, 0.01, 0.04)
    return [part], (-wall - 0.01, W + 0.15, -W / 2 - side, W / 2 + side), part.name


def _build_sliding_door(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, D, H, z0 = p["width"], p["depth"], p["height"], p["plinth"]
    _carcass(root, name + "_body", W, D, H, z0, p.get("carcass_rgba", WOOD))
    s = 1.0 if p["slide_dir"] == "right" else -1.0  # the panel slides towards +y (user's right) or -y
    ph = H - 2 * GAP - 0.004
    _box(root, f"{name}_fixed", [TF / 2, W / 4, ph / 2], [TF / 2, s * W / 4, z0 + GAP + ph / 2], p.get("front_rgba", WOOD))
    body = root.add_body(name=name, pos=[TF + 0.003, 0, z0 + GAP])
    joint = f"{name}_slide"
    travel = 0.8 * W / 2
    _joint(body, joint, "slide", [0, s, 0], travel, p)
    _box(body, f"{name}_panel", [TF / 2, W / 4 + 0.005, ph / 2], [TF / 2, -s * (W / 4 - 0.005), ph / 2],
         p.get("front_rgba", WOOD))
    hy = -s * (W / 2 - float(p["handle_offset"]))
    if p["handle"] == "flush":
        hz = ph - 0.006 - float(p.get("handle_len", 0.055)) / 2
    else:
        hz = ph / 2 + float(p.get("handle_dz", 0.0))
    site = _add_handle(body, name, p["handle"], [TF, hy, hz], [1, 0, 0], [0, 0, 1], p)
    part = Part(name, "slide", joint, name, site, p["handle"], (0.0, travel), _lock(spec, joint))
    return [part], (-D - 0.01, 0.14, -W / 2 - 0.12, W / 2 + 0.12), part.name


def _build_lid(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, D, H, z0 = p["width"], p["depth"], p["height"], p["plinth"]
    _carcass(root, name + "_box", W, D, H, z0, p.get("carcass_rgba", WOOD), open_top=True)
    _box(root, f"{name}_front_wall", [T / 2, W / 2, H / 2], [-T / 2, 0, z0 + H / 2], p.get("carcass_rgba", WOOD))
    lid = root.add_body(name=name, pos=[-D, 0, z0 + H])
    joint = f"{name}_hinge"
    rng_hi = float(p.get("lid_range", np.pi / 2))
    _joint(lid, joint, "hinge", [0, -1, 0], rng_hi, p)  # rotating about -y lifts the front edge
    _box(lid, f"{name}_panel", [(D + 0.01) / 2, W / 2 + 0.005, T / 2], [(D + 0.01) / 2, 0, T / 2],
         p.get("front_rgba", WOOD))
    hx = D - float(p["handle_offset"])
    site = _add_handle(lid, name, p["handle"], [hx, float(p.get("handle_dy", 0.0)), T], [0, 0, 1], [1, 0, 0], p)
    part = Part(name, "hinge", joint, name, site, p["handle"], (0.0, rng_hi), _lock(spec, joint))
    return [part], (-D - 0.03, 0.08, -W / 2 - 0.03, W / 2 + 0.03), part.name


def _build_flap(spec, root, name: str, p: dict) -> tuple[list[Part], tuple, str]:
    W, D, H, z0 = p["width"], p["depth"], p["height"], p["plinth"]
    _carcass(root, name + "_body", W, D, H, z0, p.get("carcass_rgba", WOOD))
    flap = root.add_body(name=name, pos=[TF, 0, z0 + GAP])  # hinge on the front face: the panel rests closed
    joint = f"{name}_hinge"
    rng_hi = float(p.get("flap_range", np.pi / 2))
    _joint(flap, joint, "hinge", [0, 1, 0], rng_hi, p)  # rotating about +y brings the top edge out
    fh = H - 2 * GAP
    _box(flap, f"{name}_panel", [TF / 2, W / 2 - GAP, fh / 2], [-TF / 2, 0, fh / 2], p.get("front_rgba", WOOD))
    hz = fh - float(p["handle_offset"])
    up = [0, 1.0, 0] if p["handle"] == "hbar" else [0, 0, 1.0]
    site = _add_handle(flap, name, p["handle"], [0, float(p.get("handle_dy", 0.0)), hz], [1, 0, 0], up, p)
    part = Part(name, "hinge", joint, name, site, p["handle"], (0.0, rng_hi), _lock(spec, joint))
    return [part], (-D - 0.01, H + 0.08, -W / 2 - 0.03, W / 2 + 0.03), part.name


_BUILDERS = {
    "drawer": _build_drawer,
    "drawer_stack": _build_drawer_stack,
    "cabinet_door": _build_cabinet_door,
    "room_door": _build_room_door,
    "sliding_door": _build_sliding_door,
    "lid": _build_lid,
    "flap": _build_flap,
}
FAMILIES = tuple(_BUILDERS)


def add_fixture(spec: mujoco.MjSpec, parent, fx: FixtureSpec, name: str, pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0)) -> Fixture:
    """Builds fixture `fx` as a static body `<name>_base` under `parent` (any MjsBody, e.g. the worldbody
    or a mocap body) at pos/quat. Element names are prefixed with `name`, so several fixtures coexist;
    a single-part fixture's part is called `name` itself."""
    root = parent.add_body(name=f"{name}_base", pos=list(pos), quat=list(quat))
    before = {b.name for b in spec.bodies}
    parts, keepout, target = _BUILDERS[fx.family](spec, root, name, dict(fx.params))
    bodies = [root.name] + [b.name for b in spec.bodies if b.name not in before]
    # A static root is welded to the world, so MuJoCo's parent-child filter does not apply: moving
    # parts would collide with the carcass they slide in. Exclude those pairs explicitly.
    for part in parts:
        for b in filter(None, (part.body, part.lever_body)):
            spec.add_exclude(bodyname1=root.name, bodyname2=b)
        if part.lever_body:
            spec.add_exclude(bodyname1=part.body, bodyname2=part.lever_body)
    return Fixture(name, fx, root.name, parts, keepout, target, bodies)


# ---------------------------------------------------------------------------------------- runtime
class FixtureRuntime:
    """Per-episode hidden causes, the lever latch, and observation of a compiled model's fixtures."""

    def __init__(self, model: mujoco.MjModel, fixtures: list[Fixture]):
        self.fixtures = fixtures
        self.parts: dict[str, Part] = {p.name: p for f in fixtures for p in f.parts}
        self.fixture_of = {p.name: f.name for f in fixtures for p in f.parts}
        self.ids: dict[str, dict] = {}
        for name, p in self.parts.items():
            j = model.joint(p.joint)
            ids = {"jnt": j.id, "qpos": int(j.qposadr[0]), "dof": int(j.dofadr[0]), "eq": model.equality(p.lock_eq).id,
                   "site": model.site(p.handle_site).id, "body": model.body(p.body).id}
            if p.lever_joint:
                lj = model.joint(p.lever_joint)
                ids.update(lever_jnt=lj.id, lever_qpos=int(lj.qposadr[0]), lever_body=model.body(p.lever_body).id)
            self.ids[name] = ids
        jid = [self.ids[n]["jnt"] for n in self.parts]
        did = [self.ids[n]["dof"] for n in self.parts]
        self._nominal = {"range": model.jnt_range[jid].copy(), "frictionloss": model.dof_frictionloss[did].copy(),
                         "damping": model.dof_damping[did].copy()}
        self.causes: dict[str, str] = {n: "none" for n in self.parts}
        self._radius: dict[str, float] = {}

    def body_names(self, model: mujoco.MjModel) -> dict[int, str]:
        """Body id -> tracking name: moving parts (and their levers) by part name, the rest by fixture."""
        out = {}
        for f in self.fixtures:
            for b in f.bodies:
                out[model.body(b).id] = f.name
        for name, ids in self.ids.items():
            out[ids["body"]] = name
            if "lever_body" in ids:
                out[ids["lever_body"]] = name
        return out

    def sample_causes(self, rng: np.random.Generator, probs: dict[str, float]) -> dict[str, str]:
        """One uniform draw per part (stable random streams); `latched` only applies to lever handles."""
        w = np.array([max(float(probs.get(c, 0.0)), 0.0) for c in CAUSES])
        cdf = np.cumsum(w) / w.sum() if w.sum() > 0 else np.r_[1.0, np.ones(len(CAUSES) - 1)]
        out = {}
        for name, p in self.parts.items():
            c = CAUSES[min(int(np.searchsorted(cdf, rng.uniform(), side="right")), len(CAUSES) - 1)]
            out[name] = "none" if (c == "latched" and not p.lever_joint) else c
        return out

    def apply_causes(self, model: mujoco.MjModel, data: mujoco.MjData, causes: dict[str, str]) -> None:
        """Restores the nominal joints, then applies the hidden causes. Call after mj_resetData."""
        for i, name in enumerate(self.parts):
            ids, p = self.ids[name], self.parts[name]
            model.jnt_range[ids["jnt"]] = self._nominal["range"][i]
            model.dof_frictionloss[ids["dof"]] = self._nominal["frictionloss"][i]
            model.dof_damping[ids["dof"]] = self._nominal["damping"][i]
            cause = causes.get(name, "none")
            if cause not in CAUSES:
                raise ValueError(f"unknown hidden cause {cause!r}")
            if cause == "latched" and not p.lever_joint:
                raise ValueError(f"{name}: only lever handles can be latched")
            if cause == "stuck":
                r = 1.0 if p.kind == "slide" else self.handle_radius(model, data, name)
                model.dof_frictionloss[ids["dof"]] = max(10 * self._nominal["frictionloss"][i], STUCK_FORCE * r)
                model.dof_damping[ids["dof"]] = STUCK_DAMPING * self._nominal["damping"][i]
            elif cause == "blocked":
                model.jnt_range[ids["jnt"], 1] = BLOCKED_FRAC * self._nominal["range"][i, 1]
        self.causes = {n: causes.get(n, "none") for n in self.parts}
        self.update(data)

    def handle_radius(self, model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
        """Distance of the handle from the hinge axis in the closed pose (for stuck torques)."""
        if name not in self._radius:
            scratch = mujoco.MjData(model)
            mujoco.mj_kinematics(model, scratch)
            ids = self.ids[name]
            rel = scratch.site_xpos[ids["site"]] - scratch.xanchor[ids["jnt"]]
            ax = scratch.xaxis[ids["jnt"]]
            self._radius[name] = float(np.linalg.norm(rel - np.dot(rel, ax) * ax))
        return self._radius[name]

    def latched(self, data: mujoco.MjData, name: str) -> bool:
        p, ids = self.parts[name], self.ids[name]
        if self.causes.get(name) != "latched" or not p.lever_joint:
            return False
        lever_turned = data.qpos[ids["lever_qpos"]] >= p.latch_threshold
        near_closed = data.qpos[ids["qpos"]] < LATCH_CATCH[p.kind]
        return near_closed and not lever_turned

    def update(self, data: mujoco.MjData) -> None:
        """Engages the lock constraints: locked parts always, latched ones while the latch holds."""
        for name, ids in self.ids.items():
            data.eq_active[ids["eq"]] = self.causes.get(name) == "locked" or self.latched(data, name)

    def opening(self, data: mujoco.MjData, name: str) -> float:
        i = list(self.parts).index(name)
        lo, hi = self._nominal["range"][i]
        return float(np.clip((data.qpos[self.ids[name]["qpos"]] - lo) / (hi - lo), 0.0, 1.0))

    def observe(self, data: mujoco.MjData) -> dict[str, dict]:
        """What a body could see or feel of each part; hidden causes are deliberately absent."""
        out = {}
        for name, p in self.parts.items():
            ids = self.ids[name]
            pos = data.site_xpos[ids["site"]].copy()
            mat = data.site_xmat[ids["site"]].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat.flatten())
            axis, anchor = data.xaxis[ids["jnt"]].copy(), data.xanchor[ids["jnt"]].copy()
            entry = {
                "fixture": self.fixture_of[name], "kind": p.kind, "handle": p.handle,
                "q": float(data.qpos[ids["qpos"]]), "opening": self.opening(data, name),
                "handle_pos": pos, "handle_quat": quat, "normal": mat[:, 0].copy(), "handle_axis": mat[:, 2].copy(),
                "joint_axis": axis, "joint_anchor": anchor, "pull_dir": _motion_dir(p.kind, axis, anchor, pos),
            }
            if p.lever_joint:
                lj = ids["lever_jnt"]
                lq = float(data.qpos[ids["lever_qpos"]])
                entry["lever"] = {"q": lq, "turn": lq / LEVER_RANGE,
                                  "pull_dir": _motion_dir("hinge", data.xaxis[lj], data.xanchor[lj], pos)}
            out[name] = entry
        return out

    def ground_truth(self) -> list[dict]:
        """Per fixture: type, family, parameters and each part's hidden cause (for logs and scoring)."""
        return [{"fixture": f.name, "type": f.spec.type, "family": f.spec.family, "target": f.target,
                 "params": _jsonable(f.spec.params),
                 "parts": [{"part": p.name, "kind": p.kind, "handle": p.handle, "cause": self.causes.get(p.name, "none"),
                            "locked": self.causes.get(p.name) == "locked", "has_lever": bool(p.lever_joint)}
                           for p in f.parts]}
                for f in self.fixtures]


def _motion_dir(kind: str, axis: np.ndarray, anchor: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Direction a point on the part moves for increasing joint position."""
    v = axis.copy() if kind == "slide" else np.cross(axis, point - anchor)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _jsonable(params: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()}


# ---------------------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class FixtureType:
    name: str
    family: str
    split: str          # train | test (held out)
    fixed: dict         # discrete choices that define the type (handle shape, hinge side, ...)
    tabletop: bool = True


def _u(rng, lo, hi) -> float:
    return float(rng.uniform(lo, hi))


def _feel(rng, scale: dict) -> dict:
    """Joint feel: smooth or stiff damping/friction, and optional spring return (self-closing)."""
    stiff = rng.uniform() < 0.3
    spring = rng.uniform() < scale.get("spring_prob", 0.25)
    d = _u(rng, *scale["damping"]) * (2.5 if stiff else 1.0)
    f = _u(rng, *scale["frictionloss"]) * (3.0 if stiff else 1.0)
    return {"feel": "stiff" if stiff else "smooth", "spring": bool(spring), "damping": d, "frictionloss": f,
            "stiffness": float(scale["spring"]) if spring else 0.0}


_SLIDE_FEEL = {"damping": (2.0, 4.0), "frictionloss": (0.5, 1.5), "spring": 8.0, "spring_prob": 0.2}
_DOOR_FEEL = {"damping": (0.02, 0.08), "frictionloss": (0.03, 0.1), "spring": 0.35}
_ROOM_FEEL = {"damping": (2.0, 4.0), "frictionloss": (1.0, 2.0), "spring": 8.0}
_FLAP_FEEL = {"damping": (0.2, 0.4), "frictionloss": (0.05, 0.1), "spring": 0.6}  # soft-drop: falls open past ~30 deg


def _handle_params(rng, handle: str, big: bool = False) -> dict:
    k = 1.5 if big else 1.0
    p = {"handle": handle, "standoff": _u(rng, 0.048, 0.055) * (1.2 if big else 1.0)}
    if handle == "bar":
        p.update(handle_len=_u(rng, 0.045, 0.055) * (4 if big else 1), handle_w=_u(rng, 0.02, 0.024) * k)
    elif handle == "knob":
        p.update(knob_r=_u(rng, 0.017, 0.02) * k)
    elif handle == "hbar":
        p.update(handle_len=_u(rng, 0.08, 0.1))
    elif handle == "lever":
        p.update(lever_len=_u(rng, 0.075, 0.085) * k, lever_spring=0.4 * (8 if big else 1))
    elif handle == "flush":
        p.update(handle_len=_u(rng, 0.05, 0.06))
    return p


def _sample_params(rng: np.random.Generator, t: FixtureType) -> dict:
    fam, fx = t.family, t.fixed
    colour = {"front_rgba": [_u(rng, 0.5, 0.9), _u(rng, 0.35, 0.75), _u(rng, 0.2, 0.6), 1.0],
              "carcass_rgba": [_u(rng, 0.45, 0.7), _u(rng, 0.35, 0.5), _u(rng, 0.25, 0.35), 1.0]}
    if fam == "drawer":
        D = _u(rng, 0.2, 0.24)
        p = {"width": _u(rng, 0.16, 0.22), "height": _u(rng, 0.1, 0.13), "depth": D, "plinth": _u(rng, 0.02, 0.04),
             "drawer_depth": _u(rng, 0.14, D - 0.04), "handle_dy": _u(rng, -0.02, 0.02),
             "handle_dz": _u(rng, -0.01, 0.01), **_feel(rng, _SLIDE_FEEL)}
    elif fam == "drawer_stack":
        n = int(fx["n_drawers"])
        D = _u(rng, 0.2, 0.22)
        wide = fx.get("handle_layout") == "staggered"
        p = {"width": _u(rng, 0.26, 0.3) if wide else _u(rng, 0.18, 0.22), "height": n * _u(rng, 0.09, 0.1),
             "depth": D, "plinth": _u(rng, 0.015, 0.03),
             "drawer_depth": _u(rng, 0.14, D - 0.04), "target": int(rng.integers(n)), "handle_dz": _u(rng, -0.005, 0.005),
             **_feel(rng, _SLIDE_FEEL)}
    elif fam == "cabinet_door":
        p = {"width": _u(rng, 0.16, 0.2), "height": _u(rng, 0.17, 0.2), "depth": _u(rng, 0.18, 0.22),
             "plinth": _u(rng, 0.02, 0.03), "handle_offset": _u(rng, 0.025, 0.035), "handle_dz": _u(rng, -0.02, 0.03),
             "door_range": np.pi / 2, **_feel(rng, _DOOR_FEEL)}
        if fx["handle"] == "lever":
            p.update(lever_top=0.022, handle_offset=_u(rng, 0.028, 0.035), latch_threshold=0.3)
    elif fam == "room_door":
        p = {"width": _u(rng, 0.8, 0.95), "height": _u(rng, 2.0, 2.1), "handle_offset": _u(rng, 0.06, 0.08),
             "door_range": 1.9, "hinge": str(rng.choice(["left", "right"])), "lever_dir": "down",
             "latch_threshold": 0.3, **_feel(rng, _ROOM_FEEL)}
        p["handle_dz"] = _u(rng, 0.95, 1.05) - (p["height"] - 0.01) / 2 - 0.01  # handle ~1 m above the floor
    elif fam == "sliding_door":
        p = {"width": _u(rng, 0.3, 0.36), "height": _u(rng, 0.15, 0.19), "depth": _u(rng, 0.16, 0.2),
             "plinth": _u(rng, 0.02, 0.03), "handle_offset": _u(rng, 0.035, 0.045), "handle_dz": _u(rng, -0.01, 0.01),
             **_feel(rng, _SLIDE_FEEL)}
    elif fam == "lid":
        spring = rng.uniform() < 0.25  # "self-closing": a loose hinge lets gravity shut it
        p = {"width": _u(rng, 0.2, 0.26), "height": _u(rng, 0.08, 0.12), "depth": _u(rng, 0.16, 0.2),
             "plinth": 0.002, "handle_offset": _u(rng, 0.025, 0.035), "handle_dy": _u(rng, -0.03, 0.03),
             "lid_range": np.pi / 2, "feel": "smooth", "spring": bool(spring), "damping": _u(rng, 0.03, 0.08),
             "frictionloss": 0.05 if spring else _u(rng, 0.4, 0.5), "stiffness": 0.0, "standoff": 0.035}
    elif fam == "flap":
        p = {"width": _u(rng, 0.18, 0.24), "height": _u(rng, 0.12, 0.15), "depth": _u(rng, 0.16, 0.2),
             "plinth": _u(rng, 0.08, 0.1), "handle_offset": _u(rng, 0.03, 0.04), "handle_dy": _u(rng, -0.02, 0.02),
             "flap_range": np.pi / 2, **_feel(rng, _FLAP_FEEL)}
    else:
        raise ValueError(fam)
    p.update(colour)
    p.update(_handle_params(rng, fx["handle"], big=fam == "room_door"))
    if fam == "lid":
        p["standoff"] = 0.035
    p.update({k: v for k, v in fx.items() if k != "handle"})
    return p


def _types() -> dict[str, FixtureType]:
    T_ = FixtureType
    ts = [
        T_("drawer_bar", "drawer", "train", {"handle": "bar"}),
        T_("drawer_knob", "drawer", "train", {"handle": "knob"}),
        T_("drawer_hbar", "drawer", "test", {"handle": "hbar"}),
        T_("drawer_flush", "drawer", "test", {"handle": "flush"}),
        T_("stack2_bar", "drawer_stack", "train", {"handle": "bar", "n_drawers": 2, "handle_layout": "staggered"}),
        T_("stack3_bar", "drawer_stack", "train", {"handle": "bar", "n_drawers": 3, "handle_layout": "centred"}),
        T_("stack2_knob", "drawer_stack", "test", {"handle": "knob", "n_drawers": 2, "handle_layout": "staggered"}),
        T_("door_bar_left", "cabinet_door", "train", {"handle": "bar", "hinge": "left"}),
        T_("door_bar_right", "cabinet_door", "train", {"handle": "bar", "hinge": "right"}),
        T_("door_knob_right", "cabinet_door", "train", {"handle": "knob", "hinge": "right"}),
        T_("door_knob_left", "cabinet_door", "test", {"handle": "knob", "hinge": "left"}),
        T_("door_lever_up_left", "cabinet_door", "train", {"handle": "lever", "hinge": "left", "lever_dir": "up"}),
        T_("door_lever_down_right", "cabinet_door", "test", {"handle": "lever", "hinge": "right", "lever_dir": "down"}),
        T_("room_door_lever", "room_door", "train", {"handle": "lever"}, tabletop=False),
        T_("room_door_knob", "room_door", "train", {"handle": "knob"}, tabletop=False),
        T_("room_door_bar", "room_door", "test", {"handle": "bar"}, tabletop=False),
        T_("sliding_bar_right", "sliding_door", "train", {"handle": "bar", "slide_dir": "right"}),
        T_("sliding_bar_left", "sliding_door", "train", {"handle": "bar", "slide_dir": "left"}),
        T_("sliding_flush", "sliding_door", "test", {"handle": "flush", "slide_dir": "right"}),
        T_("lid_knob", "lid", "train", {"handle": "knob"}),
        T_("lid_tab", "lid", "test", {"handle": "tab"}),
        T_("flap_knob", "flap", "train", {"handle": "knob"}),
        T_("flap_bar", "flap", "test", {"handle": "bar"}),
    ]
    return {t.name: t for t in ts}


FIXTURE_TYPES: dict[str, FixtureType] = _types()


def fixture_types(family: str | None = None, split: str | None = None, tabletop: bool | None = None) -> list[str]:
    return [n for n, t in FIXTURE_TYPES.items()
            if (family is None or t.family == family) and (split is None or t.split == split)
            and (tabletop is None or t.tabletop == tabletop)]


def make_fixture(rng: np.random.Generator, type_name: str, **overrides) -> FixtureSpec:
    """A concrete fixture of a named type; `overrides` fix individual parameters."""
    t = FIXTURE_TYPES[type_name]
    return FixtureSpec(t.name, t.family, {**_sample_params(rng, t), **overrides})


def sample_fixture(rng: np.random.Generator, family: str | None = None, split: str = "train",
                   tabletop: bool | None = None) -> FixtureSpec:
    """Draws a type from the split (optionally one family / tabletop-sized only), then its parameters."""
    names = fixture_types(family, split, tabletop)
    if not names:
        raise ValueError(f"no fixture types for family={family!r}, split={split!r}, tabletop={tabletop!r}")
    return make_fixture(rng, names[int(rng.integers(len(names)))])
