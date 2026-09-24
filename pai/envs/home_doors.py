"""Home doors with physical locks and keys, for any embodiment.

A wall with a full-size door, a side cabinet (drawer + top) and a wall shelf beside it, a table in the
room behind, and free rigid keys. Everything that opens or locks is a MuJoCo joint; constraints are
switched only by the positions of those joints (like the cams and springs of a real lock), never by a
flag the agent could set. Nothing about the lock state is in the observations: the agent has to find out
by acting (turning, pulling, finding and trying keys).

Mechanisms (door frame: +x towards the agent, +y the agent's right, z up; the door sits in the middle of
a 12 cm wall):
- latch + handle. The handle is a spring-returned joint: a knob (hinge about the door normal; turns both
  ways or one way), a lever (hinge about the normal, pressed down) or a push bar (slide, pressed towards
  the door). While the door is within LATCH_CATCH of closed and the handle is not turned/pressed past its
  threshold, a joint-equality lock holds the door hinge: the latch bolt in the strike. Once the door is
  ajar the latch no longer holds; a self-closing door re-latches when it swings shut, as a bevelled latch
  does.
- deadbolts are real geometry: a bolt body (slide joint) in the door's latch edge; extended, it sits in a
  pocket cut into the jamb and the pocket walls stop the door. The bolt is driven by a tailpiece joint
  through a joint equality (bolt = c * tailpiece angle).
  * thumb-turn deadbolt: the tailpiece is the thumb-turn itself, a wing on the agent's face (visible).
  * key deadbolt: the tailpiece is internal. A lock cylinder on the agent's face has a plug (hinge about
    the keyhole axis) and, inside it, a key carrier (slide along the keyhole axis, held by friction). The
    plug is held at home by an equality unless the MATCHING key is fully inserted; then the plug drives
    the tailpiece instead (turning the key a quarter turn retracts / extends the bolt). A key can only be
    withdrawn with the plug at home.
- keys are free bodies (a shaft and an L- or T-shaped bow). Insertion is detected from geometry each
  control step: the key tip within KEY_CAPTURE_DIST of the keyhole mouth, the shaft within
  KEY_CAPTURE_ANGLE of the keyhole axis and the blade within KEY_CAPTURE_ROLL of the slot. Then a weld
  joins key and carrier (the carrier slides, so the key goes in as far as it is pushed) and the shaft's
  collisions are switched off (it is inside the plug). Pulled back to the mouth with the plug at home, the
  weld lets go. Every key has a bitting id; a wrong key inserts but the plug does not turn.
- door closer: joint stiffness (spring to closed) and damping on the hinge. Push doors open away from the
  agent, pull doors towards it.

Keys rest on small cradles: on the side cabinet top ("table"), on the shelf ("shelf"), inside the drawer
("drawer": open it first) or on the table in the other room ("other_room": unreachable, a no-solution
case when the door is key-locked). Placement, lock state and which key matches are drawn per reset and
are ground truth only (HomeDoorRuntime.info and the events).

Registry: HOME_DOOR_TYPES (named types with train/test splits), make_home_door / sample_home_door. The
DoorSceneEnv adder is home_door_fixture(type_name).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from pai.envs.door_scene import HandleInfo, _yaw_quat
from pai.envs.fixtures import FixtureSpec, _add_lever, _box, _frame_quat

WALL = 0.12              # wall thickness (m)
DOOR_T = 0.04            # door leaf thickness (m)
DOOR_X0, DOOR_X1 = -0.08, -0.04   # door leaf x range: centred in the wall
GAP = 0.003
LATCH_CATCH = 0.03       # rad: the latch holds only this close to closed
BOLT_THROW = 0.028       # m, bolt travel into the jamb pocket
POCKET_DEPTH = 0.035
KEY_DEPTH = 0.03         # full insertion depth (m)
KEY_INSERTED = 0.026     # carrier position counted as fully inserted (and 0.018 to leave that state)
KEY_CAPTURE_DIST = 0.009
KEY_CAPTURE_ANGLE = np.deg2rad(15)
KEY_CAPTURE_ROLL = np.deg2rad(20)
KEY_TURNED = 1.2         # plug angle (rad) counted as a completed turn
THUMB_RANGE = np.pi / 2
DOOR_OPEN_FRAC = 0.35    # door opening fraction counted as open (0.66 rad of 1.9)
DEBOUNCE = 3
LOCK_STATES = ("unlocked", "deadbolt", "key", "both")
PLACEMENTS = ("table", "shelf", "drawer", "other_room")

WALL_RGBA = [0.86, 0.85, 0.8, 1]
FRAME_RGBA = [0.95, 0.95, 0.93, 1]
DOOR_RGBA = [0.55, 0.36, 0.22, 1]
BRASS = [0.82, 0.68, 0.32, 1]
STEEL = [0.7, 0.72, 0.75, 1]
DARK = [0.08, 0.08, 0.08, 1]
CAB_RGBA = [0.52, 0.42, 0.33, 1]
_g = mujoco.mjtGeom


# ---------------------------------------------------------------------------------------- records
@dataclass
class HomeDoor:
    """Names of everything in the compiled model that the runtime and controllers need."""
    name: str
    spec: FixtureSpec
    door_body: str
    door_joint: str
    latch_eq: str
    handle: str                     # knob_round | knob_ball | lever | push_bar
    handle_joint: str
    handle_site: str
    handle_mode: str                # both | one (knob turn directions) | press (lever / bar)
    handle_threshold: float
    handle_range: tuple
    bolts: dict = field(default_factory=dict)      # lock name -> {"joint", "tail"}
    thumb: dict | None = None       # {"joint", "site"}
    key_lock: dict | None = None    # {"plug", "carrier", "carrier_body", "tail", "site", "lock_eq", "drive_eq",
                                    #  "hold_eq", "bitting", "unlock_sign"}
    keys: list = field(default_factory=list)       # [{"body", "shaft_geoms", "weld_eq", "bitting", "grip_site"}]
    cradles: dict = field(default_factory=dict)    # placement -> [(parent body, local pos, local quat)]
    drawer: dict | None = None      # {"joint", "site", "range"}
    bodies: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------- geometry
def _wall(root, p: dict) -> None:
    """Wall with the door opening, and pockets in the latch-side jamb at the bolt heights."""
    W, H, side = p["width"], p["height"], 1.3
    ls = p["latch_sign"]
    xc = -WALL / 2
    hs = -ls
    # hinge-side panel and header
    _box(root, "wall_hinge", [WALL / 2, side / 2, (H + 0.4) / 2], [xc, hs * (W / 2 + side / 2), (H + 0.4) / 2], WALL_RGBA)
    _box(root, "wall_header", [WALL / 2, W / 2, 0.2], [xc, 0, H + 0.2], WALL_RGBA)
    # latch-side panel: full-thickness slabs between pockets; at a pocket, front/back slabs and an end slab
    pockets = sorted(p["bolt_heights"])
    y_in, y_pk, y_out = W / 2, W / 2 + POCKET_DEPTH, W / 2 + side
    z = 0.0
    k = 0
    for zb in pockets + [None]:
        z_top = H + 0.4 if zb is None else zb - 0.02
        if z_top > z:
            _box(root, f"wall_latch{k}", [WALL / 2, (y_out - y_in) / 2, (z_top - z) / 2],
                 [xc, ls * (y_in + y_out) / 2, (z + z_top) / 2], WALL_RGBA)
            k += 1
        if zb is None:
            break
        hz = 0.02
        front = (0 - (DOOR_X1 + 0.006)) / 2
        back = ((DOOR_X0 - 0.006) + WALL) / 2
        _box(root, f"pocket{k}_front", [front, (y_pk - y_in) / 2, hz], [-front, ls * (y_in + y_pk) / 2, zb], WALL_RGBA)
        _box(root, f"pocket{k}_back", [back, (y_pk - y_in) / 2, hz], [-WALL + back, ls * (y_in + y_pk) / 2, zb], WALL_RGBA)
        _box(root, f"pocket{k}_end", [WALL / 2, (y_out - y_pk) / 2, hz], [xc, ls * (y_pk + y_out) / 2, zb], WALL_RGBA)
        _box(root, f"strike{k}", [0.001, 0.012, 0.03], [0.001, ls * (y_in + 0.012), zb], STEEL,
             contype=0, conaffinity=0)
        z = zb + hz
    # frame trim (visual only)
    for s in (-1, 1):
        _box(root, f"trim{'lr'[s > 0]}", [0.006, 0.03, H / 2], [0.006, s * (W / 2 + 0.03), H / 2], FRAME_RGBA,
             contype=0, conaffinity=0)
    _box(root, "trim_top", [0.006, W / 2 + 0.06, 0.03], [0.006, 0, H + 0.03], FRAME_RGBA, contype=0, conaffinity=0)


def _handle(spec, door, name: str, p: dict, face_pos: np.ndarray, n: np.ndarray, to_hinge: np.ndarray):
    """The latch handle on the door face. Returns (joint, site, mode, threshold, range)."""
    kind = p["handle"]
    if kind == "push_bar":
        length = 0.6
        c = face_pos + to_hinge * length / 2    # from the hardware line towards the hinge (clear of the jamb)
        bar = door.add_body(name=f"{name}_pushbar", pos=(c + n * 0.07).tolist())
        j = f"{name}_pushbar_slide"
        bar.add_joint(name=j, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=(-n).tolist(), range=[0, 0.025],
                      limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=[250.0, 0, 0], damping=[5.0, 0, 0],
                      armature=0.02)
        q = _frame_quat(n, to_hinge)
        bar.add_geom(name=f"{name}_pushbar_bar", type=_g.mjGEOM_CAPSULE, size=[0.018, length / 2, 0],
                     quat=_quat_z_to(to_hinge), rgba=STEEL, density=1500)
        bar.add_site(name=f"{name}_handle", quat=q)
        for s in (-1, 1):
            _box(door, f"{name}_pushbar_mount{'ab'[s > 0]}", [0.03, 0.02, 0.03], c + n * 0.03 + s * to_hinge * (length / 2 - 0.02),
                 STEEL, _frame_quat(n, [0, 0, 1]), 2000)
        return j, f"{name}_handle", "press", 0.012, (0.0, 0.025)
    if kind == "lever":
        lp = {"standoff": 0.065, "lever_len": 0.12, "lever_dir": "down", "lever_spring": 1.5, "lever_grasp": 0.6}
        site, joint, _ = _add_lever(door, name, face_pos, n, to_hinge, lp)
        return joint, site, "press", 0.3, (0.0, 0.7)
    # knobs: a neck and a head on a hinge about the normal, spring-returned
    knob = door.add_body(name=f"{name}_knob", pos=face_pos.tolist())
    j = f"{name}_knob_hinge"
    both = p["knob_dir"] == "both"
    rng = [-1.2, 1.2] if both else [0.0, 1.2]
    knob.add_joint(name=j, type=mujoco.mjtJoint.mjJNT_HINGE, axis=n.tolist(), range=rng,
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=[0.4, 0, 0], damping=[0.02, 0, 0],
                   armature=0.001)
    door.add_geom(name=f"{name}_rose", type=_g.mjGEOM_CYLINDER, size=[0.032, 0.004, 0],
                  fromto=[*face_pos, *(face_pos + n * 0.008)], rgba=BRASS, density=3000)
    knob.add_geom(name=f"{name}_neck", type=_g.mjGEOM_CAPSULE, size=[0.011, 0, 0],
                  fromto=[*(n * 0.004), *(n * 0.05)], rgba=BRASS, density=3000)
    if kind == "knob_ball":
        knob.add_geom(name=f"{name}_head", type=_g.mjGEOM_ELLIPSOID, size=[0.022, 0.032, 0.032],
                      pos=(n * 0.068).tolist(), quat=_frame_quat(n, [0, 0, 1]), rgba=BRASS, density=3000,
                      friction=[1.2, 0.02, 0.001])
    else:
        knob.add_geom(name=f"{name}_head", type=_g.mjGEOM_CYLINDER, size=[0.029, 0.017, 0],
                      fromto=[*(n * 0.05), *(n * 0.084)], rgba=BRASS, density=3000, friction=[1.2, 0.02, 0.001])
    knob.add_site(name=f"{name}_handle", pos=(n * 0.067).tolist(), quat=_frame_quat(n, [0, 0, 1]))
    return j, f"{name}_handle", "both" if both else "one", 0.5, tuple(rng)


def _quat_z_to(axis) -> list[float]:
    """Quaternion taking +z to `axis` (capsules and cylinders lie along their z)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, a)
    return q.tolist()


def _bolt(spec, door, name: str, edge_pos: np.ndarray, to_latch: np.ndarray, rgba) -> str:
    """Deadbolt body in the latch edge; slides towards the jamb pocket."""
    b = door.add_body(name=f"{name}_bolt", pos=(edge_pos - to_latch * 0.023).tolist())
    j = f"{name}_bolt_slide"
    b.add_joint(name=j, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=to_latch.tolist(), range=[0, BOLT_THROW],
                limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=[2.0, 0, 0], armature=0.01)
    b.add_geom(name=f"{name}_bolt_geom", type=_g.mjGEOM_BOX, size=[0.008, 0.02, 0.01],
               quat=_frame_quat([1, 0, 0], [0, 0, 1]), rgba=rgba, density=7800)
    return j


def _tail_eq(spec, bolt_joint: str, tail_joint: str) -> None:
    spec.add_equality(name=f"{bolt_joint}_drive", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                      name1=bolt_joint, name2=tail_joint, active=True,
                      data=[0.0, BOLT_THROW / THUMB_RANGE] + [0.0] * 9, solref=[0.01, 1.0])


def _thumb_turn(spec, door, name: str, face_pos: np.ndarray, n: np.ndarray, to_latch: np.ndarray, p: dict) -> dict:
    """Thumb-turn wing (a 7 cm bar) on a hinge about the normal; turned a quarter towards the latch edge =
    locked (the wing then points at the bolt)."""
    body = door.add_body(name=f"{name}_thumb", pos=face_pos.tolist())
    j = f"{name}_thumb_hinge"
    axis = np.cross([0, 0, 1.0], to_latch)   # turning about this axis moves the wing's top towards the latch edge
    body.add_joint(name=j, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis.tolist(), range=[0, THUMB_RANGE],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=[0.02, 0, 0], frictionloss=0.08,
                   armature=0.002)
    door.add_geom(name=f"{name}_thumb_rose", type=_g.mjGEOM_CYLINDER, size=[0.03, 0.004, 0],
                  fromto=[*face_pos, *(face_pos + n * 0.008)], rgba=STEEL, density=3000)
    body.add_geom(name=f"{name}_thumb_post", type=_g.mjGEOM_CYLINDER, size=[0.01, 0, 0],
                  fromto=[*(n * 0.004), *(n * 0.04)], rgba=STEEL, density=3000)
    body.add_geom(name=f"{name}_thumb_wing", type=_g.mjGEOM_CAPSULE, size=[0.012, 0.035, 0],
                  pos=(n * 0.05).tolist(), quat=_quat_z_to([0, 0, 1]), rgba=STEEL, density=2000,
                  friction=[1.2, 0.02, 0.001])
    body.add_site(name=f"{name}_thumb_site", pos=(n * 0.05).tolist(), quat=_frame_quat(n, [0, 0, 1]))
    return {"joint": j, "site": f"{name}_thumb_site"}


def _key_lock(spec, door, name: str, face_pos: np.ndarray, n: np.ndarray, p: dict) -> dict:
    """Lock cylinder: rim (door), plug (hinge about the keyhole axis), carrier (slide along it)."""
    door.add_geom(name=f"{name}_cyl_rim", type=_g.mjGEOM_CYLINDER, size=[0.022, 0.005, 0],
                  fromto=[*face_pos, *(face_pos + n * 0.01)], rgba=BRASS, density=3000)
    plug = door.add_body(name=f"{name}_plug", pos=(face_pos + n * 0.01).tolist(),
                         quat=_frame_quat(-n, np.cross(-n, [0, 0, 1.0])))
    # plug frame: x = into the door (keyhole axis), y = up (the slot is vertical at home), z = x cross y
    pj = f"{name}_plug_hinge"
    plug.add_joint(name=pj, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-1.7, 1.7],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=[0.02, 0, 0], armature=0.002)
    plug.add_geom(name=f"{name}_plug_face", type=_g.mjGEOM_CYLINDER, size=[0.015, 0.0012, 0],
                  fromto=[-0.0022, 0, 0, -0.0002, 0, 0], rgba=[0.9, 0.78, 0.4, 1], contype=0, conaffinity=0,
                  density=3000)
    plug.add_geom(name=f"{name}_keyhole", type=_g.mjGEOM_BOX, size=[0.0006, 0.008, 0.0025], pos=[-0.0024, 0, 0],
                  rgba=DARK, contype=0, conaffinity=0, density=100)
    carrier = plug.add_body(name=f"{name}_carrier", pos=[0, 0, 0])
    cj = f"{name}_carrier_slide"
    carrier.add_joint(name=cj, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[0, KEY_DEPTH],
                      limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=[3.0, 0, 0], frictionloss=0.3,
                      armature=0.02)
    carrier.add_geom(name=f"{name}_carrier_geom", type=_g.mjGEOM_SPHERE, size=[0.002, 0, 0], contype=0,
                     conaffinity=0, rgba=[0, 0, 0, 0], mass=0.01)
    carrier.add_site(name=f"{name}_mouth")
    tail = door.add_body(name=f"{name}_tail", pos=(face_pos - n * 0.02).tolist())
    tj = f"{name}_tail_hinge"
    tail.add_joint(name=tj, type=mujoco.mjtJoint.mjJNT_HINGE, axis=(-n).tolist(), range=[0, THUMB_RANGE],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, frictionloss=0.05, damping=[0.01, 0, 0], armature=0.002)
    tail.add_geom(name=f"{name}_tail_geom", type=_g.mjGEOM_SPHERE, size=[0.003, 0, 0], contype=0, conaffinity=0,
                  rgba=[0, 0, 0, 0], mass=0.01)
    lock_eq, drive_eq, hold_eq = f"{name}_plug_lock", f"{name}_plug_drive", f"{name}_carrier_hold"
    spec.add_equality(name=lock_eq, type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=pj,
                      active=True, data=[0.0] * 11, solref=[0.005, 1.0])
    spec.add_equality(name=drive_eq, type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=tj,
                      name2=pj, active=False, data=[0.0, -1.0] + [0.0] * 9, solref=[0.01, 1.0])
    spec.add_equality(name=hold_eq, type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=cj,
                      active=False, data=[KEY_DEPTH] + [0.0] * 10, solref=[0.004, 1.0])
    return {"plug": pj, "carrier": cj, "carrier_body": f"{name}_carrier", "tail": tj, "site": f"{name}_mouth",
            "lock_eq": lock_eq, "drive_eq": drive_eq, "hold_eq": hold_eq, "bitting": int(p["bitting"]),
            "plug_body": f"{name}_plug"}


def _key(spec, name: str, shape: str, p: dict, rgba) -> dict:
    """A free key. Body frame: origin at the tip, x = insertion direction, y = blade width (the slot
    direction), so an aligned key has the plug's frame. Bows: 'L' (a bar at the end of the shaft, along
    +y) or 'T' (a bar centred on the shaft); `grip` is where a hand should hold it."""
    L, bar = float(p["key_shaft"]), 0.09
    body = spec.worldbody.add_body(name=name, pos=[0, 0, -5.0])
    body.add_freejoint(name=f"{name}_free")
    shaft = f"{name}_shaft"
    body.add_geom(name=shaft, type=_g.mjGEOM_BOX, size=[L / 2, 0.006, 0.0025], pos=[-L / 2, 0, 0], rgba=STEEL,
                  density=7800, friction=[1.0, 0.02, 0.001])
    y0 = 0.0 if shape == "L" else -bar / 2
    # a square 24 mm bow (the bar size every hand's grasp is tuned for); flat faces keep it from rolling
    # in the hand, as a round bar would under the shaft's weight
    body.add_geom(name=f"{name}_bow", type=_g.mjGEOM_BOX, size=[0.012, bar / 2, 0.012],
                  pos=[-L - 0.012, y0 + bar / 2, 0], rgba=rgba, density=1200, friction=[1.2, 0.02, 0.001])
    grip_y = y0 + bar / 2 if shape == "L" else y0 + bar * 0.78
    body.add_site(name=f"{name}_grip", pos=[-L - 0.012, grip_y, 0])
    body.add_site(name=f"{name}_tip")
    return {"body": name, "shaft_geoms": [shaft], "grip_site": f"{name}_grip", "shape": shape,
            "bar_y": (y0, y0 + bar), "shaft": L}


def _cradle(parent, name: str, pos, yaw: float, key_shape: str, shaft: float, bar_y) -> tuple:
    """Three posts a key lies on (bow ends and shaft), bow 5 cm above the surface. Returns the key pose
    (parent frame) that rests on it."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos = np.asarray(pos, float)
    h = 0.07                                 # post top (room for an open hand under the bow); bow mid-plane 12 mm above
    key_pos = pos + np.array([0, 0, h + 0.012])
    pts = [np.array([-shaft - 0.012, bar_y[0] + 0.008, 0]), np.array([-shaft - 0.012, bar_y[1] - 0.008, 0]),
           np.array([-0.012, 0, 0])]
    for i, loc in enumerate(pts):
        top = h + (0.0 if i < 2 else 0.0095)
        w = pos + R @ loc
        parent.add_geom(name=f"{name}_post{i}", type=_g.mjGEOM_BOX, size=[0.006, 0.006, top / 2],
                        pos=[w[0], w[1], pos[2] + top / 2], rgba=[0.3, 0.3, 0.32, 1], density=1000)
    return key_pos, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def _side_cabinet(spec, root, p: dict, cradles: dict, key_def: dict) -> dict:
    """Cabinet beside the door on the latch side (a drawer, a table-top) and a wall shelf above it."""
    ls, W = p["latch_sign"], p["width"]
    CW, CD, CH = 0.42, 0.32, 0.9
    yc = ls * (W / 2 + 0.14 + CW / 2)
    x0, x1 = 0.02, 0.02 + CD
    t = 0.015
    xc = (x0 + x1) / 2
    for nm, half, pos in [("back", [t, CW / 2, CH / 2], [x0 + t, yc, CH / 2]),
                          ("sidea", [CD / 2, t, CH / 2], [xc, yc - CW / 2 + t, CH / 2]),
                          ("sideb", [CD / 2, t, CH / 2], [xc, yc + CW / 2 - t, CH / 2]),
                          ("top", [CD / 2, CW / 2, t], [xc, yc, CH - t]),
                          ("mid", [CD / 2, CW / 2, t], [xc, yc, 0.7]),
                          ("base", [CD / 2 - 0.01, CW / 2, 0.35], [xc - 0.01, yc, 0.35])]:
        _box(root, f"cab_{nm}", half, pos, CAB_RGBA)
    # drawer between z = 0.715 and 0.87, front at x1, pulling out towards +x
    dz0, dz1 = 0.715 + 0.003, CH - 2 * t - 0.003
    dh = dz1 - dz0
    drawer = root.add_body(name="cab_drawer", pos=[x1, yc, dz0])
    dj = "cab_drawer_slide"
    depth = CD - 0.05
    drawer.add_joint(name=dj, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[0, 0.25],
                     limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=[3.0, 0, 0], frictionloss=0.5, armature=0.05)
    iw = CW / 2 - t - 0.006
    _box(drawer, "cab_drawer_front", [0.01, CW / 2 - 0.004, dh / 2], [-0.01, 0, dh / 2], [0.64, 0.5, 0.38, 1])
    _box(drawer, "cab_drawer_floor", [depth / 2, iw, 0.005], [-depth / 2, 0, 0.005], [0.75, 0.65, 0.5, 1])
    for s in (-1, 1):
        _box(drawer, f"cab_drawer_wall{'ab'[s > 0]}", [depth / 2, 0.005, dh / 2 - 0.01],
             [-depth / 2, s * (iw - 0.005), dh / 2], [0.75, 0.65, 0.5, 1])
    _box(drawer, "cab_drawer_rear", [0.005, iw, dh / 2 - 0.01], [-depth + 0.005, 0, dh / 2], [0.75, 0.65, 0.5, 1])
    # horizontal bar handle
    hc = np.array([0.0, 0, dh / 2])
    drawer.add_geom(name="cab_drawer_bar", type=_g.mjGEOM_CAPSULE, size=[0.012, 0.07, 0], pos=(hc + [0.055, 0, 0]).tolist(),
                    quat=_quat_z_to([0, 1, 0]), rgba=STEEL, density=2000, friction=[1.2, 0.02, 0.001])
    for s in (-1, 1):
        _box(drawer, f"cab_drawer_post{'ab'[s > 0]}", [0.0275, 0.007, 0.007], hc + [0.0275, s * 0.058, 0], STEEL)
    drawer.add_site(name="cab_drawer_handle", pos=(hc + [0.055, 0, 0]).tolist(), quat=_frame_quat([1, 0, 0], [0, 1, 0]))
    spec.add_exclude(bodyname1=root.name, bodyname2="cab_drawer")
    # cradles. On the table top and the shelf a key lies with its tip towards the wall and its bow at the
    # front edge, so a hand can take the bow from the front like a horizontal bar; in the drawer it lies
    # the same way and is taken from above (the drawer front is in the way).
    sh, bar_y, shape = key_def["shaft"], key_def["bar_y"], key_def["shape"]
    back = sh + 0.012 + 0.03             # tip-to-front-edge distance
    cradles["table"] = [(root.name,) + _cradle(root, f"cr_table{i}", [x1 - back, yc + dy, CH], np.pi, shape, sh, bar_y)
                        for i, dy in enumerate((-0.1, 0.1))]
    _box(root, "shelf", [0.11, 0.22, 0.012], [0.11, yc, 1.25], CAB_RGBA)
    cradles["shelf"] = [(root.name,) + _cradle(root, "cr_shelf", [0.22 - back, yc, 1.262], np.pi, shape, sh, bar_y)]
    cradles["drawer"] = [("cab_drawer",) + _cradle(drawer, "cr_drawer", [-0.21, 0.0, 0.01], np.pi, shape, sh, bar_y)]
    return {"joint": dj, "site": "cab_drawer_handle", "range": (0.0, 0.25)}


def _other_room(root, p: dict, cradles: dict, key_def: dict) -> None:
    ls = p["latch_sign"]
    y = ls * 0.25
    _box(root, "far_table_top", [0.2, 0.25, 0.015], [-0.75, y, 0.8], CAB_RGBA)
    _box(root, "far_table_leg", [0.05, 0.05, 0.39], [-0.75, y, 0.395], CAB_RGBA)
    cradles["other_room"] = [(root.name,) + _cradle(root, "cr_far", [-0.75, y, 0.815], 0.0, key_def["shape"],
                                                    key_def["shaft"], key_def["bar_y"])]


# ---------------------------------------------------------------------------------------- builder
def add_home_door(spec: mujoco.MjSpec, fx: FixtureSpec, name: str = "home", pos=(0.0, 0.0, 0.0),
                  quat=(1.0, 0.0, 0.0, 0.0)) -> HomeDoor:
    """Builds the wall, door, locks, side cabinet, shelf, far table and keys at pos/quat (world)."""
    p = dict(fx.params)
    W, H = p["width"], p["height"]
    hs = -1.0 if p["hinge"] == "left" else 1.0          # hinge on the agent's left (-y) or right (+y)
    ls = -hs
    p["latch_sign"] = ls
    heights = {"key": p["handle_z"] + p["key_dz"], "thumb": p["handle_z"] + p["thumb_dz"]}
    locks = [k for k in ("thumb", "key") if p[f"has_{k}"]]
    p["bolt_heights"] = [heights[k] for k in locks]
    root = spec.worldbody.add_body(name=f"{name}_base", pos=list(pos), quat=list(quat))
    before = {b.name for b in spec.bodies}
    _wall(root, p)

    pull = p["swing"] == "pull"
    hinge_x = DOOR_X1 if pull else DOOR_X0
    door = root.add_body(name=f"{name}_door", pos=[hinge_x, hs * (W / 2 - GAP), 0.0])
    dj = f"{name}_door_hinge"
    axis = [0, 0, hs if pull else -hs]
    door.add_joint(name=dj, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis, range=[0, 1.9],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=[float(p["damping"]), 0, 0],
                   frictionloss=float(p["frictionloss"]), stiffness=[float(p["closer"]), 0, 0], springref=0.0,
                   armature=0.2, solimp_friction=[0.99, 0.999, 0.001, 0.5, 2.0])
    wl = W - 2 * GAP
    fx_rel = DOOR_X1 - hinge_x                            # the agent-side face, door frame
    _box(door, f"{name}_door_panel", [DOOR_T / 2, wl / 2, (H - 0.01) / 2],
         [fx_rel - DOOR_T / 2, -hs * wl / 2, 0.005 + (H - 0.01) / 2], p.get("door_rgba", DOOR_RGBA), density=400)
    n = np.array([1.0, 0, 0])
    to_hinge = np.array([0, hs, 0])
    to_latch = -to_hinge
    y_hw = -hs * (wl - p["handle_offset"])                 # hardware line near the latch edge
    face = lambda z: np.array([fx_rel, y_hw, z])           # noqa: E731
    hj, hsite, mode, thr, hrange = _handle(spec, door, name, p, face(p["handle_z"]), n, to_hinge)
    latch_eq = f"{name}_latch"
    spec.add_equality(name=latch_eq, type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=dj,
                      active=False, data=[0.0] * 11, solref=[0.005, 1.0])
    edge = lambda z: np.array([(DOOR_X0 + DOOR_X1) / 2 - hinge_x, -hs * wl, z])   # noqa: E731
    rec = HomeDoor(name, fx, f"{name}_door", dj, latch_eq, p["handle"], hj, hsite, mode, thr, hrange)
    if p["has_thumb"]:
        rec.thumb = _thumb_turn(spec, door, name, face(heights["thumb"]), n, to_latch, p)
        bj = _bolt(spec, door, f"{name}_thumb", edge(heights["thumb"]), to_latch, STEEL)
        _tail_eq(spec, bj, rec.thumb["joint"])
        rec.bolts["thumb"] = {"joint": bj, "tail": rec.thumb["joint"]}
    if p["has_key"]:
        rec.key_lock = _key_lock(spec, door, name, face(heights["key"]), n, p)
        bj = _bolt(spec, door, f"{name}_key", edge(heights["key"]), to_latch, BRASS)
        _tail_eq(spec, bj, rec.key_lock["tail"])
        rec.bolts["key"] = {"joint": bj, "tail": rec.key_lock["tail"]}
    spec.add_exclude(bodyname1=root.name, bodyname2=door.name)

    # keys
    rng = np.random.default_rng(int(p["key_seed"]))
    key_defs = []
    for i in range(int(p["n_keys"])):
        rgba = [*rng.uniform(0.2, 0.95, 3), 1.0]
        kd = _key(spec, f"{name}_key{i}", p["key_shape"], p, rgba)
        kd["bitting"] = int(p["key_bittings"][i])
        weld = f"{name}_key{i}_weld"
        if rec.key_lock:
            spec.add_equality(name=weld, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=kd["body"], name2=rec.key_lock["carrier_body"], active=False,
                              data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], solref=[0.01, 1.0])
            kd["weld_eq"] = weld
        key_defs.append(kd)
    rec.keys = key_defs
    rec.drawer = _side_cabinet(spec, root, p, rec.cradles, key_defs[0])
    _other_room(root, p, rec.cradles, key_defs[0])
    rec.bodies = [root.name] + [b.name for b in spec.bodies if b.name not in before]
    return rec


# ---------------------------------------------------------------------------------------- runtime
class HomeDoorRuntime:
    """Per-episode state: lock state, key placement, the latch and key mechanics, events."""

    def __init__(self, rec: HomeDoor, causes: dict | None = None, placements: dict | None = None):
        self.rec = rec
        self.cause_probs = causes or {s: 1.0 for s in LOCK_STATES}
        self.place_probs = placements or {"table": 1.0, "shelf": 1.0, "drawer": 1.0, "other_room": 0.5}
        self.events: list[dict] = []
        self.t = 0

    # ------------------------------------------------------------------ binding
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        m, r = model, self.rec
        self.model, self.data = model, data
        J = lambda n: m.joint(n)                                  # noqa: E731
        self.door = {"q": J(r.door_joint).qposadr[0], "dof": J(r.door_joint).dofadr[0], "jnt": J(r.door_joint).id,
                     "range": tuple(m.jnt_range[J(r.door_joint).id]), "site": m.site(r.handle_site).id}
        self.latch = m.equality(r.latch_eq).id
        self.handle_q = J(r.handle_joint).qposadr[0]
        self.handle_jnt = J(r.handle_joint).id
        self.bolt_q = {k: J(v["joint"]).qposadr[0] for k, v in r.bolts.items()}
        self.tail_q = {k: J(v["tail"]).qposadr[0] for k, v in r.bolts.items()}
        if r.thumb:
            self.thumb = {"q": J(r.thumb["joint"]).qposadr[0], "jnt": J(r.thumb["joint"]).id, "site": m.site(r.thumb["site"]).id}
        if r.key_lock:
            kl = r.key_lock
            self.kl = {"plug_q": J(kl["plug"]).qposadr[0], "plug_jnt": J(kl["plug"]).id,
                       "car_q": J(kl["carrier"]).qposadr[0], "car_body": m.body(kl["carrier_body"]).id,
                       "plug_body": m.body(kl["plug_body"]).id,
                       "lock": m.equality(kl["lock_eq"]).id, "drive": m.equality(kl["drive_eq"]).id,
                       "hold": m.equality(kl["hold_eq"]).id, "site": m.site(kl["site"]).id}
        self.keys = []
        for k in r.keys:
            b = m.body(k["body"]).id
            jid = m.body_jntadr[b]
            geoms = [m.geom(g).id for g in k["shaft_geoms"]]
            self.keys.append({"body": b, "qpos": m.jnt_qposadr[jid], "qvel": m.jnt_dofadr[jid], "geoms": geoms,
                              "contype": m.geom_contype[geoms].copy(), "conaff": m.geom_conaffinity[geoms].copy(),
                              "weld": m.equality(k["weld_eq"]).id if "weld_eq" in k else None,
                              "grip": m.site(k["grip_site"]).id, "bitting": k["bitting"], "name": k["body"]})
        self.cradle_frames = {}
        for place, lst in r.cradles.items():
            self.cradle_frames[place] = [(m.body(bn).id, np.asarray(pos, float), np.asarray(q, float)) for bn, pos, q in lst]
        self.drawer = {"q": J(r.drawer["joint"]).qposadr[0], "jnt": J(r.drawer["joint"]).id,
                       "site": m.site(r.drawer["site"]).id, "range": r.drawer["range"]}

    # ------------------------------------------------------------------ reset
    def available_states(self) -> list[str]:
        r = self.rec
        return [s for s in LOCK_STATES if (s in ("unlocked",) or (s == "deadbolt" and r.thumb) or
                                            (s == "key" and r.key_lock) or (s == "both" and r.thumb and r.key_lock))]

    def reset(self, rng: np.random.Generator, lock_state: str | None = None, key_place: str | None = None) -> None:
        m, d, r = self.model, self.data, self.rec
        states = self.available_states()
        w = np.array([self.cause_probs.get(s, 0.0) for s in states], float)
        u1, u2, u3 = rng.uniform(size=3)                     # fixed number of draws per reset
        state = states[min(int(np.searchsorted(np.cumsum(w) / w.sum(), u1, side="right")), len(states) - 1)]
        state = lock_state or state
        if state not in states:
            raise ValueError(f"{r.spec.type}: lock state {state!r} not available (have {states})")
        places = list(PLACEMENTS)
        pw = np.array([self.place_probs.get(pl, 0.0) for pl in places], float)
        place = places[min(int(np.searchsorted(np.cumsum(pw) / pw.sum(), u2, side="right")), len(places) - 1)]
        place = key_place or place
        self.lock_state, self.key_place = state, place if r.key_lock else None
        # bolts: thumb and/or key deadbolt thrown
        thrown = {"thumb": state in ("deadbolt", "both"), "key": state in ("key", "both")}
        for k in r.bolts:
            ang = THUMB_RANGE if thrown[k] else 0.0
            d.qpos[self.tail_q[k]] = ang
            d.qpos[self.bolt_q[k]] = BOLT_THROW * ang / THUMB_RANGE
        # keys: the matching one (if any) at `place`, the others on the remaining cradles
        match = [i for i, k in enumerate(self.keys) if r.key_lock and k["bitting"] == r.key_lock["bitting"]]
        slots = [(pl, i) for pl in PLACEMENTS for i in range(len(self.cradle_frames[pl]))]
        order = rng.permutation(len(slots))
        used = set()
        assign = {}
        if match:
            cand = [s for s in slots if s[0] == place]
            assign[match[0]] = cand[int(u3 * len(cand)) % len(cand)]
            used.add(assign[match[0]])
        free = [slots[i] for i in order if slots[i] not in used and slots[i][0] != "other_room"]  # decoys: reachable
        for i in range(len(self.keys)):
            if i not in assign:
                assign[i] = free.pop(0)
        self.key_slots = {self.keys[i]["name"]: assign[i] for i in range(len(self.keys))}
        mujoco.mj_kinematics(m, d)
        for i, k in enumerate(self.keys):
            pl, j = assign[i]
            bid, pos, quat = self.cradle_frames[pl][j]
            R = d.xmat[bid].reshape(3, 3)
            wq = np.zeros(4)
            mujoco.mju_mulQuat(wq, d.xquat[bid], quat)
            d.qpos[k["qpos"]:k["qpos"] + 3] = d.xpos[bid] + R @ pos
            d.qpos[k["qpos"] + 3:k["qpos"] + 7] = wq
            d.qvel[k["qvel"]:k["qvel"] + 6] = 0
            m.geom_contype[k["geoms"]] = k["contype"]
            m.geom_conaffinity[k["geoms"]] = k["conaff"]
            if k["weld"] is not None:
                d.eq_active[k["weld"]] = 0
        if r.key_lock:
            m.eq_data[self.kl["lock"], 0] = 0.0
        self.welded: int | None = None
        self.inserted = False
        self.cooldown: set[int] = set()
        self._drive_on = False
        self.t = 0
        self.events = []
        self._state: dict = {}
        self._pending: dict = {}
        self._last_key = None
        self.update()
        self._init_state()
        self._log("home_reset", truth=self.info())

    # ------------------------------------------------------------------ mechanics
    def handle_released(self) -> bool:
        q = self.data.qpos[self.handle_q]
        if self.rec.handle_mode == "both":
            return abs(q) > self.rec.handle_threshold
        return q > self.rec.handle_threshold

    def update(self) -> None:
        """Called every control step before the physics: the latch, key capture / release, plug coupling."""
        m, d, r = self.model, self.data, self.rec
        door_q = d.qpos[self.door["q"]]
        d.eq_active[self.latch] = bool(door_q < LATCH_CATCH and not self.handle_released())
        if not r.key_lock:
            return
        kl = self.kl
        car = d.qpos[kl["car_q"]]
        plug = d.qpos[kl["plug_q"]]
        mouth = d.xpos[kl["car_body"]] - car * d.xmat[kl["car_body"]].reshape(3, 3)[:, 0]  # s = 0 point
        Rm = d.xmat[kl["car_body"]].reshape(3, 3)
        if self.welded is None:
            for i, k in enumerate(self.keys):
                tip = d.xpos[k["body"]]
                dist = np.linalg.norm(tip - mouth)
                if i in self.cooldown:
                    if dist > 0.02:
                        self.cooldown.discard(i)
                    continue
                if dist > KEY_CAPTURE_DIST or abs(plug) > 0.1:
                    continue
                Rk = d.xmat[k["body"]].reshape(3, 3)
                if np.dot(Rk[:, 0], Rm[:, 0]) < np.cos(KEY_CAPTURE_ANGLE):
                    continue
                roll = np.dot(Rk[:, 1], Rm[:, 1])
                if abs(roll) < np.cos(KEY_CAPTURE_ROLL):
                    continue
                # capture: weld the key to the carrier (flipped keys fit the other way round)
                m.eq_data[k["weld"], 3:10] = [0, 0, 0, 1, 0, 0, 0] if roll > 0 else [0, 0, 0, 0, 1, 0, 0]
                d.eq_active[k["weld"]] = 1
                m.geom_contype[k["geoms"]] = 0
                m.geom_conaffinity[k["geoms"]] = 0
                self.welded, self._deep = i, False
                d.qpos[kl["car_q"]] = 0.0
                break
        else:
            k = self.keys[self.welded]
            self._deep = self._deep or car > 0.01
            if self._deep and car < 0.002 and abs(plug) < 0.1:   # pulled out at home: let go
                d.eq_active[k["weld"]] = 0
                m.geom_contype[k["geoms"]] = k["contype"]
                m.geom_conaffinity[k["geoms"]] = k["conaff"]
                self.cooldown.add(self.welded)
                self.welded = None
        # fully inserted from KEY_INSERTED on; stays so down to 18 mm, and always while the plug is turned
        # (a turned key cannot be withdrawn)
        self.inserted = self.welded is not None and (car > KEY_INSERTED or (self.inserted and (car > 0.018 or
                                                                                              abs(plug) > 0.12)))
        match = self.inserted and self.keys[self.welded]["bitting"] == r.key_lock["bitting"]
        if match and not self._drive_on:     # engage the plug -> tailpiece coupling without a jump
            tail = d.qpos[self.tail_q["key"]]
            m.eq_data[kl["drive"], 0] = tail + plug        # tail = a0 - plug
            self._drive_on = True
        elif not match and self._drive_on:
            self._drive_on = False
            m.eq_data[kl["lock"], 0] = plug       # the plug stays where it is (it only gets here at home)
        d.eq_active[kl["drive"]] = self._drive_on
        d.eq_active[kl["lock"]] = not self._drive_on
        d.eq_active[kl["hold"]] = bool(self.welded is not None and abs(plug) > 0.12)

    # ------------------------------------------------------------------ sensing
    def _part(self, jnt: int, qadr: int, site: int, lo: float, hi: float) -> dict:
        d = self.data
        q = float(d.qpos[qadr])
        R = d.site_xmat[site].reshape(3, 3)
        axis, anchor = d.xaxis[jnt].copy(), d.xanchor[jnt].copy()
        pos = d.site_xpos[site].copy()
        kind = self.model.jnt_type[jnt]
        v = axis if kind == mujoco.mjtJoint.mjJNT_SLIDE else np.cross(axis, pos - anchor)
        nv = np.linalg.norm(v)
        return {"q": q, "opening": float(np.clip((q - lo) / (hi - lo), 0, 1)) if hi > lo else 0.0,
                "handle_pos": pos, "handle_rot": R.copy(), "normal": R[:, 0].copy(), "handle_axis": R[:, 2].copy(),
                "joint_axis": axis, "joint_anchor": anchor, "pull_dir": v / nv if nv > 1e-9 else v}

    def observe(self) -> dict:
        """What a body could see or feel: joint angles of visible parts, visible keys. No bolts, latch,
        carrier depth, bittings or lock state."""
        d, r = self.data, self.rec
        lo, hi = self.door["range"]
        arts = {"door": self._part(self.door["jnt"], self.door["q"], self.door["site"], lo, hi)}
        hl, hh = r.handle_range
        arts["handle"] = self._part(self.handle_jnt, self.handle_q, self.door["site"], min(hl, 0), hh)
        arts["handle"]["kind"] = r.handle
        if r.thumb:
            arts["thumb_turn"] = self._part(self.thumb["jnt"], self.thumb["q"], self.thumb["site"], 0, THUMB_RANGE)
        if r.key_lock:
            arts["key_lock"] = self._part(self.kl["plug_jnt"], self.kl["plug_q"], self.kl["site"], -1.7, 1.7)
            R = d.xmat[self.kl["plug_body"]].reshape(3, 3)
            arts["key_lock"].update(keyhole_pos=d.xpos[self.kl["plug_body"]].copy(), keyhole_axis=R[:, 0].copy(),
                                    slot_dir=R[:, 1].copy())
        dl, dh = self.drawer["range"]
        arts["drawer"] = self._part(self.drawer["jnt"], self.drawer["q"], self.drawer["site"], dl, dh)
        keys = {}
        door_open = arts["door"]["opening"] > 0.3
        drawer_open = arts["drawer"]["opening"] > 0.5
        for k in self.keys:
            pl = self.key_slots[k["name"]][0]
            visible = self._left_slot(k) or not ((pl == "drawer" and not drawer_open)
                                                 or (pl == "other_room" and not door_open))
            if visible:
                keys[k["name"]] = {"pos": d.xpos[k["body"]].copy(), "quat": d.xquat[k["body"]].copy(),
                                   "grip_pos": d.site_xpos[k["grip"]].copy()}
        return {"articulations": arts, "keys": keys}

    def _left_slot(self, k: dict) -> bool:
        """The key has been taken off its cradle (so it is no longer hidden where it was)."""
        pl, j = self.key_slots[k["name"]]
        bid, pos, _ = self.cradle_frames[pl][j]
        d = self.data
        rest = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ pos
        return bool(np.linalg.norm(d.xpos[k["body"]] - rest) > 0.05)

    def bolt_thrown(self, name: str) -> bool:
        return bool(self.data.qpos[self.bolt_q[name]] > 0.5 * BOLT_THROW)

    def info(self) -> dict:
        """Ground truth: type, parameters, lock state, where each key lies and which one matches."""
        r = self.rec
        return {"type": r.spec.type, "params": {k: v for k, v in r.spec.params.items() if not k.endswith("rgba")},
                "lock_state": self.lock_state, "key_place": self.key_place,
                "deadbolt_locked": "thumb" in r.bolts and self.bolt_thrown("thumb"),
                "key_locked": "key" in r.bolts and self.bolt_thrown("key"),
                "lock_bitting": r.key_lock["bitting"] if r.key_lock else None,
                "keys": {k["name"]: {"bitting": k["bitting"], "slot": list(self.key_slots[k["name"]]),
                                     "matches": bool(r.key_lock and k["bitting"] == r.key_lock["bitting"])}
                         for k in self.keys},
                "solvable": self.solvable()}

    def solvable(self) -> bool:
        if self.lock_state in ("key", "both"):
            return self.key_place != "other_room"
        return True

    # ------------------------------------------------------------------ events
    def _flags(self) -> dict:
        d, r = self.data, self.rec
        lo, hi = self.door["range"]
        door_frac = (d.qpos[self.door["q"]] - lo) / (hi - lo)
        f = {"knob_turned": self.handle_released(),
             "latch_released": bool(d.qpos[self.door["q"]] < LATCH_CATCH and self.handle_released()),
             "door_opened": door_frac > DOOR_OPEN_FRAC, "door_closed": d.qpos[self.door["q"]] < 0.02,
             "drawer_opened": (d.qpos[self.drawer["q"]] / self.drawer["range"][1]) > 0.5}
        for k in r.bolts:
            f[f"bolt_extended:{k}"] = self.bolt_thrown(k)
        if r.key_lock:
            f["key_inserted"] = self.inserted
            f["key_turned"] = bool(abs(d.qpos[self.kl["plug_q"]]) > KEY_TURNED)
        return f

    def _init_state(self) -> None:
        self._state = self._flags()

    def settled(self) -> None:
        """After the env's settling steps: events start from the settled state at t = 0."""
        self.t = 0
        self._pending = {}
        self._init_state()

    def post_step(self) -> None:
        """Debounced edge events, after the physics step."""
        self.t += 1
        for name, v in self._flags().items():
            if v == self._state.get(name):
                self._pending.pop(name, None)
                continue
            self._pending[name] = self._pending.get(name, 0) + 1
            if self._pending[name] < DEBOUNCE:
                continue
            self._pending.pop(name)
            self._state[name] = v
            base, _, which = name.partition(":")
            if base == "bolt_extended":
                self._log("bolt_extended" if v else "bolt_retracted", bolt=which)
            elif base == "key_inserted":
                k = self.keys[self.welded] if self.welded is not None else None
                name_k = k["name"] if k else self._last_key
                self._log("key_inserted" if v else "key_removed", key=name_k,
                          matches=bool(name_k and self._key_bitting(name_k) == self.rec.key_lock["bitting"]))
                self._last_key = name_k
            elif v:   # knob_turned, latch_released, key_turned, door_opened, door_closed, drawer_opened
                self._log(base)

    def _key_bitting(self, name: str) -> int:
        return next(k["bitting"] for k in self.keys if k["name"] == name)

    def _log(self, kind: str, **details) -> None:
        self.events.append({"t": self.t, "type": kind, **details})

    def of_type(self, *kinds: str) -> list[dict]:
        return [e for e in self.events if e["type"] in kinds]


# ---------------------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class HomeDoorType:
    name: str
    split: str
    fixed: dict


def _types() -> dict[str, HomeDoorType]:
    T = HomeDoorType
    ts = [
        T("hd_knob_pull_left", "train", {"handle": "knob_round", "knob_dir": "both", "swing": "pull", "hinge": "left",
                                         "has_thumb": True, "has_key": True, "key_shape": "L", "closer_kind": "light"}),
        T("hd_knob_push_right", "train", {"handle": "knob_round", "knob_dir": "one", "swing": "push", "hinge": "right",
                                          "has_thumb": False, "has_key": True, "key_shape": "L", "closer_kind": "none"}),
        T("hd_lever_pull_right", "train", {"handle": "lever", "knob_dir": "both", "swing": "pull", "hinge": "right",
                                           "has_thumb": True, "has_key": True, "key_shape": "L", "closer_kind": "strong"}),
        T("hd_pushbar_push_left", "train", {"handle": "push_bar", "knob_dir": "both", "swing": "push", "hinge": "left",
                                            "has_thumb": True, "has_key": False, "key_shape": "L", "closer_kind": "strong"}),
        T("hd_ball_pull_left", "train", {"handle": "knob_ball", "knob_dir": "both", "swing": "pull", "hinge": "left",
                                         "has_thumb": False, "has_key": True, "key_shape": "L", "closer_kind": "none"}),
        # held out: unseen combinations, T-shaped keys, a lever on a push door, a ball knob turning one way
        T("hd_lever_push_left", "test", {"handle": "lever", "knob_dir": "both", "swing": "push", "hinge": "left",
                                         "has_thumb": True, "has_key": True, "key_shape": "T", "closer_kind": "light"}),
        T("hd_ball_push_right", "test", {"handle": "knob_ball", "knob_dir": "one", "swing": "push", "hinge": "right",
                                         "has_thumb": True, "has_key": True, "key_shape": "T", "closer_kind": "strong"}),
        T("hd_knob_pull_right", "test", {"handle": "knob_round", "knob_dir": "one", "swing": "pull", "hinge": "right",
                                         "has_thumb": True, "has_key": True, "key_shape": "T", "closer_kind": "light"}),
    ]
    return {t.name: t for t in ts}


HOME_DOOR_TYPES: dict[str, HomeDoorType] = _types()
CLOSERS = {"none": 0.0, "light": 3.0, "strong": 8.0}   # Nm/rad


def home_door_types(split: str | None = None) -> list[str]:
    return [n for n, t in HOME_DOOR_TYPES.items() if split is None or t.split == split]


def make_home_door(rng: np.random.Generator, type_name: str, **overrides) -> FixtureSpec:
    t = HOME_DOOR_TYPES[type_name]
    u = lambda lo, hi: float(rng.uniform(lo, hi))   # noqa: E731
    n_keys = int(rng.integers(1, 4))
    bitting = int(rng.integers(10))
    decoys = [int(b) for b in rng.choice([b for b in range(10) if b != bitting], size=n_keys - 1, replace=False)]
    bittings = [bitting] + decoys
    rng.shuffle(bittings)
    p = {"width": u(0.82, 0.92), "height": u(2.0, 2.08), "handle_z": u(0.95, 1.02), "handle_offset": u(0.065, 0.075),
         "key_dz": u(0.13, 0.16), "thumb_dz": u(0.27, 0.31), "damping": u(1.5, 3.0), "frictionloss": u(0.3, 0.8),
         "bitting": bitting, "n_keys": n_keys, "key_bittings": bittings, "key_shaft": u(0.08, 0.095),
         "key_seed": int(rng.integers(1 << 30)),
         "door_rgba": [u(0.4, 0.7), u(0.25, 0.45), u(0.15, 0.3), 1.0]}
    p.update(t.fixed)
    p["closer"] = CLOSERS[p["closer_kind"]]
    p.update(overrides)
    if "n_keys" in overrides and "key_bittings" not in overrides:   # keep the bitting list consistent
        n = int(p["n_keys"])
        p["key_bittings"] = ([p["bitting"]] + [b for b in range(10) if b != p["bitting"]][:n - 1])
    return FixtureSpec(t.name, "home_door", p)


def sample_home_door(rng: np.random.Generator, split: str = "train") -> FixtureSpec:
    names = home_door_types(split)
    return make_home_door(rng, names[int(rng.integers(len(names)))])


# ---------------------------------------------------------------------------------------- scene adder
ANCHOR_SHIFT = 0.18   # m: bodies are placed around a point this far from the handle towards the side cabinet


def body_options(params: dict, yaw_world: float) -> dict:
    """Where bodies stand for a home door. The Panda stands on a raised stand (base 0.5 m below the
    handle) close enough to reach the thumb-turn (1.3 m), the drawer and the key cradles; for a pull door
    it stands further to the latch side (the door swings in on the hinge side), for a push door closer
    (it follows the door away). Floating hands get a 1 m workspace so they can follow a push door."""
    lat = 1.0 if params["hinge"] == "left" else -1.0
    c, s = np.cos(yaw_world), np.sin(yaw_world)
    side = lat * np.array([-s, c, 0.0])                   # world direction of the latch side
    pull = params["swing"] == "pull"
    panda = {"stand_back": 0.62 if pull else 0.5, "stand_drop": 0.5, "stand_offset": tuple(side * (0.08 if pull else 0.0))}
    hand = {"reach": 1.0}
    return {"panda_2f": panda, **{n: hand for n in ("robotiq_2f85", "allegro", "leap", "shadow")}}

def home_door_fixture(type_name: str | None = None, fixture: FixtureSpec | None = None,
                      causes: dict | None = None, placements: dict | None = None, **overrides):
    """DoorSceneEnv adder: a home door of `type_name` (parameters drawn from the env's rng) or the given
    FixtureSpec. causes / placements: probabilities of the lock states / key placements per reset."""

    def adder(spec: mujoco.MjSpec, pose=((0.5, 0.0, 0.0), 0.0), rng=None) -> HandleInfo:
        rng = rng or np.random.default_rng()
        fx = fixture or make_home_door(rng, type_name, **overrides)
        (x, y, z), yaw = pose
        rec = add_home_door(spec, fx, "home", pos=(x, y, z), quat=_yaw_quat(yaw + np.pi))
        rt = HomeDoorRuntime(rec, causes, placements)
        # place the body between the door handle and the side cabinet (on the latch side)
        lat = 1.0 if fx.params["hinge"] == "left" else -1.0     # latch side, fixture-frame y
        c, s = np.cos(yaw + np.pi), np.sin(yaw + np.pi)
        shift = ANCHOR_SHIFT * lat * np.array([-s, c, 0.0])
        return HandleInfo("home", "home_door", rec.door_joint, rec.handle_site, (0, 0, 1), (-1, 0, 0), (0.0, 1.9),
                          DOOR_OPEN_FRAC * 1.9, runtime=rt, anchor_shift=tuple(shift),
                          body_options=body_options(fx.params, yaw + np.pi))

    return adder
