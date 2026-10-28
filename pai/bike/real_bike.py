"""A realistic 24-inch bicycle and an outdoor road scene for the G1 (Stage 7 with the Stage 2 body).

The first Stage 7 bicycle (``pai/bike/env.py``) is a functional sketch: capsule tubes and flat disc
wheels on a bare plane. This module builds a bicycle with standard geometry, sized for the 1.32 m
Unitree G1 (a 24-inch wheel, 0.98 m wheelbase, 73 degree seat tube, 70 degree head angle, 0.42 m
chainstays, about 12 kg), with torus tyres and rims, spokes, a saddle on a seat post, a stem and
handlebar with grips, a chainring, crank arms and pedals, on an asphalt road through grass with a
sky, sunlight and shadows, trees and kerbs.

The detailed shapes are visual. Physics uses robust collision shapes: each wheel touches the ground
through a thin cylinder of the tyre's radius, as in the first bicycle. Joint, actuator and sensor names
match ``pai/bike/env.py`` (``root``, ``rear_spin``, ``front_spin``, ``steering``, ``pedal``,
``steer``, ``orientation``, ``angular_velocity``, ``velocity``), plus a ``crank`` joint geared to the
rear wheel and sites ``saddle``, ``grip_l/r``, ``pedal_l/r`` for a rider.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

R_TYRE = 0.305          # 24-inch wheel, outer radius (m)
R_RIM = 0.27
TYRE_TUBE = 0.024
WHEELBASE = 0.98
CRANK = 0.15
GEAR = 0.5              # crank angle per rear-wheel angle

# side-view points (x forward from the rear axle, z up), metres
REAR_AXLE = np.array([0.0, R_TYRE])
FRONT_AXLE = np.array([WHEELBASE, R_TYRE])
BB = np.array([0.42, 0.27])
SEAT_TOP = BB + 0.44 * np.array([-math.cos(math.radians(73)), math.sin(math.radians(73))])
HEAD_TOP = np.array([0.80, 0.76])
HEAD_AXIS = np.array([math.cos(math.radians(110)), math.sin(math.radians(110))])  # 70 deg head angle
HEAD_BOT = HEAD_TOP - 0.13 * HEAD_AXIS

FRAME_RGBA = [0.75, 0.08, 0.08, 1]
BLACK = [0.06, 0.06, 0.07, 1]
SILVER = [0.75, 0.76, 0.78, 1]


def _p(xz, y=0.0):
    return [float(xz[0]), float(y), float(xz[1])]


def _torus(spec, name, major, minor, n_major=40, n_minor=10):
    """A torus mesh in the x-z plane (axis along y), for tyres and rims."""
    verts, faces = [], []
    for i in range(n_major):
        a = 2 * math.pi * i / n_major
        for j in range(n_minor):
            b = 2 * math.pi * j / n_minor
            r = major + minor * math.cos(b)
            verts.append([r * math.cos(a), minor * math.sin(b), r * math.sin(a)])
    for i in range(n_major):
        for j in range(n_minor):
            a, b = i * n_minor + j, ((i + 1) % n_major) * n_minor + j
            c, d = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor, i * n_minor + (j + 1) % n_minor
            faces += [[a, b, c], [a, c, d]]
    spec.add_mesh(name=name, uservert=np.asarray(verts, float).ravel().tolist(),
                  userface=np.asarray(faces, int).ravel().tolist())


def _tube(body, a, b, r, rgba, mass=0.0, origin=(0.0, 0.0, 0.0)):
    o = np.asarray(origin, float)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=list(np.asarray(a) - o) + list(np.asarray(b) - o),
                  size=[r, 0, 0], rgba=rgba, mass=mass, contype=0, conaffinity=0)


def _wheel(parent, name, center, origin):
    w = parent.add_body(name=name, pos=list(np.asarray(center) - np.asarray(origin)))
    w.add_joint(name=name.replace("_wheel", "_spin"), axis=[0, 1, 0], damping=0.002)
    # contact: a thin cylinder of the tyre radius (robust rolling contact)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[R_TYRE, 0.012, 0], euler=[math.pi / 2, 0, 0],
               mass=1.6, rgba=[0, 0, 0, 0], friction=[1.2, 0.005, 0.001], condim=4, contype=1, conaffinity=2)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tyre", rgba=BLACK, contype=0, conaffinity=0, mass=0)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="rim", rgba=SILVER, contype=0, conaffinity=0, mass=0)
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.035, 0], euler=[math.pi / 2, 0, 0],
               rgba=SILVER, contype=0, conaffinity=0, mass=0)
    for k in range(16):
        a = 2 * math.pi * k / 16
        side = 0.03 if k % 2 else -0.03
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[0, side, 0, (R_RIM - 0.01) * math.cos(a), 0, (R_RIM - 0.01) * math.sin(a)],
                   size=[0.0015, 0, 0], rgba=SILVER, contype=0, conaffinity=0, mass=0)
    return w


def _scenery(spec, rng):
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.45, 0.65, 0.92],
                     rgb2=[0.92, 0.95, 1.0], width=512, height=3072)
    spec.add_texture(name="grass", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[0.27, 0.45, 0.18], rgb2=[0.2, 0.36, 0.13],
                     mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.25, markrgb=[0.36, 0.55, 0.22],
                     width=512, height=512)
    spec.add_texture(name="asphalt", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[0.22, 0.22, 0.23], rgb2=[0.2, 0.2, 0.2],
                     mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.35, markrgb=[0.33, 0.33, 0.34],
                     width=512, height=512)
    grass = spec.add_material(name="grass", texrepeat=[60, 60], reflectance=0.0)
    grass.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grass"
    road = spec.add_material(name="road", texrepeat=[40, 2], reflectance=0.0)
    road.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "asphalt"
    wb = spec.worldbody
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=[-0.4, 0.5, -1.0],
                 diffuse=[0.85, 0.82, 0.75], specular=[0.2, 0.2, 0.2], castshadow=True)
    wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[200, 200, 0.1], material="grass",
                contype=2, conaffinity=1, friction=[1.2, 0.005, 0.001], condim=4)
    # the road: a visual strip on the ground, with a dashed centre line and kerbs
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[20, 0, 0.001], size=[40, 1.6, 0.001], material="road",
                contype=0, conaffinity=0)
    for x in np.arange(-18, 60, 3.0):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[float(x), 0.9, 0.0025], size=[0.75, 0.05, 0.001],
                    rgba=[0.95, 0.95, 0.9, 1], contype=0, conaffinity=0)
    for y in (-1.65, 1.65):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[20, y, 0.06], size=[40, 0.06, 0.06],
                    rgba=[0.78, 0.77, 0.74, 1], contype=0, conaffinity=0)
    # trees along both sides
    for k, x in enumerate(np.arange(-12, 60, 6.5)):
        for side in (-1, 1):
            y = side * (4.5 + 2.0 * rng.random())
            xx = float(x + 2.5 * rng.random())
            h = 2.2 + 1.2 * rng.random()
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[xx, y, h / 2], size=[0.12, h / 2, 0],
                        rgba=[0.36, 0.25, 0.16, 1], contype=0, conaffinity=0)
            green = [0.16 + 0.08 * rng.random(), 0.38 + 0.12 * rng.random(), 0.14, 1]
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=[xx, y, h + 0.6],
                        size=[1.1, 1.1, 1.3], rgba=green, contype=0, conaffinity=0)


def build_spec(scenery=True, seed: int = 0) -> mujoco.MjSpec:
    """``scenery``: True (the straight road), False (bare ground) or a callable(spec, rng) building a scene."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.quality.shadowsize = 4096
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    _torus(spec, "tyre", R_TYRE - TYRE_TUBE, TYRE_TUBE)
    _torus(spec, "rim", R_RIM, 0.012, n_minor=6)
    if callable(scenery):
        scenery(spec, np.random.default_rng(seed))      # e.g. pai/bike/town.py
    elif scenery:
        _scenery(spec, np.random.default_rng(seed))
    else:
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[200, 200, 0.1],
                                contype=2, conaffinity=1, friction=[1.2, 0.005, 0.001], condim=4)
    o = _p(BB)                                     # the frame body sits at the bottom bracket
    frame = spec.worldbody.add_body(name="frame", pos=o)
    frame.add_freejoint(name="root")
    frame.add_site(name="imu", pos=[0, 0, 0.2], size=[0.015, 0, 0], rgba=[0, 0, 0, 0])
    # diamond frame: (start, end, radius, mass)
    tubes = [(BB, SEAT_TOP, 0.016, 1.2), (SEAT_TOP, HEAD_TOP - 0.02 * HEAD_AXIS, 0.015, 1.1),
             (BB, HEAD_BOT, 0.019, 1.5), (HEAD_BOT, HEAD_TOP, 0.022, 0.4)]
    for a, b, r, mass in tubes:
        _tube(frame, _p(a), _p(b), r, FRAME_RGBA, mass, o)
    for side in (-0.06, 0.06):
        _tube(frame, _p(BB, side * 0.6), _p(REAR_AXLE, side), 0.010, FRAME_RGBA, 0.35, o)        # chainstays
        _tube(frame, _p(SEAT_TOP - [0, 0.03], side * 0.5), _p(REAR_AXLE, side), 0.009, FRAME_RGBA, 0.3, o)  # seatstays
    post_top = SEAT_TOP + 0.10 * np.array([-math.cos(math.radians(73)), math.sin(math.radians(73))])
    _tube(frame, _p(SEAT_TOP), _p(post_top), 0.013, SILVER, 0.25, o)
    frame.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=list(np.asarray(_p(post_top + [0.02, 0.025])) - o),
                   size=[0.13, 0.07, 0.025], rgba=BLACK, mass=0.3, contype=0, conaffinity=0)
    frame.add_site(name="saddle", pos=list(np.asarray(_p(post_top + [0.0, 0.05])) - o), size=[0.02, 0, 0], rgba=[0, 0, 0, 0])
    _wheel(frame, "rear_wheel", _p(REAR_AXLE), o)
    # steering: fork, stem, handlebar and front wheel turn about the head-tube axis
    fork = frame.add_body(name="fork", pos=list(np.asarray(_p(HEAD_BOT)) - o))
    fork.add_joint(name="steering", axis=[float(HEAD_AXIS[0]), 0, float(HEAD_AXIS[1])], range=[-0.65, 0.65],
                   damping=1.0)
    fo = np.asarray(_p(HEAD_BOT))
    crown = HEAD_BOT
    for side in (-0.055, 0.055):
        _tube(fork, _p(crown, side * 0.5), _p(FRONT_AXLE, side), 0.011, FRAME_RGBA, 0.4, fo)
    stem_top = HEAD_TOP + 0.10 * HEAD_AXIS
    bar = stem_top + np.array([-0.06, 0.05])
    _tube(fork, _p(HEAD_TOP), _p(stem_top), 0.014, SILVER, 0.15, fo)
    _tube(fork, _p(stem_top), _p(bar), 0.013, SILVER, 0.1, fo)
    _tube(fork, _p(bar, -0.27), _p(bar, 0.27), 0.012, SILVER, 0.3, fo)
    for side in (-0.24, 0.24):
        _tube(fork, _p(bar, side - 0.05), _p(bar, side + 0.05), 0.018, BLACK, 0.02, fo)
        fork.add_site(name="grip_l" if side > 0 else "grip_r", pos=list(np.asarray(_p(bar, side)) - fo),
                      size=[0.02, 0, 0], rgba=[0, 0, 0, 0])
    _wheel(fork, "front_wheel", _p(FRONT_AXLE), fo)
    # crank, chainring and pedals, geared to the rear wheel
    crank = frame.add_body(name="crank", pos=[0, 0, 0])
    crank.add_joint(name="crank", axis=[0, 1, 0], damping=0.005)
    crank.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, -0.07, 0], size=[0.075, 0.003, 0],
                   euler=[math.pi / 2, 0, 0], rgba=[0.32, 0.33, 0.35, 1], mass=0.3, contype=0, conaffinity=0)
    for side, sign in ((0.085, 1), (-0.085, -1)):
        tip = [0, side, -sign * CRANK]
        crank.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, side, 0] + tip, size=[0.011, 0, 0],
                       rgba=SILVER, mass=0.15, contype=0, conaffinity=0)
        crank.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, side + sign * 0.05, -sign * CRANK],
                       size=[0.045, 0.045, 0.01], rgba=BLACK, mass=0.1, contype=0, conaffinity=0)
        crank.add_site(name="pedal_l" if side > 0 else "pedal_r", pos=[0, side + sign * 0.05, -sign * CRANK],
                       size=[0.02, 0, 0], rgba=[0, 0, 0, 0])
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="crank", name2="rear_spin")
    eq.data[:5] = [0.0, GEAR, 0.0, 0.0, 0.0]
    spec.add_actuator(name="pedal", target="rear_spin", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      ctrlrange=[-12, 12])
    st = spec.add_actuator(name="steer", target="steering", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                           ctrlrange=[-0.6, 0.6])
    st.set_to_position(kp=100, kv=12)
    spec.add_sensor(name="orientation", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                    objname="frame")
    spec.add_sensor(name="angular_velocity", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="imu")
    spec.add_sensor(name="velocity", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="imu")
    return spec
