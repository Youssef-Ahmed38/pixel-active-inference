"""Free bicycle dynamics: two rolling wheels, steering and a leaning torso.

This is an engineering baseline, not a humanoid or a video-trained policy.
Actuation is rear-wheel torque plus steering/torso angle servos. No force or
torque is applied directly to the floating frame to keep it upright.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np

from pai.cognition.contracts import EntityState, StateBelief, SkillCommand, SkillOutcome


XML = """
<mujoco model="bicycle">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint damping="0.03"/>
    <geom friction="1.2 0.005 0.001" condim="4" contype="1" conaffinity="2"/>
  </default>
  <worldbody>
    <light pos="0 -3 5"/>
    <geom name="ground" type="plane" size="100 100 0.1" rgba="0.6 0.6 0.6 1" contype="2" conaffinity="1"/>
    <body name="frame" pos="0 0 0.69">
      <freejoint name="root"/>
      <geom type="capsule" fromto="-0.45 0 0 0.45 0 0" size="0.035" mass="8" rgba="0.8 0.2 0.1 1"/>
      <geom type="capsule" fromto="-0.55 0 -0.35 0 0 0.15" size="0.025" mass="1"/>
      <geom type="capsule" fromto="0 0 0.15 0.5 0 -0.15" size="0.025" mass="1"/>
      <site name="imu" size="0.015"/>
      <body name="rear_wheel" pos="-0.55 0 -0.35">
        <joint name="rear_spin" axis="0 1 0"/>
        <geom type="cylinder" size="0.32 0.02" euler="1.57079632679 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="fork" pos="0.50 0 -0.10">
        <joint name="steering" axis="-0.3 0 0.953939" range="-0.65 0.65" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.05 0 -0.25" size="0.025" mass="1"/>
        <body name="front_wheel" pos="0.05 0 -0.25">
          <joint name="front_spin" axis="0 1 0"/>
          <geom type="cylinder" size="0.32 0.02" euler="1.57079632679 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
        </body>
      </body>
      <body name="rider" pos="-0.05 0 0.1">
        <joint name="torso_lean" axis="1 0 0" range="-0.3 0.3" damping="5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.11" mass="45" rgba="0.2 0.4 0.8 1"/>
        <geom type="sphere" pos="0 0 0.75" size="0.10" mass="4"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="pedal" joint="rear_spin" ctrlrange="-12 12"/>
    <position name="steer" joint="steering" kp="100" kv="12" ctrlrange="-0.6 0.6"/>
    <position name="lean" joint="torso_lean" kp="100" kv="15" ctrlrange="-0.25 0.25"/>
  </actuator>
  <sensor>
    <framequat name="orientation" objtype="xbody" objname="frame"/>
    <gyro name="angular_velocity" site="imu"/>
    <velocimeter name="velocity" site="imu"/>
  </sensor>
</mujoco>
"""


class BicycleEnv:
    domain = "bike"

    def __init__(self, distance=10.0, control_dt=.02, max_seconds=10.0):
        if distance <= 0 or control_dt < .002 or max_seconds <= 0:
            raise ValueError("positive distance, duration and simulation step required")
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.distance = distance
        self.substeps = max(1, round(control_dt / self.model.opt.timestep))
        self.control_dt = self.substeps * self.model.opt.timestep
        self.max_seconds = max_seconds
        self.joints = {name: self.model.joint(name) for name in
                       ("rear_spin", "front_spin", "steering", "torso_lean")}
        self.steps = 0
        self.events = []
        self.reset()

    def reset(self, seed=0, speed=4.0, roll=None):
        if speed < 0:
            raise ValueError("initial speed must be nonnegative")
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        angle = float(rng.uniform(-.03, .03)) if roll is None else float(roll)
        self.data.qpos[3:7] = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
        self.data.qvel[0] = speed
        for name in ("rear_spin", "front_spin"):
            self.data.qvel[self.joints[name].dofadr[0]] = speed / .32
        self.steps = 0
        self.events = []
        self.origin = self.data.qpos[:2].copy()
        mujoco.mj_forward(self.model, self.data)
        return self.sensation()

    def sensation(self):
        w, x, y, z = self.data.sensor("orientation").data
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, self.data.sensor("angular_velocity").data[0],
                         self.data.sensor("velocity").data[0],
                         self.data.qpos[self.joints["steering"].qposadr[0]],
                         self.data.qpos[self.joints["torso_lean"].qposadr[0]], yaw], dtype=float)

    @property
    def fallen(self):
        return bool(abs(self.sensation()[0]) > .7 or self.data.qpos[2] < .40)

    @property
    def travelled(self):
        # Forward displacement, not wheel rotation or distance travelled in circles.
        return float(self.data.qpos[0] - self.origin[0])

    @property
    def done(self):
        return self.fallen or self.travelled >= self.distance or self.data.time >= self.max_seconds

    def observe(self):
        sensation = self.sensation()
        relations = set()
        if self.fallen:
            relations.add(("fallen", "rider", None))
        elif self.travelled >= self.distance:
            relations.add(("distance_reached", "rider", None))
        return StateBelief(self.domain, self.steps,
            (EntityState("rider", "cyclist", {"roll": sensation[0], "roll_rate": sensation[1],
                "speed": sensation[2], "steer": sensation[3], "lean": sensation[4], "yaw": sensation[5],
                "distance": self.travelled}),), frozenset(relations))

    def available_skills(self):
        return (SkillCommand("ride"),)

    def execute(self, command):
        if command.name != "ride" or len(command.parameters) != 3:
            raise ValueError("ride requires pedal torque, steering angle and torso lean")
        action = np.asarray(command.parameters, dtype=float)
        if not np.all(np.isfinite(action)):
            raise ValueError("ride controls must be finite")
        if self.done:
            return SkillOutcome(command, False, True, 0)
        self.data.ctrl[:] = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.fallen and not any(e["type"] == "fall" for e in self.events):
            self.events.append({"type": "fall", "step": self.steps, "roll": float(self.sensation()[0])})
        return SkillOutcome(command, True, self.fallen, self.control_dt)


@dataclass
class BalanceController:
    roll_gain: float = 2.0
    rate_gain: float = .8
    yaw_gain: float = .15
    target_speed: float = 4.0

    def action(self, sensation):
        roll, rate, speed, steer, lean, yaw = sensation
        torque = np.clip(3 * (self.target_speed - speed), -12, 12)
        steering = np.clip(-self.roll_gain * roll - self.rate_gain * rate - self.yaw_gain * yaw, -.6, .6)
        return np.array([torque, steering, 0.0])

    def __call__(self, state, adapter):
        if adapter.done:
            return None
        return SkillCommand("ride", parameters=tuple(map(float, self.action(adapter.sensation()))))
