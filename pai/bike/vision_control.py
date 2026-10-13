"""Hybrid visual balance: RGB attitude and externally measured forward speed."""
import math
import numpy as np
from pai.bike.env import BalanceController


class ImageBalanceController:
    def __init__(self, model, controller=None, rate_time_constant=.1):
        if not math.isfinite(rate_time_constant) or rate_time_constant <= 0:
            raise ValueError("positive finite rate smoothing constant required")
        self.model = model
        self.controller = controller or BalanceController()
        self.rate_time_constant = rate_time_constant
        self.reset()

    def reset(self):
        self.previous_time = self.previous_roll = None
        self.roll_rate = 0.0

    def action(self, image, time, speed, steer=0.0, lean=0.0):
        if not all(math.isfinite(v) for v in (time, speed, steer, lean)):
            raise ValueError("finite time, external speed and joint measurements required")
        if self.previous_time is not None and time <= self.previous_time:
            raise ValueError("image timestamps must increase")
        angles = np.asarray(self.model.predict(image))
        if angles.shape != (3,) or not np.all(np.isfinite(angles)):
            raise ValueError("finite image orientation required")
        roll, _, yaw = angles
        if self.previous_time is not None:
            dt = time - self.previous_time
            delta = (roll - self.previous_roll + np.pi) % (2 * np.pi) - np.pi
            alpha = dt / (self.rate_time_constant + dt)
            self.roll_rate += alpha * (delta / dt - self.roll_rate)
        self.previous_time, self.previous_roll = time, roll
        return self.controller.action(np.array([roll, self.roll_rate, speed, steer, lean, yaw]))
