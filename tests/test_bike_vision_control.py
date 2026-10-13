import numpy as np
import pytest
from pai.bike.vision_control import ImageBalanceController


class Orientation:
    def predict(self, image):
        return np.asarray(image)


class Capture:
    def action(self, sensation):
        return sensation


def test_visual_rate_wrap_timestamps_and_episode_reset():
    policy = ImageBalanceController(Orientation(), Capture(), rate_time_constant=.1)
    initial = policy.action([np.pi - .01, 0, .2], 0, 4)
    assert initial[1] == 0 and initial[2] == 4 and initial[5] == .2
    next_frame = policy.action([-np.pi + .01, 0, .3], .1, 3)
    assert next_frame[1] == pytest.approx(.1)  # wrapped .02 rad / .1 s, smoothed by .5
    with pytest.raises(ValueError):
        policy.action([0, 0, 0], .1, 3)
    policy.reset()
    reset = policy.action([0, 0, 0], 0, 3, steer=.1, lean=-.2)
    assert reset[1] == 0 and reset[3] == .1 and reset[4] == -.2


def test_invalid_visual_measurements_do_not_update_state():
    policy = ImageBalanceController(Orientation())
    with pytest.raises(ValueError):
        policy.action([np.nan, 0, 0], 0, 4)
    assert policy.previous_time is None
    with pytest.raises(ValueError):
        policy.action([0, 0, 0], 0, np.inf)
