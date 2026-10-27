"""Proprioceptive surprise, a locomotion event log, and fall-cause inference.

The idea is the one of ``pai/causes/inference.py`` (explain a surprise by comparing hidden-cause
hypotheses), applied to the walking body. The learned body model (``world_model.py``) predicts
the next proprioceptive state from the current one and the body's *own* motor command. What it
cannot predict is evidence for something outside the controller:

    push   an external force on the torso: the pelvis velocity changes by more than the body
           model predicts from the motor commands (unexplained impulse U, in m/s);
    slip   the ground gives way under a stance foot: a sole that is down slides sideways
           (sustained slip speed S, in m/s, from the sole sensors);
    gait   the body's own gait error: the fall develops as the body model predicts from the
           commands, with no unexplained impulse and no sliding stance foot.

Evidence is summarised over the last ``window`` control steps by two statistics (U, S). Their
distribution when nothing external happens is *calibrated* on falls of undisturbed bodies
(calibration seeds, never the test seeds), like ``pai.causes.inference.Calibration`` on clean
episodes. Hypotheses:

    gait:  log U ~ N(mu_U, sd_U),         log S ~ N(mu_S, sd_S)
    push:  log U ~ Broad (above normal),  log S ~ N(mu_S, sd_S)
    slip:  log S ~ Broad (above normal),  log U ~ 0.5 N(mu_U, sd_U) + 0.5 Broad
           (a slip also jolts the pelvis a little)

"Broad" is a uniform density on [mu, mu + BROAD_SPAN] in log units (an effect larger than usual,
of unknown size). Posterior = prior x likelihood, normalised (Bayesian model comparison).
Nothing here reads the simulator's push forces or friction: those are only in the event log,
which is the answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pai.locomotion.env import PROPRIO

CAUSES = ("gait", "push", "slip")
PRIORS = np.array([1 / 3, 1 / 3, 1 / 3])
V_HEAD = slice(6, 8)          # pelvis horizontal velocity inside the PROPRIO vector
BROAD_SPAN = 6.0              # log-units (a factor of ~400) above the normal median
IMPULSE_STEPS = 10            # residuals summed over 0.2 s (a push lasts 0.2 s)
COLLAPSE_STEPS = 0            # dropping the last 0.2 s was tried on dev seeds: 79% -> 59% (big pushes fell within it)
SLIP_SUSTAIN = 3              # a sliding stance foot must slide for 3 steps (60 ms)
EPS = 1e-3


@dataclass
class Calibration:
    mu: np.ndarray = field(default_factory=lambda: np.array([np.log(0.1), np.log(0.05)]))
    sd: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7]))
    n: int = 0

    @classmethod
    def fit(cls, features: np.ndarray) -> "Calibration":
        lf = np.log(np.asarray(features) + EPS)
        return cls(mu=lf.mean(0), sd=np.maximum(lf.std(0), 0.15), n=len(lf))

    def to_dict(self):
        return {"mu_log": self.mu.tolist(), "sd_log": self.sd.tolist(), "n": self.n,
                "median_U": float(np.exp(self.mu[0])), "median_S": float(np.exp(self.mu[1]))}


def _lognormal(lf, mu, sd):
    return -0.5 * ((lf - mu) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


def _broad(lf, mu):
    inside = (lf >= mu - 1.0) & (lf <= mu + BROAD_SPAN)
    return np.where(inside, -np.log(BROAD_SPAN + 1.0), -50.0)


def posterior(features: np.ndarray, cal: Calibration) -> np.ndarray:
    """(n, 2) features [U, S] -> (n, 3) posterior over CAUSES."""
    lf = np.log(np.atleast_2d(features) + EPS)
    lu, ls = lf[:, 0], lf[:, 1]
    nu, ns = _lognormal(lu, cal.mu[0], cal.sd[0]), _lognormal(ls, cal.mu[1], cal.sd[1])
    bu, bs = _broad(lu, cal.mu[0]), _broad(ls, cal.mu[1])
    gait = nu + ns
    push = bu + ns
    slip = bs + np.logaddexp(nu + np.log(0.5), bu + np.log(0.5))
    ll = np.stack([gait, push, slip], 1) + np.log(PRIORS)
    ll -= ll.max(1, keepdims=True)
    p = np.exp(ll)
    return p / p.sum(1, keepdims=True)


class SurpriseMonitor:
    """Per-body ring buffers of body-model residuals and stance-foot sliding.

    Call ``update`` after every env step with the transition the body made. ``surprise`` is the
    ensemble-mean squared prediction error in normalised units (per step); ``features`` returns
    the (U, S) evidence of the last ``window`` steps.
    """

    def __init__(self, body_model, n: int, window: int = 100, contact_height: float = 0.03,
                 device="cpu"):
        self.model = body_model
        self.n, self.window = n, window
        self.contact_height = contact_height
        self.device = torch.device(device)
        self.res = np.zeros((n, window, 2))          # unexplained pelvis velocity change (m/s)
        self.slide = np.zeros((n, window))           # sliding speed of the feet that are down
        self.sur = np.zeros((n, window))
        self.k = 0

    def reset(self, ids):
        ids = np.atleast_1d(ids)
        self.res[ids] = 0.0
        self.slide[ids] = 0.0
        self.sur[ids] = 0.0

    @torch.no_grad()
    def update(self, obs, action, next_obs, sole_height, sole_vel):
        """obs/next_obs: (n, OBS_DIM); sole_height (n, 2) above flat ground; sole_vel (n, 2, 3)."""
        s = torch.as_tensor(np.asarray(obs)[:, PROPRIO], dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(np.asarray(next_obs)[:, PROPRIO], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(action), dtype=torch.float32, device=self.device)
        pred = self.model.predict(s, a)
        sur, _ = self.model.surprise_and_disagreement(s, a, s2)
        res = (s2 - pred)[:, V_HEAD].cpu().numpy()
        down = sole_height < self.contact_height
        slide = np.max(np.linalg.norm(sole_vel[:, :, :2], axis=2) * down, axis=1)
        i = self.k % self.window
        self.res[:, i] = res
        self.slide[:, i] = slide
        self.sur[:, i] = sur.cpu().numpy()
        self.k += 1
        return self.sur[:, i]

    def _ordered(self, arr, ids):
        i = self.k % self.window
        return np.roll(arr[ids], -i, axis=1)        # oldest first

    def features(self, ids, exclude_last: int = 0) -> np.ndarray:
        """(U, S) over the window; ``exclude_last`` drops the final steps (the collapse itself,
        where every foot slides and the body model is far outside its experience)."""
        ids = np.atleast_1d(ids)
        res = self._ordered(self.res, ids)
        if exclude_last:
            res = res[:, :-exclude_last]
        csum = np.cumsum(res, axis=1)
        csum = np.concatenate([np.zeros((len(ids), 1, 2)), csum], 1)
        win = csum[:, IMPULSE_STEPS:] - csum[:, :-IMPULSE_STEPS]
        U = np.linalg.norm(win, axis=2).max(1)
        sl = self._ordered(self.slide, ids)
        if exclude_last:
            sl = sl[:, :-exclude_last]
        sustained = np.min(np.stack([sl[:, j:sl.shape[1] - SLIP_SUSTAIN + 1 + j]
                                     for j in range(SLIP_SUSTAIN)], 0), 0)
        S = sustained.max(1)
        return np.stack([U, S], 1)

    def recent_surprise(self, ids, steps: int = 10) -> np.ndarray:
        sur = self._ordered(self.sur, np.atleast_1d(ids))
        return sur[:, -steps:].mean(1)


class LocomotionEventLog:
    """Answer key for one body (index ``i`` of a batched env): contacts, slips, pushes, falls."""

    SLIP_SPEED = 0.3

    def __init__(self, i: int = 0):
        self.i = i
        self.events: list[dict] = []
        self._contact = np.array([True, True])
        self._slip = np.array([0, 0])

    def log(self, t, kind, **details):
        self.events.append({"t": round(float(t), 3), "type": kind, **details})

    def observe(self, t, sole_height, sole_vel, contact_height=0.03, fell=False):
        c = sole_height < contact_height
        for f, name in enumerate(("left_foot", "right_foot")):
            if c[f] and not self._contact[f]:
                self.log(t, "contact_start", a=name, b="floor")
            elif self._contact[f] and not c[f]:
                self.log(t, "contact_end", a=name, b="floor")
            speed = float(np.linalg.norm(sole_vel[f, :2]))
            self._slip[f] = self._slip[f] + 1 if (c[f] and speed > self.SLIP_SPEED) else 0
            if self._slip[f] == SLIP_SUSTAIN:
                self.log(t, "slip", foot=name, speed=round(speed, 3))
        self._contact = c
        if fell:
            self.log(t, "fall")

    def of_type(self, *kinds):
        return [e for e in self.events if e["type"] in kinds]
