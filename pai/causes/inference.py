"""Explicit Bayesian hidden-cause inference: the "teacher" the HRM thinker later learns to imitate.

Input: per planner step, the world model's predicted next state and wrist force (mean and variance)
and what was actually observed. Evidence channels: gripper position (3, m) and wrist force (3, N). The standardised prediction error z = (observed - predicted) / sigma is the
surprise signal. When it spikes, competing explanations are compared on the recent window:

    none            errors are noise:                    r ~ N(0, sigma^2)
    push            an external force on the arm: a constant offset u on all six channels (force and
                    the displacement it causes) on an interval [t0, t1)
    heavier_object  a held object weighs more than expected: while holding, a constant extra pull c
                    on the wrist force z (c >= 0) and a constant gripper sag s (s >= 0)
    unknown         none of the above: errors explained only as much broader noise (a catch-all)

Each hypothesis is scored by its maximised log-likelihood minus a BIC complexity penalty,
0.5 * n_params * log(n), an approximation of the log model evidence. The posterior over causes is
then proportional to prior * exp(evidence): Bayesian model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CAUSES = ("none", "push", "heavier_object", "unknown")
PRIORS = {"none": 0.78, "push": 0.1, "heavier_object": 0.1, "unknown": 0.02}
UNKNOWN_SCALE = 4.0  # the catch-all explains errors as noise UNKNOWN_SCALE times larger than expected
N_CH = 6               # gripper position (3, m) + wrist force (3, N)
SIGMA_FLOOR = np.array([0.0015] * 3 + [0.3] * 3)  # minimum noise per channel: 1.5 mm, 0.3 N
SURPRISE_THRESHOLD = 6.0  # mean 0.5*|z|^2 over 3 steps that triggers cause inference


@dataclass
class Calibration:
    """How wrong the world model really is when nothing is disturbed, measured on clean calibration
    episodes (never the evaluation episodes).

    floor: per-channel noise added to the model's own variance, *per task step* (context-dependent
    precision): grasping produces large, poorly predictable contact forces, while carrying is almost
    noise-free. One floor for all steps would be set by the noisiest step and hide a heavier object
    while carrying. `floor` is the fallback for steps without enough calibration data.
    threshold: surprise level that counts as a spike."""
    floor: np.ndarray = field(default_factory=lambda: SIGMA_FLOOR.copy())
    threshold: float = SURPRISE_THRESHOLD
    step_floors: dict[str, np.ndarray] = field(default_factory=dict)

    def floor_for(self, step: str | None) -> np.ndarray:
        return self.step_floors.get(step, self.floor) if step is not None else self.floor


@dataclass
class StepEvidence:
    t: int
    residual: np.ndarray   # (6,) observed - predicted: gripper position (m), wrist force (N)
    sigma: np.ndarray      # (6,) std: model uncertainty combined with the calibrated floor
    action: np.ndarray     # (A,) the executed action
    holding: bool          # the agent believed it was holding the object

    @property
    def surprise(self) -> float:
        z = self.residual / self.sigma
        return float(0.5 * (z**2).sum())


def combined_sigma(model_sigma: np.ndarray, calib: Calibration, step: str | None = None) -> np.ndarray:
    return np.sqrt(model_sigma**2 + calib.floor_for(step) ** 2)


def _floor(res: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Noise the model does not account for: residual power beyond its own predicted variance."""
    return np.sqrt(np.maximum((res**2).mean(0) - (sig**2).mean(0), SIGMA_FLOOR**2))


def calibrate(clean_residuals: np.ndarray, model_sigmas: np.ndarray, steps: list[str] | None = None,
              quantile: float = 0.995, margin: float = 1.5, min_samples: int = 20) -> Calibration:
    """clean_residuals, model_sigmas (n, N_CH) from undisturbed episodes; steps: the task step of each
    sample. Steps with at least min_samples samples get their own floor."""
    floor = _floor(clean_residuals, model_sigmas)
    step_floors = {}
    if steps is not None:
        steps = np.asarray(steps)
        for name in np.unique(steps):
            m = steps == name
            if m.sum() >= min_samples:
                step_floors[str(name)] = _floor(clean_residuals[m], model_sigmas[m])
        per = np.stack([step_floors.get(str(n), floor) for n in steps])
    else:
        per = np.broadcast_to(floor, clean_residuals.shape)
    sig = np.sqrt(model_sigmas**2 + per**2)
    surprise = 0.5 * ((clean_residuals / sig) ** 2).sum(-1)
    level = np.convolve(surprise, np.ones(3) / 3, mode="valid")  # the monitor averages 3 steps
    return Calibration(floor=floor, threshold=float(margin * np.quantile(level, quantile)), step_floors=step_floors)


@dataclass
class CauseReport:
    t_start: int
    t_end: int
    posterior: dict[str, float]
    params: dict[str, dict] = field(default_factory=dict)

    @property
    def best(self) -> str:
        return max(self.posterior, key=self.posterior.get)


def _loglik(r: np.ndarray, s: np.ndarray) -> float:
    return float(-0.5 * (((r / s) ** 2) + 2 * np.log(s) + np.log(2 * np.pi)).sum())


def infer_cause(window: list[StepEvidence], dt: float) -> CauseReport:
    R = np.stack([e.residual for e in window])          # (n, N_CH)
    S = np.stack([e.sigma for e in window])  # already includes the calibrated floor
    n = len(window)
    logn = np.log(R.size)
    evidence, params = {}, {}

    evidence["none"] = _loglik(R, S)
    params["none"] = {}

    # push: constant offset u on a contiguous interval; search the interval, u in closed form
    best = (-np.inf, None)
    for t0 in range(n):
        for t1 in range(t0 + 1, n + 1):
            u = R[t0:t1].mean(0)
            resid = R.copy()
            resid[t0:t1] -= u
            ll = _loglik(resid, S)
            if ll > best[0]:
                best = (ll, (t0, t1, u))
    t0, t1, u = best[1]
    evidence["push"] = best[0] - 0.5 * (N_CH + 2) * logn  # offset per channel + the interval
    params["push"] = {"onset": window[t0].t, "end": window[t1 - 1].t,
                      "force_N": u[3:].round(1).tolist(), "offset_mm": (1000 * u[:3]).round(1).tolist()}

    # heavier object: while holding, extra pull on the wrist (force z up) and a sag of the gripper (z down)
    hold = np.array([e.holding for e in window])
    if hold.sum() >= 2:
        h = hold.astype(float)
        w_f, w_z = 1.0 / S[:, 5] ** 2, 1.0 / S[:, 2] ** 2
        c = max(0.0, float((w_f * h * R[:, 5]).sum() / (w_f * h).sum()))      # extra weight (N), >= 0
        s = max(0.0, float((w_z * h * -R[:, 2]).sum() / (w_z * h).sum()))     # sag (m), >= 0
        resid = R.copy()
        resid[:, 5] -= c * h
        resid[:, 2] += s * h
        evidence["heavier_object"] = _loglik(resid, S) - 0.5 * 2 * logn
        params["heavier_object"] = {"extra_weight_N": round(c, 2), "extra_mass_kg": round(c / 9.81, 2),
                                    "sag_mm": round(1000 * s, 2)}
    else:
        evidence["heavier_object"] = -np.inf  # nothing held: this cause cannot explain anything
        params["heavier_object"] = {}

    # unknown: a catch-all with much broader noise and a small prior. It wins only when no specific
    # cause fits: the errors are too large or have a shape none of the known causes produces.
    evidence["unknown"] = _loglik(R, S * UNKNOWN_SCALE)
    params["unknown"] = {"residual_rms": np.sqrt((R / S) ** 2).mean().round(2).item()}

    logpost = {c: np.log(PRIORS[c]) + evidence[c] for c in CAUSES}
    m = max(logpost.values())
    unnorm = {c: np.exp(v - m) for c, v in logpost.items()}
    z = sum(unnorm.values())
    return CauseReport(window[0].t, window[-1].t, {c: float(v / z) for c, v in unnorm.items()}, params)


class SurpriseMonitor:
    """Tracks surprise online and runs cause inference when it spikes (a noradrenaline-like trigger)."""

    def __init__(self, dt: float, window: int = 20, cooldown: int = 15, calibration: Calibration | None = None):
        self.dt, self.window, self.cooldown = dt, window, cooldown
        self.calibration = calibration or Calibration()
        self.history: list[StepEvidence] = []
        self.reports: list[CauseReport] = []
        self._last_trigger = -10**9

    def add(self, ev: StepEvidence) -> CauseReport | None:
        self.history.append(ev)
        recent = self.history[-3:]
        level = np.mean([e.surprise for e in recent])
        if level > self.calibration.threshold and ev.t - self._last_trigger > self.cooldown:
            self._last_trigger = ev.t
            report = infer_cause(self.history[-self.window :], self.dt)
            self.reports.append(report)
            return report
        return None

    def episode_verdict(self) -> CauseReport | None:
        """The explanation for the episode: re-run inference on the most surprising window."""
        if not self.reports:
            return None
        peak = max(range(len(self.history)), key=lambda i: self.history[i].surprise)
        lo = max(0, peak - self.window + 5)
        return infer_cause(self.history[lo : lo + self.window], self.dt)
