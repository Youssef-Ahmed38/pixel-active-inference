"""Explicit Bayesian hidden-cause inference: the "teacher" the HRM thinker later learns to imitate.

Input: per planner step, the world model's predicted next state and wrist force (mean and variance)
and what was actually observed. Evidence channels (CHANNELS): gripper position (3, m), wrist force
(3, N), position of the manipulated object (3, m) and mean position of the other objects, "the
scene" (3, m). The standardised prediction error z = (observed - predicted) / sigma is the surprise
signal. When it spikes, competing explanations are compared on the recent window:

    none             errors are noise:                    r ~ N(0, sigma^2)
    push             an external force on the arm: a constant offset on the gripper, force and
                     object channels on an interval [t0, t1) (the rest of the scene is not pushed)
    heavier_object   a held object weighs more than expected: while holding, a constant extra pull c
                     on the wrist force z (c >= 0) and a sag s of hand and object (s >= 0)
    slippery_object  the object slides in the hand: while holding, on an interval, the object sinks and
                     the wrist feels less of its weight (both downward, gravity-driven), while the
                     gripper itself moves as predicted
    camera_shift     the camera was bumped: at one step t0, everything seen (object and scene) jumps
                     by the same vector while the gripper, felt through the body, does not
    unknown          none of the above: errors explained only as much broader noise (a catch-all)

Each hypothesis is scored by its maximised log-likelihood minus a BIC complexity penalty,
0.5 * n_params * log(n), an approximation of the log model evidence. The posterior over causes is
then proportional to prior * exp(evidence): Bayesian model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CAUSES = ("none", "push", "heavier_object", "slippery_object", "camera_shift", "unknown")
PRIORS = {"none": 0.72, "push": 0.065, "heavier_object": 0.065, "slippery_object": 0.065,
          "camera_shift": 0.065, "unknown": 0.02}
UNKNOWN_SCALE = 4.0  # the catch-all explains errors as noise UNKNOWN_SCALE times larger than expected
CHANNELS = ("gripper_x", "gripper_y", "gripper_z", "force_x", "force_y", "force_z",
            "object_x", "object_y", "object_z", "scene_x", "scene_y", "scene_z")
N_CH = len(CHANNELS)
G, F, O, S = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
SIGMA_FLOOR = np.array([0.0015] * 3 + [0.3] * 3 + [0.0015] * 6)  # minimum noise per channel: 1.5 mm, 0.3 N
SURPRISE_THRESHOLD = float(N_CH)  # mean 0.5*|z|^2 over 3 steps (twice its noise-only mean) that triggers inference


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
    residual: np.ndarray   # (N_CH,) observed - predicted, channels as in CHANNELS
    sigma: np.ndarray      # (N_CH,) std: model uncertainty combined with the calibrated floor
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
    Sg = np.stack([e.sigma for e in window])  # already includes the calibrated floor
    n = len(window)
    logn = np.log(R.size)
    hold = np.array([e.holding for e in window], float)
    evidence, params = {}, {}

    evidence["none"] = _loglik(R, Sg)
    params["none"] = {}

    # push: a constant offset u on gripper, force and object channels on a contiguous interval;
    # search the interval, u in closed form (precision-weighted mean)
    arm = slice(0, 9)
    best = (-np.inf, None)
    for t0 in range(n):
        for t1 in range(t0 + 1, n + 1):
            w = 1.0 / Sg[t0:t1, arm] ** 2
            u = (w * R[t0:t1, arm]).sum(0) / w.sum(0)
            resid = R.copy()
            resid[t0:t1, arm] -= u
            ll = _loglik(resid, Sg)
            if ll > best[0]:
                best = (ll, (t0, t1, u))
    t0, t1, u = best[1]
    evidence["push"] = best[0] - 0.5 * (9 + 2) * logn  # offset per channel + the interval
    params["push"] = {"onset": window[t0].t, "end": window[t1 - 1].t,
                      "force_N": u[F].round(1).tolist(), "offset_mm": (1000 * u[G]).round(1).tolist()}

    # heavier object: while holding, extra pull on the wrist (force z up) and a sag of hand and object
    if hold.sum() >= 2:
        w_f = 1.0 / Sg[:, 5] ** 2
        w_z = 1.0 / Sg[:, 2] ** 2 + 1.0 / Sg[:, 8] ** 2
        c = max(0.0, float((w_f * hold * R[:, 5]).sum() / (w_f * hold).sum()))  # extra weight (N), >= 0
        sag = -(R[:, 2] / Sg[:, 2] ** 2 + R[:, 8] / Sg[:, 8] ** 2)
        s_ = max(0.0, float((hold * sag).sum() / (w_z * hold).sum()))           # sag (m), >= 0
        resid = R.copy()
        resid[:, 5] -= c * hold
        resid[:, 2] += s_ * hold
        resid[:, 8] += s_ * hold
        evidence["heavier_object"] = _loglik(resid, Sg) - 0.5 * 2 * logn
        params["heavier_object"] = {"extra_weight_N": round(c, 2), "extra_mass_kg": round(c / 9.81, 3),
                                    "sag_mm": round(1000 * s_, 2)}
    else:
        evidence["heavier_object"] = -np.inf  # nothing held: this cause cannot explain anything
        params["heavier_object"] = {}

    # slippery object: while holding, on an interval [t0, t1), the object sinks in the hand and the
    # wrist carries less of its weight, while the gripper itself moves as predicted. Slipping is
    # driven by gravity, so both effects point down: object z <= 0 and force z <= 0 (the pull of the
    # object on the wrist is lost). Sideways errors cannot be a slip; a push produces those.
    ch = np.array([5, 8])  # force z, object z
    best = (-np.inf, None)
    if hold.sum() >= 2:
        steps = np.arange(n)
        for t0 in range(n):
            for t1 in range(t0 + 1, n + 1):
                m = hold * ((steps >= t0) & (steps < t1))
                if m.sum() < 1:
                    continue
                w = m[:, None] / Sg[:, ch] ** 2
                v = np.minimum(0.0, (w * R[:, ch]).sum(0) / w.sum(0))
                resid = R.copy()
                resid[:, ch] -= m[:, None] * v
                ll = _loglik(resid, Sg)
                if ll > best[0]:
                    best = (ll, (t0, v))
    if best[1] is not None:
        t0, v = best[1]
        evidence["slippery_object"] = best[0] - 0.5 * (2 + 2) * logn
        params["slippery_object"] = {"onset": window[t0].t, "sink_mm": round(-1000 * float(v[1]), 1),
                                     "weight_lost_N": round(-float(v[0]), 2)}
    else:
        evidence["slippery_object"] = -np.inf
        params["slippery_object"] = {}

    # camera shift: at one step everything seen (object and scene) jumps by the same vector d,
    # while the gripper, sensed through the body, does not
    best = (-np.inf, None)
    for t0 in range(n):
        w = 1.0 / Sg[t0, 6:12].reshape(2, 3) ** 2
        d = (w * R[t0, 6:12].reshape(2, 3)).sum(0) / w.sum(0)
        resid = R.copy()
        resid[t0, 6:12] -= np.r_[d, d]
        ll = _loglik(resid, Sg)
        if ll > best[0]:
            best = (ll, (t0, d))
    t0, d = best[1]
    evidence["camera_shift"] = best[0] - 0.5 * (3 + 1) * logn
    # a camera moved by +delta makes the world appear displaced by -delta
    params["camera_shift"] = {"onset": window[t0].t, "camera_offset_mm": (-1000 * d).round(1).tolist(),
                              "seen_jump": d.round(4).tolist()}

    # unknown: a catch-all with much broader noise and a small prior. It wins only when no specific
    # cause fits: the errors are too large or have a shape none of the known causes produces.
    evidence["unknown"] = _loglik(R, Sg * UNKNOWN_SCALE)
    params["unknown"] = {"residual_rms": np.sqrt((R / Sg) ** 2).mean().round(2).item()}

    logpost = {c: np.log(PRIORS[c]) + evidence[c] for c in CAUSES}
    m = max(logpost.values())
    unnorm = {c: np.exp(v - m) for c, v in logpost.items()}
    z = sum(unnorm.values())
    return CauseReport(window[0].t, window[-1].t, {c: float(v / z) for c, v in unnorm.items()}, params)


class SurpriseMonitor:
    """Tracks surprise online and runs cause inference when it spikes (a noradrenaline-like trigger).

    Two streams of evidence are kept. `add(ev, raw)`: `ev` is the prediction error after the agent's
    adaptations (what it still cannot explain), which decides *when* to think; `raw` is the error
    before them, which is what explanations are fitted to. Otherwise an adapted agent would explain
    the remainder of an effect it has already accounted for (e.g. call the leftover of an expected
    extra weight a push). A trigger is followed by a second look `followup` steps later, when the
    aftermath of the event is visible too (a camera jump is over at once; a push lasts).
    """

    def __init__(self, dt: float, window: int = 20, cooldown: int = 15, calibration: Calibration | None = None,
                 followup: int = 3):
        self.dt, self.window, self.cooldown, self.followup = dt, window, cooldown, followup
        self.calibration = calibration or Calibration()
        self.history: list[StepEvidence] = []
        self.raw_history: list[StepEvidence] = []
        self.reports: list[CauseReport] = []
        self._last_trigger = -10**9
        self._pending: int | None = None

    def add(self, ev: StepEvidence, raw: StepEvidence | None = None) -> CauseReport | None:
        self.history.append(ev)
        self.raw_history.append(raw if raw is not None else ev)
        recent = self.history[-3:]
        level = np.mean([e.surprise for e in recent])
        due = self._pending is not None and ev.t >= self._pending
        if level > self.calibration.threshold and ev.t - self._last_trigger > self.cooldown:
            self._last_trigger = ev.t
            self._pending = ev.t + self.followup
        elif due:
            self._pending = None
        else:
            return None
        report = infer_cause(self.raw_history[-self.window :], self.dt)
        self.reports.append(report)
        return report

    def episode_verdict(self) -> CauseReport | None:
        """The explanation for the episode: re-run inference on the most surprising window."""
        if not self.reports:
            return None
        hist = self.raw_history
        peak = max(range(len(hist)), key=lambda i: hist[i].surprise)
        lo = max(0, peak - self.window + 5)
        return infer_cause(hist[lo : lo + self.window], self.dt)
