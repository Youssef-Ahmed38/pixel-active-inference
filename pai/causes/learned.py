"""Learned arm episode likelihoods from the arm's own prediction errors.

The old analytic cause fits remain parameter estimators and optional features.
No event log, disturbance schedule, physical perturbation or teaching label is
accepted by extraction/inference. Labels enter fit() only. Separate episode and
prefix likelihoods support terminal diagnosis and optional learned online control.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import math
import numpy as np

from pai.causes.inference import CAUSES, PRIORS, CauseReport, StepEvidence, SurpriseMonitor, infer_cause

KNOWN = CAUSES[:-1]


def peak_fit(history, dt, window=20):
    if not history:
        raise ValueError('At least one own-evidence transition is required')
    peak = int(np.argmax([e.surprise for e in history]))
    lo = max(0, peak - window + 5)
    return infer_cause(history[lo:lo + window], dt)


def extract_features(history: list[StepEvidence], dt=.1, triggered=False):
    """Hand-designed evidence summaries; values/distributions are learned later.

    Whole-episode and contact-conditioned channels retain signed force and
    object/hand disagreement that the old interval fit can mistake for a push.
    Onset is searched from evidence only, never supplied to this function.
    """
    if not history:
        raise ValueError('Empty evidence')
    r = np.stack([e.residual for e in history]).astype(float)
    s = np.stack([e.sigma for e in history]).astype(float)
    if r.shape[1:] != (12,) or s.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('Twelve finite residual/sigma channels, sigma > 0 required')
    z = r / s
    holding = np.array([e.holding for e in history], bool)
    grasp = np.array([e.step == 'grasped' for e in history])
    contact = holding | grasp
    features = {'raw_triggered': float(triggered), 'raw_hold_fraction': float(holding.mean()),
                'raw_grasp_fraction': float(grasp.mean())}
    for i in range(12):
        features[f'raw_z{i}_rms'] = float(np.sqrt(np.mean(z[:, i] ** 2)))
        features[f'raw_z{i}_peak'] = float(z[np.argmax(np.abs(z[:, i])), i])
    for name, mask in [('hold', holding), ('grasp', grasp), ('free', ~contact)]:
        features[f'raw_{name}_present'] = float(mask.any())
        for axis in range(3):
            features[f'raw_{name}_force{axis}'] = float(np.mean(z[mask, 3 + axis])) if mask.any() else 0.
            rel = r[:, 6 + axis] - r[:, axis]
            noise = np.sqrt(s[:, 6 + axis] ** 2 + s[:, axis] ** 2)
            features[f'raw_{name}_relative{axis}'] = float(np.sqrt(np.mean((rel[mask] / noise[mask]) ** 2))) if mask.any() else 0.
        features[f'raw_{name}_hand'] = float(np.sqrt(np.mean(z[mask, :3] ** 2))) if mask.any() else 0.
    # Common visual displacement versus independent object/hand movement.
    scene = z[:, 9:12]
    obj = z[:, 6:9]
    common = (obj + scene) / 2
    features['raw_common_peak'] = float(np.linalg.norm(common, axis=1).max())
    features['raw_visual_disagreement'] = float(np.sqrt(np.mean((obj - scene) ** 2)))
    for axis in range(3):
        features[f'raw_scene{axis}_peak'] = float(r[np.argmax(np.abs(r[:, 9 + axis])), 9 + axis])
    # Analytic models are features/parameter readouts, not the new decision rule.
    report = peak_fit(history, dt)
    for i, cause in enumerate(CAUSES):
        features[f'fit_logp_{i}'] = float(np.log(max(report.posterior[cause], 1e-12)))
    prm = report.params
    features['fit_extra_mass'] = prm['heavier_object'].get('extra_mass_kg', 0.)
    features['fit_sag'] = prm['heavier_object'].get('sag_mm', 0.)
    features['fit_sink'] = prm['slippery_object'].get('sink_mm', 0.)
    features['fit_lost_weight'] = prm['slippery_object'].get('weight_lost_N', 0.)
    features['fit_push_force'] = float(np.linalg.norm(prm['push']['force_N']))
    features['fit_push_offset'] = float(np.linalg.norm(prm['push']['offset_mm']))
    features['fit_camera_offset'] = float(np.linalg.norm(prm['camera_shift']['camera_offset_mm']))
    features['fit_sliding'] = float(np.linalg.norm(prm['slippery_object'].get('slide_mm', [0, 0])))
    features['fit_triggered'] = float(triggered)
    features['fit_holding'] = float(holding.any())
    return features, report


@dataclass
class LearnedArmLikelihood:
    names: list
    raw_scale: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    feature_set: str = 'fused'
    unknown_scale: float = 4.
    unknown_prior: float = .03
    likelihood_family: str = 'gaussian'
    prior_kind: str = 'uniform'

    def __post_init__(self):
        self._precisions = np.linalg.inv(self.covariances)
        signs, self._logdets = np.linalg.slogdet(self.covariances)
        if not (signs > 0).all() or not np.isfinite(self._precisions).all():
            raise ValueError('Finite positive-definite class covariance required')
        if self.likelihood_family not in ('gaussian','student_t4') or self.prior_kind not in ('uniform','legacy'):
            raise ValueError('Unknown density family or prior')

    @classmethod
    def fit(cls, rows, labels, feature_set='fused', shrinkage=.9, unknown_scale=4., unknown_prior=.02, weights=None,
            likelihood_family='gaussian',prior_kind='uniform'):
        if feature_set not in ('raw', 'fits', 'fused'):
            raise ValueError(feature_set)
        names = sorted(k for k in rows[0] if feature_set == 'fused' or k.startswith('raw_' if feature_set == 'raw' else 'fit_'))
        x = np.asarray([[r[k] for k in names] for r in rows], float)
        labels = np.asarray(labels)
        if not np.isfinite(x).all() or len(x) != len(labels) or set(labels) != set(KNOWN):
            raise ValueError('Finite development evidence and all five labelled causes required')
        weights = np.ones(len(x)) if weights is None else np.asarray(weights, float)
        if weights.shape != (len(x),) or (weights <= 0).any() or not np.isfinite(weights).all():
            raise ValueError('Positive finite observation weights required')
        raw_scale = np.maximum(np.sqrt(np.average(x*x, axis=0, weights=weights)), 1e-6)
        y = np.arcsinh(x / raw_scale)
        center = np.average(y, axis=0, weights=weights)
        scale = np.maximum(np.sqrt(np.average((y-center)**2, axis=0, weights=weights)), .05)
        z = (y - center) / scale
        means, covariances = [], []
        for cause in KNOWN:
            part = z[labels == cause]
            if len(part) < 3:
                raise ValueError('At least three labelled development episodes per cause')
            w = weights[labels == cause]
            mean = np.average(part, axis=0, weights=w)
            means.append(mean)
            diff = part - mean
            cov = (diff.T * w) @ diff / w.sum()
            cov = ((1 - shrinkage) * cov + shrinkage * np.diag(np.maximum(np.diag(cov), .1))
                   + .1 * np.eye(len(names)))
            covariances.append(cov)
        return cls(names, raw_scale, center, scale, np.asarray(means), np.asarray(covariances),
                   feature_set, unknown_scale, unknown_prior,likelihood_family,prior_kind)

    def posterior(self, features):
        x = np.asarray([features[k] for k in self.names], float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite inference features')
        z = (np.arcsinh(x / self.raw_scale) - self.center) / self.scale
        d = len(z)
        ll = []
        priors=np.asarray([PRIORS[c] for c in KNOWN]) if self.prior_kind=='legacy' else np.ones(5)
        priors=priors/priors.sum()*(1-self.unknown_prior)
        for mean, precision, logdet,prior in zip(self.means, self._precisions, self._logdets,priors):
            delta = z - mean
            mahal=float(delta @ precision @ delta)
            if self.likelihood_family=='gaussian':
                density=-.5*(mahal+logdet+d*np.log(2*np.pi))
            else:
                nu=4.
                density=(math.lgamma((nu+d)/2)-math.lgamma(nu/2)-d/2*np.log(nu*np.pi)
                         -.5*logdet-(nu+d)/2*np.log1p(mahal/nu))
            ll.append(density+np.log(prior))
        nu = 3.
        q = float(z @ z) / self.unknown_scale**2
        ll.append(math.lgamma((nu+d)/2)-math.lgamma(nu/2)-d/2*np.log(nu*np.pi)
                  -d*np.log(self.unknown_scale)-(nu+d)/2*np.log1p(q/nu)+np.log(self.unknown_prior))
        p = np.exp(np.asarray(ll)-max(ll)); p /= p.sum()
        return dict(zip(CAUSES, map(float, p)))

    def infer(self, history, dt=.1, triggered=False):
        features, old_fit = extract_features(history, dt, triggered)
        posterior = self.posterior(features)
        params = dict(old_fit.params)
        # If the peak window precedes load bearing, obtain the mass estimate
        # from a window around the strongest OWN held residual instead.
        if max(posterior, key=posterior.get) == 'heavier_object' and not params['heavier_object']:
            candidates = [i for i, e in enumerate(history) if e.holding]
            if candidates:
                peak = max(candidates, key=lambda i: history[i].surprise)
                lo = max(0, peak - 15)
                params['heavier_object'] = infer_cause(history[lo:lo+20], dt).params['heavier_object']
        return CauseReport(history[0].t, history[-1].t, posterior, params)

    def to_dict(self):
        return {f.name: getattr(self,f.name).tolist() if isinstance(getattr(self,f.name),np.ndarray)
                else getattr(self,f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, record):
        d = dict(record)
        for k in ('raw_scale', 'center', 'scale', 'means', 'covariances'):
            d[k] = np.asarray(d[k], float)
        return cls(**d)


class LearnedEpisodeMonitor(SurpriseMonitor):
    """Learned terminal diagnosis and optionally prefix-trained online reports."""
    def __init__(self, *args, likelihood, online_likelihood=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.likelihood = likelihood
        self.online_likelihood = online_likelihood
        self.withheld_adaptations = []

    def add(self, ev, raw=None):
        report = super().add(ev, raw)
        if report is None or self.online_likelihood is None:
            return report
        learned = self.online_likelihood.infer(self.raw_history, self.dt, True)
        self.reports[-1] = learned
        # Keep the learned diagnosis intact, but do not call an adaptation that
        # requires physical estimates which own evidence cannot yet provide.
        required = {'heavier_object': ('extra_weight_N', 'sag_mm', 'extra_mass_kg'),
                    'camera_shift': ('onset', 'seen_jump')}.get(learned.best, ())
        if any(k not in learned.params.get(learned.best, {}) for k in required):
            self.withheld_adaptations.append({'t': ev.t, 'cause': learned.best,
                                             'reason': 'parameter readout unavailable'})
            return None
        return learned

    def episode_verdict(self):
        return self.likelihood.infer(self.raw_history, self.dt, bool(self.reports)) if self.raw_history else None


@contextmanager
def learned_episode_monitor(likelihood, online_likelihood=None):
    """Install the terminal monitor for one sequential slice-agent call.

    This scoped compatibility hook avoids edits to the shared arm/language
    files. It is process-local and is not intended for concurrent calls in one
    process. Without an online likelihood, adaptation uses the unchanged
    analytic reports; with one, prefix-trained reports drive adaptation.
    """
    from pai.agents import slice_agent
    original = slice_agent.SurpriseMonitor
    def factory(*args, **kwargs):
        return LearnedEpisodeMonitor(*args, likelihood=likelihood, online_likelihood=online_likelihood, **kwargs)
    slice_agent.SurpriseMonitor = factory
    try:
        yield
    finally:
        slice_agent.SurpriseMonitor = original
