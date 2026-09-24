"""Real agent episodes as a thinker task: learn the Bayesian teacher's verdict from the evidence.

Source: `results/<tag>_evidence.npz` written by the evaluation script (the contract):
    z (S, C) standardised prediction errors, holding (S,), phase (S,), episode (S,) -- per planner step
    labels (E,) true cause, teacher (E, K) posterior, causes (K,), onset (E,) -- per episode

Input window. The teacher (`SurpriseMonitor.episode_verdict`) judges an episode on one window of
20 steps around its most surprising step: history[peak - 15 : peak + 5]. The student gets the same
window (recomputed from z, no label or onset needed), so the distillation target is a function of
what it sees; a whole padded episode would add ~40 steps of mostly noise to 60-ish training
episodes. `window`/`after` make it longer if the thinker should look further (e.g. window=64).

Features per step: asinh(z) (monotone, keeps sign and order, tames the heavy tails of real
surprise spikes), holding, one-hot phase (n_phase slots, clipped; -1 = no subgoal = all zeros).
Context (added to every token): log(1 + peak surprise) and the fraction of the window that is
inside the episode.

Augmentation (training only): the window start is jittered by up to `jitter` steps, always
keeping the peak in view. It is honest: the window position is computed from z alone, never from
the label or the true onset, and validation/test always use the teacher's canonical window.

Split: episode-level, stratified by true label, seeded; windows never mix episodes, so there is no
step-level leakage between train, val and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pai.causes.inference import CAUSES, CHANNELS, N_CH, F, O, S, StepEvidence, infer_cause

SPLITS = ("train", "val", "test")


@dataclass
class RealConfig:
    window: int = 20        # steps seen by the thinker (teacher: 20)
    after: int = 5          # steps kept after the peak (teacher: 5)
    jitter: int = 3         # max random shift of the training window start
    n_phase: int = 10       # phase one-hot slots
    val_frac: float = 0.2
    test_frac: float = 0.2

    @property
    def horizon(self) -> int:  # what build_model / the encoder call the sequence length
        return self.window

    @classmethod
    def from_dict(cls, d: dict) -> "RealConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_evidence(path: str | Path) -> dict:
    """Read one evidence file into plain numpy arrays (strings as str)."""
    try:
        f = np.load(path, allow_pickle=False)
    except ValueError:  # string arrays saved with dtype=object need pickle; these are our own results
        f = np.load(path, allow_pickle=True)
    d = {k: f[k] for k in f.files}
    for k in ("labels", "causes", "channels"):
        if k in d:
            d[k] = np.asarray([str(s) for s in d[k]])
    return d


def split_episodes(labels: np.ndarray, seed: int, val_frac: float, test_frac: float) -> dict[str, np.ndarray]:
    """Episode-level split, stratified by label so rare causes appear in every split when possible."""
    rng = np.random.default_rng(seed)
    out = {s: [] for s in SPLITS}
    for lab in sorted(set(labels.tolist())):
        idx = rng.permutation(np.flatnonzero(labels == lab))
        n_test, n_val = int(round(len(idx) * test_frac)), int(round(len(idx) * val_frac))
        out["test"] += idx[:n_test].tolist()
        out["val"] += idx[n_test:n_test + n_val].tolist()
        out["train"] += idx[n_test + n_val:].tolist()
    return {s: np.sort(np.array(v, dtype=np.int64)) for s, v in out.items()}


class RealEvidence:
    """All episodes of one or more evidence files, as padded per-episode feature tensors on `device`.

    Has the task interface `train_thinker`/`build_model` expect (n_channels, ctx_dim, n_causes,
    cfg.horizon, sample): `sample` draws jittered windows from the training split."""

    ctx_dim = 2

    def __init__(self, paths, cfg: RealConfig | None = None, seed: int = 0, device: torch.device | str = "cpu"):
        self.cfg, self.device = cfg or RealConfig(), torch.device(device)
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        files = [load_evidence(p) for p in paths]
        self.causes = tuple(files[0]["causes"].tolist())
        self.channels = tuple(files[0]["channels"].tolist()) if "channels" in files[0] else ()
        for p, f in zip(paths, files):
            if tuple(f["causes"].tolist()) != self.causes or f["z"].shape[1] != files[0]["z"].shape[1]:
                raise ValueError(f"{p}: causes/channels differ from {paths[0]}")
        self.paths = [str(p) for p in paths]

        feats, lengths, peaks, peak_s, labels, teacher, onset, source = [], [], [], [], [], [], [], []
        for fi, f in enumerate(files):
            ep = np.asarray(f["episode"])
            for e in range(len(f["labels"])):
                sel = np.flatnonzero(ep == e)
                if len(sel) == 0:
                    raise ValueError(f"{paths[fi]}: episode {e} has no steps")
                z = np.asarray(f["z"][sel], dtype=np.float32)
                surprise = 0.5 * (z ** 2).sum(-1)
                feats.append(self._features(z, np.asarray(f["holding"][sel]), np.asarray(f["phase"][sel])))
                lengths.append(len(sel))
                peaks.append(int(surprise.argmax()))
                peak_s.append(float(surprise.max()))
            labels += f["labels"].tolist()
            teacher.append(np.asarray(f["teacher"], dtype=np.float32))
            onset.append(np.asarray(f["onset"], dtype=np.int64))
            source += [fi] * len(f["labels"])

        self.labels = np.array(labels)  # raw true-cause names, may include causes the teacher lacks
        # True cause as a class index; a cause missing from the teacher's library counts as "unknown".
        fallback = self.causes.index("unknown") if "unknown" in self.causes else -1
        y = np.array([self.causes.index(l) if l in self.causes else fallback for l in labels], dtype=np.int64)
        if (y < 0).any():
            raise ValueError(f"labels {sorted(set(self.labels[y < 0]))} not in causes {self.causes} and no 'unknown'")
        self.n_episodes, t_max, self.n_channels = len(labels), max(lengths), feats[0].shape[1]
        pad = np.zeros((self.n_episodes, t_max, self.n_channels), dtype=np.float32)
        for i, x in enumerate(feats):
            pad[i, : len(x)] = x
        dev = self.device
        self.feats = torch.tensor(pad, device=dev)
        self.length = torch.tensor(lengths, dtype=torch.long, device=dev)
        self.peak = torch.tensor(peaks, dtype=torch.long, device=dev)
        self.log_peak = torch.tensor(np.log1p(peak_s), dtype=torch.float32, device=dev)
        self.teacher = torch.tensor(np.concatenate(teacher), device=dev)
        self.teacher = self.teacher / self.teacher.sum(-1, keepdim=True)
        self.y = torch.tensor(y, device=dev)
        self.onset = np.concatenate(onset)
        self.source = np.array(source)
        # The teacher's window: history[lo : lo + window], lo = max(0, peak - window + after).
        self.start = (self.peak - self.cfg.window + self.cfg.after).clamp_min(0)
        self.set_split(seed)

    def _features(self, z: np.ndarray, holding: np.ndarray, phase: np.ndarray) -> np.ndarray:
        onehot = np.zeros((len(z), self.cfg.n_phase), dtype=np.float32)
        ok = phase >= 0
        onehot[np.flatnonzero(ok), np.clip(phase[ok], 0, self.cfg.n_phase - 1)] = 1.0
        return np.concatenate([np.arcsinh(z), holding.astype(np.float32)[:, None], onehot], -1)

    @property
    def n_causes(self) -> int:
        return len(self.causes)

    def set_split(self, seed: int) -> None:
        self.split_seed = seed
        idx = split_episodes(self.labels, seed, self.cfg.val_frac, self.cfg.test_frac)
        self.splits = {s: torch.tensor(v, device=self.device) for s, v in idx.items()}

    def _windows(self, ep: torch.Tensor, start: torch.Tensor) -> dict:
        w = self.cfg.window
        pos = start[:, None] + torch.arange(w, device=self.device)[None]  # [B, W]
        mask = (pos < self.length[ep, None]).float()
        x = self.feats[ep[:, None], pos.clamp(max=self.feats.shape[1] - 1)] * mask[..., None]
        ctx = torch.stack([self.log_peak[ep], mask.mean(-1)], -1)
        return {"x": x, "mask": mask, "ctx": ctx, "teacher": self.teacher[ep], "y": self.y[ep], "episode": ep}

    def sample(self, n: int, gen: torch.Generator | None = None) -> dict:
        """Training batch: random training episodes (with replacement), jittered windows."""
        train = self.splits["train"]
        ep = train[torch.randint(0, len(train), (n,), device=self.device, generator=gen)]
        j = self.cfg.jitter
        shift = torch.randint(-j, j + 1, (n,), device=self.device, generator=gen) if j > 0 else 0
        lo = (self.peak[ep] - self.cfg.window + 1).clamp_min(0)  # the peak must stay in view
        start = torch.minimum(torch.maximum(self.start[ep] + shift, lo), self.peak[ep])
        return self._windows(ep, start)

    def split(self, name: str) -> dict:
        """Every episode of a split, on the teacher's canonical window (no augmentation)."""
        ep = self.splits[name]
        return self._windows(ep, self.start[ep])

    def summary(self) -> dict:
        return {"files": self.paths, "episodes": self.n_episodes, "causes": list(self.causes),
                "channels": list(self.channels), "in_dim": self.n_channels,
                "label_counts": {l: int((self.labels == l).sum()) for l in sorted(set(self.labels.tolist()))},
                "split_seed": self.split_seed, "split_sizes": {s: len(v) for s, v in self.splits.items()},
                "window": self.cfg.window, "after": self.cfg.after, "jitter": self.cfg.jitter}


def confusion(labels: np.ndarray, pred: np.ndarray, causes) -> dict[str, dict[str, int]]:
    """Counts: true label (raw name) -> predicted cause name."""
    out = {}
    for lab in sorted(set(labels.tolist())):
        sel = labels == lab
        out[lab] = {c: int((pred[sel] == k).sum()) for k, c in enumerate(causes)}
    return out


# --------------------------------------------------------------------------------------------
# Fake evidence files (tests and smoke runs): same format, with the real teacher on a plausible z
# --------------------------------------------------------------------------------------------


def _signature(label: str, z: np.ndarray, holding: np.ndarray, rng: np.random.Generator) -> int:
    """Add the cause's signature (in z units) to the noise, in place; returns the onset (-1: none)."""
    t = len(z)
    held = np.flatnonzero(holding)
    amp = lambda k: rng.choice([-1, 1], k) * rng.uniform(3, 5, k)  # noqa: E731
    if label == "push":  # offset on gripper and force; a held object moves with the hand
        onset = int(rng.integers(5, t - 10))
        g, f = amp(3), amp(3)
        z[onset:onset + 6, :3] += g
        z[onset:onset + 6, F] += f
        z[onset:onset + 6, O] += g * holding[onset:onset + 6, None]
    elif label == "heavier_object":  # extra pull on force z, hand and object sag, while holding
        onset = int(held[0])
        z[holding, 5] += rng.uniform(3, 5)
        z[holding, 2] -= rng.uniform(2, 4)
        z[holding, 8] -= rng.uniform(2, 4)
    elif label == "slippery_object":  # object off from t0 while holding, gripper as predicted
        onset = int(held[rng.integers(0, max(1, len(held) // 2))])
        m = holding & (np.arange(t) >= onset)
        # gravity-driven, as in the teacher: the object slides (sideways any way, down only) and
        # the wrist feels less of its weight
        z[m, O.start:O.start + 2] += amp(2)
        z[m, O.start + 2] -= rng.uniform(3, 5)
        z[m, F.start + 2] -= rng.uniform(3, 5)
    elif label == "camera_shift":  # one-step jump of everything seen, same vector for object and scene
        onset = int(rng.integers(5, t - 5))
        d = amp(3)
        z[onset, O] += d
        z[onset, S] += d
    elif label == "unknown":  # much broader noise
        onset = int(rng.integers(5, t - 10))
        z[onset:onset + 8] *= 5.0
    else:
        onset = -1
    return onset


def write_fake_evidence(path: str | Path, n_episodes: int = 40, seed: int = 0, n_phase: int = 4) -> Path:
    """Episodes of 30-50 steps over the teacher's CHANNELS and CAUSES (balanced labels); each
    disturbance leaves its signature in z (see `_signature`, mirroring the teacher's hypotheses).
    The teacher posterior is the real `infer_cause` on its own window around the surprise peak."""
    rng = np.random.default_rng(seed)
    z_all, hold_all, phase_all, ep_all, labels, teacher, onsets = [], [], [], [], [], [], []
    for e in range(n_episodes):
        label = CAUSES[e % len(CAUSES)]
        t = int(rng.integers(30, 51))
        bounds = np.sort(rng.choice(np.arange(5, t - 5), n_phase - 1, replace=False))
        phase = np.searchsorted(bounds, np.arange(t), side="right")
        holding = (phase >= 1) & (phase <= 2)  # grasped during the middle subgoals
        z = rng.standard_normal((t, N_CH)).astype(np.float32)
        onset = _signature(label, z, holding, rng)
        # The teacher's verdict window around the most surprising step (as SurpriseMonitor.episode_verdict).
        peak = int((0.5 * (z ** 2).sum(-1)).argmax())
        lo = max(0, peak - 20 + 5)
        window = [StepEvidence(i, z[i].astype(float), np.ones(N_CH), np.zeros(1), bool(holding[i]))
                  for i in range(lo, min(t, lo + 20))]
        post = infer_cause(window, dt=0.05).posterior
        z_all.append(z), hold_all.append(holding), phase_all.append(phase), ep_all.append(np.full(t, e))
        labels.append(label), teacher.append([post[c] for c in CAUSES]), onsets.append(onset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, z=np.concatenate(z_all), holding=np.concatenate(hold_all),
             phase=np.concatenate(phase_all).astype(np.int64), episode=np.concatenate(ep_all).astype(np.int64),
             labels=np.array(labels), teacher=np.array(teacher, dtype=np.float32), causes=np.array(CAUSES),
             onset=np.array(onsets, dtype=np.int64), channels=np.array(CHANNELS))
    return path
