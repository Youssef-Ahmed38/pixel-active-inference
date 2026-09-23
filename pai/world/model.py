"""Entity world model: a transformer over entity tokens that predicts one planner step ahead.

For every entity it outputs the change of its dynamic features (position, velocity, yaw, gripper
opening) as a Gaussian: a mean and a log-variance. It also predicts the next wrist force on the
gripper (absolute, not a change) from the state alone: force features are masked out of the input,
so the model has to *expect* a force from what it believes is happening (e.g. "I am holding the red
block"), which is what makes a heavier-than-expected object or a push surprising.

The log-variances are the model's own uncertainty per feature, i.e. learned (inverse) precisions,
which the free-energy terms weight prediction errors with. An ensemble of independent members adds
epistemic uncertainty: where they disagree the model has not learned the dynamics yet (curiosity).

Everything is normalised with statistics stored in the checkpoint, so the model is self-contained.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from pai.world.entities import DYNAMIC, FORCE, TOKEN_DIM

N_DYN = DYNAMIC.stop - DYNAMIC.start
N_FORCE = FORCE.stop - FORCE.start


class Prediction(NamedTuple):
    next_tokens: torch.Tensor   # (B, N, TOKEN_DIM), from the ensemble mean
    delta: torch.Tensor         # (B, N, N_DYN) mean change of the dynamic features
    aleatoric: torch.Tensor     # (B, N, N_DYN) predicted noise variance of the change
    epistemic: torch.Tensor     # (B, N, N_DYN) ensemble disagreement (variance of member means)
    force: torch.Tensor         # (B, N_FORCE) expected next wrist force (N)
    force_var: torch.Tensor     # (B, N_FORCE) aleatoric + epistemic variance of that force


class EntityDynamics(nn.Module):
    def __init__(self, action_dim: int, d_model: int = 128, n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.action = nn.Sequential(nn.Linear(action_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * N_DYN))
        self.force_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * N_FORCE))

    def forward(self, tokens: torch.Tensor, action: torch.Tensor):
        """tokens (B, N, TOKEN_DIM) normalised, action (B, A) normalised ->
        delta mean, delta logvar (B, N, N_DYN), force mean, force logvar (B, N_FORCE)."""
        tokens = tokens.clone()
        tokens[..., FORCE] = 0.0  # sensation is predicted, never read
        h = self.embed(tokens)
        h = torch.cat([h[:, :1] + self.action(action)[:, None], h[:, 1:]], dim=1)  # the action acts through the gripper
        h = self.encoder(h)
        out = self.head(h)
        f = self.force_head(h[:, 0])
        return (out[..., :N_DYN], out[..., N_DYN:].clamp(-12.0, 4.0),
                f[..., :N_FORCE], f[..., N_FORCE:].clamp(-12.0, 4.0))


class EnsembleWorldModel(nn.Module):
    """Ensemble of EntityDynamics plus normalisation, working in raw (unnormalised) units."""

    def __init__(self, action_dim: int, n_members: int = 5, **kw):
        super().__init__()
        self.members = nn.ModuleList([EntityDynamics(action_dim, **kw) for _ in range(n_members)])
        self._config = {"action_dim": action_dim, "n_members": n_members, **kw}
        for name, shape in (("tok_mean", TOKEN_DIM), ("tok_std", TOKEN_DIM), ("act_mean", action_dim),
                            ("act_std", action_dim), ("delta_mean", N_DYN), ("delta_std", N_DYN),
                            ("force_mean", N_FORCE), ("force_std", N_FORCE)):
            self.register_buffer(name, torch.zeros(shape) if "mean" in name else torch.ones(shape))

    def config(self) -> dict:
        return dict(self._config)

    def set_normalisation(self, tokens, actions, deltas, forces) -> None:
        eps = 1e-4
        self.tok_mean.copy_(tokens.reshape(-1, TOKEN_DIM).mean(0))
        self.tok_std.copy_(tokens.reshape(-1, TOKEN_DIM).std(0).clamp_min(eps))
        self.act_mean.copy_(actions.mean(0))
        self.act_std.copy_(actions.std(0).clamp_min(eps))
        self.delta_mean.copy_(deltas.reshape(-1, N_DYN).mean(0))
        self.delta_std.copy_(deltas.reshape(-1, N_DYN).std(0).clamp_min(eps))
        self.force_mean.copy_(forces.mean(0))
        self.force_std.copy_(forces.std(0).clamp_min(eps))

    def _norm_in(self, tokens, action):
        return (tokens - self.tok_mean) / self.tok_std, (action - self.act_mean) / self.act_std

    def member_forward(self, i: int, tokens, action):
        """Normalised (delta mean, delta logvar, force mean, force logvar) for member i (training)."""
        t, a = self._norm_in(tokens, action)
        return self.members[i](t, a)

    def predict(self, tokens: torch.Tensor, action: torch.Tensor, member: int | None = None) -> Prediction:
        """Next tokens in raw units. member=None uses the whole ensemble; member=i one member (cheap rollouts)."""
        t, a = self._norm_in(tokens, action)
        idx = range(len(self.members)) if member is None else [member]
        means, vars_, fmeans, fvars = [], [], [], []
        for i in idx:
            m, lv, fm, flv = self.members[i](t, a)
            means.append(m * self.delta_std + self.delta_mean)
            vars_.append(lv.exp() * self.delta_std**2)
            fmeans.append(fm * self.force_std + self.force_mean)
            fvars.append(flv.exp() * self.force_std**2)
        means, vars_, fmeans, fvars = map(torch.stack, (means, vars_, fmeans, fvars))
        mean, fmean = means.mean(0), fmeans.mean(0)
        epistemic = means.var(0, unbiased=False) if len(idx) > 1 else torch.zeros_like(mean)
        f_epi = fmeans.var(0, unbiased=False) if len(idx) > 1 else torch.zeros_like(fmean)
        nxt = tokens.clone()
        nxt[..., DYNAMIC] = tokens[..., DYNAMIC] + mean
        nxt[..., 0, FORCE] = fmean
        return Prediction(nxt, mean, vars_.mean(0), epistemic, fmean, fvars.mean(0) + f_epi)


def load_world_model(path, device="cpu") -> EnsembleWorldModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EnsembleWorldModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
