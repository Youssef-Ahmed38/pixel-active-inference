"""One-shot baselines with the same encoder/readout interface as the HRM thinker.

They answer "does the HRM structure help?": FixedDepthTransformer has the same blocks and
roughly the same parameter count as the HRM (H + L layers stacked once, no recurrence, no
halting); MLPThinker is a plain MLP on the flattened evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pai.thinker.hrm import SegmentOut, ThinkResult
from pai.thinker.layers import EvidenceEncoder, ReadoutHeads, Stack, lecun_init_


class _OneShot(nn.Module):
    steps_per_segment = 1

    def _logits(self, x, mask, ctx) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward_segments(self, x, mask, ctx=None, n_segments: int | None = None) -> list[SegmentOut]:
        return [SegmentOut(self._logits(x, mask, ctx), None)]

    @torch.no_grad()
    def think(self, x, mask, ctx=None, max_segments: int | None = None, halt: bool = True) -> ThinkResult:
        ones = torch.ones(x.shape[0], dtype=torch.long, device=x.device)
        return ThinkResult(self._logits(x, mask, ctx), ones, 1)


class FixedDepthTransformer(_OneShot):
    def __init__(self, in_dim: int, ctx_dim: int, max_len: int, heads: dict[str, int],
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4, patch: int = 1):
        super().__init__()
        self.encoder = EvidenceEncoder(in_dim, ctx_dim, max_len, d_model, patch)
        self.body = Stack(d_model, n_heads, n_layers)
        self.readout = ReadoutHeads(d_model, heads)
        lecun_init_(self)

    def _logits(self, x, mask, ctx):
        return self.readout(self.body(self.encoder(x, mask, ctx))[:, 0])


class MLPThinker(_OneShot):
    def __init__(self, in_dim: int, ctx_dim: int, max_len: int, heads: dict[str, int],
                 hidden: int = 512, n_layers: int = 4):
        super().__init__()
        dims = [max_len * (in_dim + 1) + ctx_dim] + [hidden] * n_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.readout = ReadoutHeads(hidden, heads)
        lecun_init_(self)

    def _logits(self, x, mask, ctx):
        parts = [(x * mask[..., None]).flatten(1), mask.to(x.dtype)]
        if ctx is not None:
            parts.append(ctx)
        return self.readout(self.body(torch.cat(parts, -1)))
