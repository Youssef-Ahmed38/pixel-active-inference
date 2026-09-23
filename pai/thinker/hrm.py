"""The thinker: a Hierarchical Reasoning Model (Wang et al. 2025, arXiv 2506.21734).

Two coupled recurrent modules over the evidence tokens:
- L (fast): N*T updates per segment, z_L <- f_L(z_L + z_H + x~)
- H (slow): one update every T L-steps,  z_H <- f_H(z_H + z_L)
Readout heads and the halting Q-head read the readout token of z_H.

A *segment* is one N x T pass. Deep supervision runs several segments on the same input, each
starting from the previous segment's detached state. Inside a segment only the final L and H
step carry gradients (the paper's one-step gradient: output -> z_H -> z_L -> input embedding),
so memory is O(1) in the number of thinking steps.

Learned halting (the paper's ACT with Q-learning): Q-head -> (halt, continue) logits; an example
stops after segment m when q_halt > q_continue, or at the segment cap. `think` really drops halted
examples from the batch, so the step count it records is also the compute actually spent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pai.thinker.layers import EvidenceEncoder, ReadoutHeads, Stack, lecun_init_


@dataclass
class SegmentOut:
    logits: dict[str, torch.Tensor]  # head name -> [B, n_classes]
    q: torch.Tensor | None  # [B, 2] (halt, continue) logits; None for models without halting


@dataclass
class ThinkResult:
    logits: dict[str, torch.Tensor]  # readout at the segment where each example halted
    segments: torch.Tensor  # [B] long, segments actually used (1..max)
    steps_per_segment: int = 1  # low-level steps per segment, to convert to fine-grained steps

    def probs(self, head: str = "cause") -> torch.Tensor:
        return self.logits[head].softmax(-1)


class HRM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        max_len: int,
        heads: dict[str, int],
        d_model: int = 128,
        n_heads: int = 4,
        h_layers: int = 2,
        l_layers: int = 2,
        h_cycles: int = 2,
        l_steps: int = 2,
        max_segments: int = 8,
        patch: int = 1,
    ):
        super().__init__()
        self.h_cycles, self.l_steps, self.max_segments = h_cycles, l_steps, max_segments
        self.encoder = EvidenceEncoder(in_dim, ctx_dim, max_len, d_model, patch)
        self.H = Stack(d_model, n_heads, h_layers)
        self.L = Stack(d_model, n_heads, l_layers)
        self.readout = ReadoutHeads(d_model, heads)
        self.q_head = nn.Linear(d_model, 2)
        lecun_init_(self)
        # Q starts near "don't halt yet" (as in the reference code), so early training explores depth.
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)
        # Fixed initial states: one truncated-normal vector each, broadcast over tokens.
        self.register_buffer("z_H0", nn.init.trunc_normal_(torch.empty(d_model), std=1.0, a=-2.0, b=2.0))
        self.register_buffer("z_L0", nn.init.trunc_normal_(torch.empty(d_model), std=1.0, a=-2.0, b=2.0))

    @property
    def steps_per_segment(self) -> int:
        return self.h_cycles * self.l_steps

    def _init_state(self, xt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.z_H0.to(xt.dtype).expand_as(xt), self.z_L0.to(xt.dtype).expand_as(xt)

    def _segment(self, z_h: torch.Tensor, z_l: torch.Tensor, xt: torch.Tensor):
        """One N x T pass; all but the last L and H update run without gradients."""
        n_total = self.h_cycles * self.l_steps
        with torch.no_grad():
            for i in range(n_total - 1):
                z_l = self.L(z_l, z_h + xt)
                if (i + 1) % self.l_steps == 0:
                    z_h = self.H(z_h, z_l)
        z_l = self.L(z_l, z_h + xt)
        z_h = self.H(z_h, z_l)
        return z_h, z_l

    def _read(self, z_h: torch.Tensor) -> SegmentOut:
        r = z_h[:, 0]
        return SegmentOut(self.readout(r), self.q_head(r).float())

    def forward_segments(self, x, mask, ctx=None, n_segments: int | None = None) -> list[SegmentOut]:
        """Training pass: every segment's readout, states detached between segments."""
        xt = self.encoder(x, mask, ctx)
        z_h, z_l = self._init_state(xt)
        outs = []
        for _ in range(n_segments or self.max_segments):
            z_h, z_l = self._segment(z_h.detach(), z_l.detach(), xt)
            outs.append(self._read(z_h))
        return outs

    @torch.no_grad()
    def think(self, x, mask, ctx=None, max_segments: int | None = None, halt: bool = True) -> ThinkResult:
        """Inference with learned early stopping. halt=False forces exactly max_segments."""
        cap = max_segments or self.max_segments
        b = x.shape[0]
        xt = self.encoder(x, mask, ctx)
        z_h, z_l = self._init_state(xt)
        active = torch.arange(b, device=x.device)
        segments = torch.zeros(b, dtype=torch.long, device=x.device)
        final: dict[str, torch.Tensor] = {}
        for m in range(1, cap + 1):
            z_h, z_l = self._segment(z_h, z_l, xt)
            out = self._read(z_h)
            stop = torch.full_like(active, m == cap, dtype=torch.bool)
            if halt:
                stop |= out.q[:, 0] > out.q[:, 1]
            for name, lg in out.logits.items():
                final.setdefault(name, lg.new_zeros(b, lg.shape[-1]))[active[stop]] = lg[stop]
            segments[active[stop]] = m
            keep = ~stop
            if not keep.any():
                break
            active, z_h, z_l, xt = active[keep], z_h[keep], z_l[keep], xt[keep]
        return ThinkResult(final, segments, self.steps_per_segment)
