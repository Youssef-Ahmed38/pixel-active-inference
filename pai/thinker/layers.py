"""Building blocks shared by the HRM thinker and its baselines.

Follows HRM (Wang et al. 2025, arXiv 2506.21734): bias-free encoder-only transformer blocks with
gated linear units, post-norm RMSNorm without learnable scale, truncated LeCun-normal init.
Positions use a learned absolute embedding instead of RoPE: evidence sequences are short (~32)
and fixed-length, so rotary encoding buys nothing here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale-free RMSNorm (HRM drops the learnable gain)."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def lecun_init_(module: nn.Module) -> None:
    """Truncated LeCun normal on every Linear weight: std 1/sqrt(fan_in), cut at 2 std."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            std = 1.0 / math.sqrt(m.in_features)
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SwiGLU(nn.Module):
    def __init__(self, d: int, expansion: float = 4.0):
        super().__init__()
        hidden = int(round(expansion * d * 2 / 3 / 32)) * 32  # same param count as a 4x MLP
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)  # bidirectional: evidence is a set/sequence, not a stream
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    """Post-norm transformer block (HRM uses post-norm for stability of the long recurrence)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.attn = Attention(d, n_heads)
        self.mlp = SwiGLU(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rms_norm(x + self.attn(x))
        return rms_norm(x + self.mlp(x))


class Stack(nn.Module):
    """A recurrent module f(z, injection): add the injection, then run the blocks."""

    def __init__(self, d: int, n_heads: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))

    def forward(self, z: torch.Tensor, injection: torch.Tensor | None = None) -> torch.Tensor:
        if injection is not None:
            z = z + injection
        for blk in self.blocks:
            z = blk(z)
        return z


class EvidenceEncoder(nn.Module):
    """Evidence [B, T, C] (+ validity mask, + context vector) -> tokens [B, 1 + T / patch, d].

    Token 0 is a learned readout token: all heads read from it, so the rest of the tokens can
    stay a per-time-step workspace. `patch` consecutive time steps share one token, which
    divides the thinker's cost by ~patch (every thinking step runs over all tokens). The context
    (e.g. noise level, evidence length; later the lower levels' precisions) is added to every
    token, so uncertainty goes *in* everywhere.
    """

    def __init__(self, in_dim: int, ctx_dim: int, max_len: int, d: int, patch: int = 1):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear((in_dim + 1) * patch, d, bias=False)
        self.ctx = nn.Linear(ctx_dim, d, bias=False) if ctx_dim > 0 else None
        self.pos = nn.Parameter(torch.randn(-(-max_len // patch) + 1, d) / math.sqrt(d))
        self.readout_token = nn.Parameter(torch.randn(d) / math.sqrt(d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        feats = torch.cat([x * mask[..., None], mask[..., None].to(x.dtype)], -1)
        if t % self.patch:
            feats = nn.functional.pad(feats, (0, 0, 0, self.patch - t % self.patch))
        t = feats.shape[1] // self.patch
        tok = self.proj(feats.reshape(b, t, -1))
        tok = torch.cat([self.readout_token.expand(b, 1, -1).to(tok.dtype), tok], 1) + self.pos[: t + 1]
        if self.ctx is not None and ctx is not None:
            tok = tok + self.ctx(ctx)[:, None]
        # Unit RMS, like the recurrent states it is added to (HRM scales by sqrt(d) for the same reason).
        return rms_norm(tok)


class ReadoutHeads(nn.Module):
    """Named categorical heads on the readout vector. Add a head = add an entry to `heads`."""

    def __init__(self, d: int, heads: dict[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(d, n) for name, n in heads.items()})

    def forward(self, r: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(r).float() for name, head in self.heads.items()}
