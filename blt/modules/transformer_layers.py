"""Shared transformer primitive layers: RMSNorm, SwiGLU, RoPE, and SDPA CausalSelfAttention.

Matches the LLaMA/BLT architectural conventions:
- Pre-RMSNorm
- SwiGLU feed-forward networks (8/3 expansion)
- Rotary Position Embeddings (RoPE) with theta = 500,000 (BLT paper §4.7)
- Causal self-attention powered by torch.nn.functional.scaled_dot_product_attention (SDPA / FlashAttention)
- Optional gradient checkpointing for memory efficiency on consumer GPUs (e.g. RTX 3050 4GB)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.weight).to(input_dtype)


def precompute_rope_freqs_cis(dim: int, max_seq_len: int, theta: float = 500000.0) -> torch.Tensor:
    """Precompute complex exponential frequencies for Rotary Position Embeddings (RoPE).

    Args:
        dim: Dimension per head (must be even).
        max_seq_len: Maximum context length in tokens/bytes.
        theta: Frequency base (paper default: 500,000).

    Returns:
        Tensor of shape (max_seq_len, dim // 2) containing complex cis values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # xq: (batch, seq_len, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim // 2)
    seq_len = xq.shape[1]
    freqs = freqs_cis[:seq_len].to(xq.device)

    # Reshape xq, xk into complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs over batch and heads: (1, seq_len, 1, head_dim // 2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    """Gated feed-forward network using SwiGLU activation."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 64):
        super().__init__()
        if hidden_dim is None:
            # Paper convention: 8/3 expansion
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and FlashAttention SDPA."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
        sliding_window: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.sliding_window = sliding_window
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Precompute RoPE
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs_cis(self.head_dim, max_seq_len, theta=rope_theta),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        if use_rope:
            q, k = apply_rope(q, k, self.freqs_cis[:seq_len])

        # Transpose for SDPA: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Construct causal or sliding window mask if needed
        is_causal = mask is None and self.sliding_window is None
        attn_mask = mask

        if self.sliding_window is not None and mask is None:
            # Local causal sliding window
            i_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
            j_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
            diff = i_idx - j_idx
            # Valid positions: 0 <= diff < sliding_window
            causal_window_mask = (diff >= 0) & (diff < self.sliding_window)
            attn_mask = causal_window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            is_causal = False

        # FlashAttention / Memory efficient attention via PyTorch SDPA
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-RMSNorm, Causal Self-Attention, and SwiGLU."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
        sliding_window: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = CausalSelfAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        def _forward(h, m):
            h = h + self.attn(self.attn_norm(h), mask=m)
            h = h + self.ffn(self.ffn_norm(h))
            return h

        if use_checkpoint and self.training:
            return checkpoint(_forward, x, mask, use_reentrant=False)
        return _forward(x, mask)
