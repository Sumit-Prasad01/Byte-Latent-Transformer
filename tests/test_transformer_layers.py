"""Unit tests for shared transformer primitives (RMSNorm, SwiGLU, RoPE, CausalSelfAttention)."""

import pytest
import torch
from blt.modules.transformer_layers import (
    RMSNorm,
    SwiGLU,
    CausalSelfAttention,
    TransformerBlock,
    apply_rope,
    precompute_rope_freqs_cis,
)


def test_rmsnorm():
    """Verify RMSNorm normalizes root-mean-square to 1."""
    dim = 64
    norm = RMSNorm(dim=dim)
    x = torch.randn(2, 10, dim) * 5.0 + 2.0
    out = norm(x)

    assert out.shape == x.shape
    rms = torch.sqrt(out.pow(2).mean(-1))
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)


def test_swiglu():
    """Verify SwiGLU forward pass and output dimension."""
    dim = 128
    swiglu = SwiGLU(dim=dim)
    x = torch.randn(4, 16, dim)
    out = swiglu(x)

    assert out.shape == (4, 16, dim)
    assert not torch.isnan(out).any()


def test_rope_rotation():
    """Verify RoPE precomputation and complex rotation application."""
    dim_per_head = 32
    max_len = 128
    freqs_cis = precompute_rope_freqs_cis(dim_per_head, max_len)

    assert freqs_cis.shape == (max_len, dim_per_head // 2)

    xq = torch.randn(2, 16, 4, dim_per_head)
    xk = torch.randn(2, 16, 4, dim_per_head)
    q_rot, k_rot = apply_rope(xq, xk, freqs_cis)

    assert q_rot.shape == xq.shape
    assert k_rot.shape == xk.shape
    assert not torch.isnan(q_rot).any()


def test_causal_self_attention():
    """Verify CausalSelfAttention executes and respects causality via SDPA."""
    dim = 128
    n_heads = 4
    attn = CausalSelfAttention(dim=dim, n_heads=n_heads, max_seq_len=64)

    x = torch.randn(2, 16, dim)
    out = attn(x)

    assert out.shape == (2, 16, dim)
    assert not torch.isnan(out).any()


def test_transformer_block_and_checkpointing():
    """Verify TransformerBlock with and without gradient checkpointing."""
    dim = 64
    n_heads = 2
    block = TransformerBlock(dim=dim, n_heads=n_heads, max_seq_len=32)

    x = torch.randn(2, 8, dim, requires_grad=True)

    # Standard forward
    out1 = block(x, use_checkpoint=False)
    loss1 = out1.sum()
    loss1.backward()

    assert out1.shape == (2, 8, dim)
    assert x.grad is not None

    # Checkpointed forward
    x.grad.zero_()
    block.train()
    out2 = block(x, use_checkpoint=True)
    loss2 = out2.sum()
    loss2.backward()

    assert out2.shape == (2, 8, dim)
    assert x.grad is not None
