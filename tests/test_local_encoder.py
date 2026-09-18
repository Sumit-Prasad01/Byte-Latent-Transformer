"""Unit tests for LocalEncoder."""

import pytest
import torch
from blt.modules.local_encoder import LocalEncoder


def test_local_encoder_shapes_and_forward():
    bsz, seq_len = 2, 16
    dim = 64
    patch_dim = 128
    n_heads = 4
    window_size = 8
    num_patches = 4

    encoder = LocalEncoder(
        dim=dim,
        n_layers=2,
        n_heads=n_heads,
        window_size=window_size,
        patch_dim=patch_dim,
        cross_attn_heads=n_heads,
    )

    x = torch.randn(bsz, seq_len, dim, requires_grad=True)

    # 1. Forward without patch indices (returns byte representations only)
    byte_hidden = encoder(x)
    assert byte_hidden.shape == (bsz, seq_len, dim)
    assert not torch.isnan(byte_hidden).any()

    # 2. Forward with patch indices (returns byte_hidden and patch_hidden)
    patch_indices = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
    ], dtype=torch.long)

    b_out, p_out = encoder(x, patch_indices=patch_indices, num_patches=num_patches)
    assert b_out.shape == (bsz, seq_len, dim)
    assert p_out.shape == (bsz, num_patches, patch_dim)
    assert not torch.isnan(p_out).any()

    # Test gradient flow
    loss = p_out.sum() + b_out.sum()
    loss.backward()
    assert x.grad is not None
