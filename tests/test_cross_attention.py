"""Unit tests for Encoder and Decoder Cross-Attention modules."""

import pytest
import torch
from blt.modules.cross_attention import (
    EncoderCrossAttention,
    DecoderCrossAttention,
    create_patch_membership_mask,
    create_decoder_causal_patch_mask,
)


def test_cross_attention_masks():
    # Batch 2, seq_len 6, 3 patches
    # Batch 0: patches [0, 0, 1, 1, 2, 2]
    # Batch 1: patches [0, 1, 1, 2, 2, 2]
    patch_indices = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 2, 2, 2],
    ], dtype=torch.long)
    num_patches = 3

    # 1. Membership mask: (batch, 1, num_patches, seq_len)
    memb_mask = create_patch_membership_mask(patch_indices, num_patches)
    assert memb_mask.shape == (2, 1, 3, 6)
    assert memb_mask.dtype == torch.bool

    # For batch 0, patch 0 should match bytes 0 and 1
    assert bool(memb_mask[0, 0, 0, 0]) is True
    assert bool(memb_mask[0, 0, 0, 1]) is True
    assert bool(memb_mask[0, 0, 0, 2]) is False

    # 2. Causal mask: (batch, 1, seq_len, num_patches)
    causal_mask = create_decoder_causal_patch_mask(patch_indices, num_patches)
    assert causal_mask.shape == (2, 1, 6, 3)
    assert causal_mask.dtype == torch.bool

    # In batch 0, byte 0 is in patch 0, so it can only attend to patch 0, not patch 1 or 2
    assert bool(causal_mask[0, 0, 0, 0]) is True
    assert bool(causal_mask[0, 0, 0, 1]) is False
    assert bool(causal_mask[0, 0, 0, 2]) is False

    # In batch 0, byte 3 is in patch 1, so it can attend to patch 0 and 1, but not 2
    assert bool(causal_mask[0, 0, 3, 0]) is True
    assert bool(causal_mask[0, 0, 3, 1]) is True
    assert bool(causal_mask[0, 0, 3, 2]) is False


def test_encoder_cross_attention():
    bsz, seq_len = 2, 8
    byte_dim, patch_dim = 64, 128
    n_heads = 4
    num_patches = 3

    enc_cross = EncoderCrossAttention(
        byte_dim=byte_dim,
        patch_dim=patch_dim,
        n_heads=n_heads,
    )

    byte_hidden = torch.randn(bsz, seq_len, byte_dim, requires_grad=True)
    patch_indices = torch.tensor([
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 0, 1, 1, 1, 1, 2, 2],
    ], dtype=torch.long)

    patch_repr = enc_cross(byte_hidden, patch_indices, num_patches)
    assert patch_repr.shape == (bsz, num_patches, patch_dim)
    assert not torch.isnan(patch_repr).any()

    # Test gradient flow
    loss = patch_repr.sum()
    loss.backward()
    assert byte_hidden.grad is not None
    assert enc_cross.q_proj.weight.grad is not None


def test_decoder_cross_attention():
    bsz, seq_len = 2, 8
    byte_dim, patch_dim = 64, 128
    n_heads = 4
    num_patches = 3

    dec_cross = DecoderCrossAttention(
        byte_dim=byte_dim,
        patch_dim=patch_dim,
        n_heads=n_heads,
    )

    byte_hidden = torch.randn(bsz, seq_len, byte_dim, requires_grad=True)
    patch_hidden = torch.randn(bsz, num_patches, patch_dim, requires_grad=True)
    patch_indices = torch.tensor([
        [0, 0, 0, 1, 1, 2, 2, 2],
        [0, 0, 1, 1, 1, 1, 2, 2],
    ], dtype=torch.long)

    out = dec_cross(byte_hidden, patch_hidden, patch_indices)
    assert out.shape == (bsz, seq_len, byte_dim)
    assert not torch.isnan(out).any()

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert byte_hidden.grad is not None
    assert patch_hidden.grad is not None
    assert dec_cross.q_proj.weight.grad is not None
