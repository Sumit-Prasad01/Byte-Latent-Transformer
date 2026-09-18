"""Unit tests for LocalDecoder."""

import pytest
import torch
import torch.nn.functional as F
from blt.modules.local_decoder import LocalDecoder


def test_local_decoder_shapes_and_loss():
    bsz, seq_len = 2, 12
    byte_dim = 64
    patch_dim = 128
    vocab_size = 260
    n_heads = 4
    num_patches = 3

    decoder = LocalDecoder(
        dim=byte_dim,
        patch_dim=patch_dim,
        n_layers=2,
        n_heads=n_heads,
        cross_attn_heads=n_heads,
        window_size=8,
        vocab_size=vocab_size,
    )

    byte_hidden = torch.randn(bsz, seq_len, byte_dim, requires_grad=True)
    patch_hidden = torch.randn(bsz, num_patches, patch_dim, requires_grad=True)
    patch_indices = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    ], dtype=torch.long)

    logits = decoder(byte_hidden, patch_hidden, patch_indices)
    assert logits.shape == (bsz, seq_len, vocab_size)
    assert not torch.isnan(logits).any()

    # Targets for cross entropy loss
    targets = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.long)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    assert not torch.isnan(loss)
    loss.backward()

    assert byte_hidden.grad is not None
    assert patch_hidden.grad is not None
    assert decoder.lm_head.weight.grad is not None
