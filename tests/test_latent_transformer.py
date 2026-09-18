"""Unit tests for LatentTransformer."""

import pytest
import torch
from blt.modules.latent_transformer import LatentTransformer


def test_latent_transformer_shapes_and_forward():
    bsz, num_patches = 2, 8
    dim = 64
    n_heads = 4
    n_layers = 3

    latent_tr = LatentTransformer(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        grad_checkpointing=False,
    )

    x = torch.randn(bsz, num_patches, dim, requires_grad=True)
    out = latent_tr(x)
    assert out.shape == (bsz, num_patches, dim)
    assert not torch.isnan(out).any()

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_latent_transformer_gradient_checkpointing_parity():
    torch.manual_seed(42)
    bsz, num_patches = 2, 6
    dim = 32
    n_heads = 4

    latent_tr = LatentTransformer(
        dim=dim,
        n_layers=2,
        n_heads=n_heads,
        grad_checkpointing=False,
    )

    latent_tr.eval()
    x = torch.randn(bsz, num_patches, dim)

    out_no_cp = latent_tr(x, use_checkpoint=False)
    out_cp = latent_tr(x, use_checkpoint=True)

    assert torch.allclose(out_no_cp, out_cp, atol=1e-5)
