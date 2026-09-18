"""Unit and integration tests for ByteLatentTransformer (BLT) forward, backward, and generation."""

import pytest
import torch
from blt.model.blt_model import ByteLatentTransformer, compute_patch_indices_from_boundaries


@pytest.fixture
def tiny_blt():
    """Construct a tiny BLT model for fast unit testing."""
    return ByteLatentTransformer(
        vocab_size=260,
        byte_dim=32,
        patch_dim=64,
        encoder_layers=1,
        encoder_heads=2,
        encoder_window=16,
        latent_layers=2,
        latent_heads=2,
        latent_max_patches=64,
        decoder_layers=1,
        decoder_heads=2,
        decoder_window=16,
        cross_attn_heads=2,
        ngram_sizes=(3,),
        vocab_per_ngram=100,
        max_seq_len=128,
        grad_checkpointing=False,
    )


def test_blt_forward_shapes(tiny_blt):
    bsz, seq_len = 2, 16
    tokens = torch.randint(0, 256, (bsz, seq_len))
    targets = torch.randint(0, 260, (bsz, seq_len))

    # 1. Forward with default strided patches
    logits, loss = tiny_blt(tokens, targets=targets)
    assert logits.shape == (bsz, seq_len, 260)
    assert loss is not None
    assert not torch.isnan(loss)

    # 2. Forward with explicit patch boundaries
    boundaries = torch.zeros(bsz, seq_len, dtype=torch.bool)
    boundaries[:, 0] = True
    boundaries[:, 4] = True
    boundaries[:, 10] = True

    logits_b, loss_b = tiny_blt(tokens, targets=targets, patch_boundaries=boundaries)
    assert logits_b.shape == (bsz, seq_len, 260)
    assert loss_b is not None
    assert not torch.isnan(loss_b)


def test_blt_backward_and_gradients(tiny_blt):
    bsz, seq_len = 2, 8
    tokens = torch.randint(0, 256, (bsz, seq_len))
    targets = torch.randint(0, 260, (bsz, seq_len))

    _, loss = tiny_blt(tokens, targets=targets)
    loss.backward()

    # Verify gradients reach all 4 core components
    assert tiny_blt.embeddings.byte_embedding.weight.grad is not None
    assert tiny_blt.local_encoder.layers[0].attn.q_proj.weight.grad is not None
    assert tiny_blt.latent_transformer.layers[0].attn.q_proj.weight.grad is not None
    assert tiny_blt.local_decoder.lm_head.weight.grad is not None


def test_blt_overfitting_toy_sequence(tiny_blt):
    """Overfit a tiny repeating sequence to prove model capacity and gradient correctness."""
    torch.manual_seed(42)
    # Target phrase in UTF-8: "hello world! " repeated
    phrase = b"hello world! hello world! "
    tokens = torch.tensor(list(phrase), dtype=torch.long).unsqueeze(0)  # (1, 26)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    optimizer = torch.optim.AdamW(tiny_blt.parameters(), lr=0.01)

    tiny_blt.train()
    initial_loss = None
    final_loss = None

    for step in range(50):
        optimizer.zero_grad()
        _, loss = tiny_blt(inputs, targets=targets)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.25, f"Expected final loss < {initial_loss * 0.25:.4f}, got {final_loss:.4f}"


def test_blt_generation(tiny_blt):
    tiny_blt.eval()
    prompt = torch.tensor([[ord("h"), ord("e"), ord("l"), ord("l")]], dtype=torch.long)
    gen = tiny_blt.generate(prompt, max_new_tokens=6, temperature=0.8)
    assert gen.shape == (1, 10)
    assert gen[0, :4].tolist() == prompt[0].tolist()
