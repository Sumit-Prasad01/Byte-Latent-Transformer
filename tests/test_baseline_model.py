"""Unit tests for BaselineTransformer (byte stride-1 and BPE modes)."""

import pytest
import torch
import torch.optim as optim
from blt.model.baseline_bpe_model import BaselineTransformer


def test_baseline_transformer_shapes():
    """Verify forward pass output shapes and logits dimensionality."""
    model = BaselineTransformer(
        vocab_size=260,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
    )

    tokens = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 16), dtype=torch.long)

    logits, loss = model(tokens, targets=targets)

    assert logits.shape == (2, 16, 260)
    assert loss is not None
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_baseline_overfitting_toy_sequence():
    """Verify small model can overfit a simple repeating byte sequence (integration sanity check)."""
    torch.manual_seed(42)
    model = BaselineTransformer(
        vocab_size=260,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    # Toy text: "Hello Hello Hello "
    text = "Hello Hello Hello "
    byte_vals = [ord(c) for c in text]
    x = torch.tensor([byte_vals[:-1]], dtype=torch.long)
    y = torch.tensor([byte_vals[1:]], dtype=torch.long)

    # Initial loss
    model.eval()
    _, initial_loss = model(x, targets=y)

    # Train for 40 steps
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()

    model.eval()
    _, final_loss = model(x, targets=y)

    assert final_loss.item() < initial_loss.item() * 0.3, (
        f"Model did not overfit: initial={initial_loss.item():.4f}, final={final_loss.item():.4f}"
    )


def test_baseline_generation():
    """Verify autoregressive generation emits expected sequence length."""
    model = BaselineTransformer(
        vocab_size=260,
        dim=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=64,
    )

    prompt = torch.tensor([[65, 66]], dtype=torch.long)  # 'AB'
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.7, top_p=0.9)

    assert generated.shape == (1, 12)
    assert (generated >= 0).all() and (generated < 260).all()
