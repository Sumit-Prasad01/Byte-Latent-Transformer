"""Unit tests for ByteEntropyModel and entropy computation."""

import pytest
import torch
import torch.optim as optim
from blt.patching.entropy_model import ByteEntropyModel


def test_entropy_model_shapes():
    """Verify ByteEntropyModel output dimensions and loss calculation."""
    model = ByteEntropyModel(
        vocab_size=260,
        dim=64,
        n_layers=2,
        n_heads=2,
        sliding_window=64,
        max_seq_len=128,
    )

    tokens = torch.randint(0, 256, (2, 32), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 32), dtype=torch.long)

    logits, loss = model(tokens, targets=targets)

    assert logits.shape == (2, 32, 260)
    assert loss is not None
    assert loss.item() > 0.0


def test_compute_entropy():
    """Verify compute_entropy returns per-byte Shannon entropy."""
    model = ByteEntropyModel(
        vocab_size=260,
        dim=32,
        n_layers=1,
        n_heads=2,
        sliding_window=32,
        max_seq_len=64,
    )

    text = "Once upon a time"
    entropy = model.compute_entropy(text)

    assert entropy.shape == (len(text.encode("utf-8")),)
    assert (entropy >= 0.0).all()
    # Position 0 must equal prior entropy (8.0)
    assert abs(entropy[0].item() - 8.0) < 1e-4


def test_entropy_drop_on_repeating_pattern():
    """Verify entropy model learns to assign lower entropy to predictable repeating sequences."""
    torch.manual_seed(42)
    model = ByteEntropyModel(
        vocab_size=260,
        dim=64,
        n_layers=2,
        n_heads=2,
        sliding_window=64,
        max_seq_len=128,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    # Repeating sequence: "abababababababab"
    pattern = b"abababababababab"
    tokens = torch.tensor([list(pattern)], dtype=torch.long)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    # Train for 35 steps
    model.train()
    for _ in range(35):
        optimizer.zero_grad()
        _, loss = model(inputs, targets=targets)
        loss.backward()
        optimizer.step()

    # Predictable characters should have much lower entropy than random bytes
    ent_predictable = model.compute_entropy(b"abababab")
    ent_random = model.compute_entropy(bytes([42, 189, 7, 211, 99, 14, 88, 245]))

    # Mean entropy for predictable pattern (after position 0) should be substantially lower
    mean_pred = ent_predictable[2:].mean().item()
    mean_rand = ent_random[2:].mean().item()

    assert mean_pred < mean_rand, (
        f"Predictable entropy {mean_pred:.2f} was not lower than random {mean_rand:.2f}"
    )
