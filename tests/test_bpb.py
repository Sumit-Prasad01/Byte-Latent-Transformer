"""Unit tests for Bits-Per-Byte (BPB) calculation."""

import math
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from blt.eval.bpb import compute_bpb, loss_to_bpb, evaluate_byte_model_bpb
from blt.model.baseline_bpe_model import BaselineTransformer


def test_bpb_theoretical_uniform():
    """Verify uniform random distribution over 256 bytes produces exactly 8.0 BPB."""
    # Entropy of uniform distribution over 256 symbols is log_2(256) = 8 bits
    num_bytes = 1000
    loss_nats = math.log(256) * num_bytes

    bpb = compute_bpb(loss_nats, num_bytes)
    assert abs(bpb - 8.0) < 1e-5


def test_loss_to_bpb():
    """Verify conversion of per-byte nats to bits."""
    loss_nats = 2.0
    expected_bits = 2.0 / math.log(2)
    assert abs(loss_to_bpb(loss_nats) - expected_bits) < 1e-5


def test_evaluate_byte_model_bpb():
    """Verify evaluate_byte_model_bpb executes over a DataLoader."""
    model = BaselineTransformer(vocab_size=260, dim=32, n_layers=1, n_heads=2, max_seq_len=32)
    dummy_data = torch.randint(0, 256, (8, 16), dtype=torch.long)
    loader = DataLoader(TensorDataset(dummy_data), batch_size=4)

    bpb = evaluate_byte_model_bpb(model, loader)
    assert isinstance(bpb, float)
    assert bpb > 0.0
    # For untrained model on bytes, BPB should be close to uniform (around 7.5 - 8.5)
    assert 5.0 < bpb < 10.0
