"""Unit tests for ByteDataset, ByteDataLoader, and data sharding."""

import os
import numpy as np
import pytest
import torch
from blt.data.dataset import (
    ByteDataset,
    ByteDataLoader,
    BYTE_VOCAB_SIZE,
    DOC_BOUNDARY_TOKEN,
)


def test_byte_dataset_basic():
    """Verify ByteDataset chunks raw strings correctly."""
    text = "Hello world! This is a test for ByteDataset chunking."
    seq_len = 16
    ds = ByteDataset(text, sequence_length=seq_len, stride=8)

    expected_samples = (len(text.encode("utf-8")) - seq_len) // 8 + 1
    assert len(ds) == expected_samples

    sample0 = ds[0]
    assert isinstance(sample0, torch.Tensor)
    assert sample0.shape == (seq_len,)
    assert sample0.dtype == torch.int64


def test_byte_dataset_lossless_roundtrip():
    """Verify text -> bytes -> decode roundtrip is lossless."""
    original_text = "Once upon a time in a byte-level latent transformer, utf-8 works 🚀! 12345."
    ds = ByteDataset(original_text, sequence_length=len(original_text.encode("utf-8")))

    sample = ds[0]
    decoded = ByteDataset.decode_bytes(sample)
    assert decoded == original_text


def test_byte_dataset_memmap_shard():
    """Verify zero-copy memory-mapped reading of binary shard if available."""
    train_bin = os.path.join("data", "processed", "train.bin")
    if os.path.exists(train_bin):
        ds = ByteDataset(train_bin, sequence_length=256, stride=256)
        assert len(ds) > 0

        sample = ds[0]
        assert sample.shape == (256,)
        assert sample.dtype == torch.int64

        decoded = ByteDataset.decode_bytes(sample[:50])
        assert len(decoded) > 0


def test_byte_dataloader_batching():
    """Verify DataLoader yields batched tensors with expected shapes."""
    text = "A" * 1000
    ds = ByteDataset(text, sequence_length=128, stride=64)
    loader = ByteDataLoader(ds, batch_size=4, shuffle=False)

    for batch in loader:
        assert batch.shape == (4, 128)
        assert batch.dtype == torch.int64
        break
