"""Unit tests for multi-strategy patchers, boundary rules, and threshold calibration."""

import numpy as np
import pytest
import torch
from blt.patching.patchers import (
    strided_patcher,
    space_patcher,
    entropy_patcher_global,
    entropy_patcher_monotonic,
    StreamingPatcher,
)
from blt.patching.boundary_rules import (
    compute_average_patch_size,
    calibrate_monotonic_threshold,
)


def test_strided_patcher():
    """Verify strided patcher places boundaries every k bytes."""
    text = "123456789012"
    mask = strided_patcher(text, k=4)

    assert len(mask) == len(text)
    assert mask[0] == 1
    assert mask[4] == 1
    assert mask[8] == 1
    assert mask[1] == 0
    assert mask.sum() == 3


def test_space_patcher():
    """Verify space patcher places boundaries after whitespace delimiters."""
    text = "Cat dog puppy"
    mask = space_patcher(text)

    # Cat (0), dog (4), puppy (8)
    assert mask[0] == 1
    assert mask[4] == 1
    assert mask[8] == 1
    assert mask[1] == 0
    assert mask[2] == 0


def test_entropy_patcher_global():
    """Verify global constraint patcher triggers on entropy threshold."""
    entropy = np.array([2.0, 1.0, 4.5, 0.5, 5.0], dtype=np.float32)
    mask = entropy_patcher_global(entropy, theta_g=3.0)

    # 0 is always 1, 4.5 > 3.0 (idx 2), 5.0 > 3.0 (idx 4)
    assert mask[0] == 1
    assert mask[1] == 0
    assert mask[2] == 1
    assert mask[3] == 0
    assert mask[4] == 1


def test_calibrate_monotonic_threshold():
    """Verify threshold calibration converges to target average patch size."""
    np.random.seed(42)
    # 5 sample sequences of length 200
    sample_entropies = [np.random.uniform(0.5, 4.5, size=200).astype(np.float32) for _ in range(5)]

    target_patch_size = 4.5
    calibrated_theta = calibrate_monotonic_threshold(
        sample_entropies=sample_entropies,
        target_avg_patch_size=target_patch_size,
        tolerance=0.2,
    )

    assert calibrated_theta > 0.0

    # Verify calibrated theta produces average patch size close to target
    total_bytes = 0
    total_patches = 0
    for ent in sample_entropies:
        mask = entropy_patcher_monotonic(ent, theta_r=calibrated_theta)
        total_bytes += len(mask)
        total_patches += mask.sum()

    actual_avg = total_bytes / total_patches
    assert abs(actual_avg - target_patch_size) <= 0.3


def test_streaming_patcher_matches_offline():
    """Verify StreamingPatcher produces identical boundary decisions to offline batch patcher."""
    entropy = np.array([1.0, 1.1, 2.5, 1.0, 3.2, 0.8, 0.9], dtype=np.float32)
    bytes_seq = [ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g')]
    theta_r = 1.0

    offline_mask = entropy_patcher_monotonic(entropy, bytes_seq=bytes_seq, theta_r=theta_r, max_patch_size=8)

    sp = StreamingPatcher(theta_r=theta_r, max_patch_size=8, reset_on_newline=True)
    online_mask = [sp.feed(b, h) for b, h in zip(bytes_seq, entropy)]

    np.testing.assert_array_equal(offline_mask, online_mask)
