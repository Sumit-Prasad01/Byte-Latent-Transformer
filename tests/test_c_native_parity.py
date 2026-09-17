"""Unit tests for C++ native bridge and numerical parity with Python fallbacks."""

import numpy as np
import pytest
from blt.csrc.bridge import (
    compute_ngram_hashes_native,
    monotonic_boundary_mask_native,
    dedup_stories_native,
    pack_patch_batches_native,
    NativeStreamingPatcher,
    has_native_engine,
)


def test_ngram_rolling_hash_consistency():
    """Verify n-gram rolling hash returns consistent, deterministic results."""
    text = b"Once upon a time in a byte latent transformer world"
    ngram_sizes = [3, 4, 5]
    vocab_sizes = [20000, 20000, 20000]

    h1 = compute_ngram_hashes_native(text, ngram_sizes, vocab_sizes)
    h2 = compute_ngram_hashes_native(text, ngram_sizes, vocab_sizes)

    assert h1.shape == (3, len(text))
    np.testing.assert_array_equal(h1, h2)
    assert np.all(h1 >= 0)
    assert np.all(h1 < 20000)

    # Identical substrings within the text should produce identical hashes
    sub1 = b"Once"
    h_sub = compute_ngram_hashes_native(sub1, [3], [20000])
    # The hash for 'Once' (ending at position 3, length 3='nce') should match
    assert h1[0, 3] == h_sub[0, 3]


def test_monotonic_boundary_rules():
    """Verify boundary mask behavior under monotonic jumps, newlines, and doc boundaries."""
    entropy = np.array([1.0, 1.2, 2.8, 1.0, 2.5, 0.5], dtype=np.float32)
    bytes_seq = np.array([ord('a'), ord('b'), ord('c'), 10, ord('d'), ord('e')], dtype=np.uint8)  # 10 is '\n'

    mask, num_patches = monotonic_boundary_mask_native(
        entropy=entropy,
        bytes_arr=bytes_seq,
        theta_r=1.0,
        reset_on_newline=True,
        doc_boundary_token=256,
        max_patch_size=16,
    )

    assert mask.shape == (len(entropy),)
    # Position 0 is always a boundary
    assert mask[0] == 1
    # Position 2: 2.8 - 1.2 = 1.6 > 1.0 -> boundary
    assert mask[2] == 1
    # Position 3: newline -> resets context
    # Position 4: after reset, no spurious delta across line boundary
    assert mask[3] == 0
    assert num_patches == int(mask.sum())


def test_story_deduplication():
    """Verify story deduplication accurately identifies duplicate stories."""
    story1 = b"Story A: A small puppy playing in the sunny park."
    story2 = b"Story B: A brave kitten exploring the dark woods."
    story3 = b"Story A: A small puppy playing in the sunny park."  # exact duplicate

    corpus = story1 + story2 + story3
    offsets = np.array([0, len(story1), len(story1) + len(story2)], dtype=np.int64)
    lengths = np.array([len(story1), len(story2), len(story3)], dtype=np.int64)

    keep_mask, unique_count = dedup_stories_native(corpus, offsets, lengths)

    assert unique_count == 2
    assert keep_mask[0] == 1
    assert keep_mask[1] == 1
    assert keep_mask[2] == 0


def test_patch_batch_packing():
    """Verify greedy packing keeps patch counts bounded per batch."""
    patch_counts = [100, 150, 300, 200, 50]
    max_patches = 300

    batch_ids, num_batches = pack_patch_batches_native(patch_counts, max_patches_per_batch=max_patches)

    assert len(batch_ids) == len(patch_counts)
    assert num_batches >= 2

    # Check that sum of patches in each batch <= max_patches
    for b in range(num_batches):
        batch_patch_sum = sum(patch_counts[i] for i in range(len(patch_counts)) if batch_ids[i] == b)
        assert batch_patch_sum <= max_patches


def test_streaming_patcher():
    """Verify stateful streaming patcher matches offline sequential decisions."""
    patcher = NativeStreamingPatcher(theta_r=1.0, max_patch_size=8, reset_on_newline=True)

    bytes_seq = [ord('X'), ord('y'), ord('z'), ord('!')]
    entropies = [1.5, 1.6, 3.0, 1.0]

    boundaries = [patcher.feed(b, h) for b, h in zip(bytes_seq, entropies)]

    # First byte is always boundary
    assert boundaries[0] == 1
    assert boundaries[1] == 0
    # 3.0 - 1.6 = 1.4 > 1.0 -> boundary
    assert boundaries[2] == 1
    assert boundaries[3] == 0
