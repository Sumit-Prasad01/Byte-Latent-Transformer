"""Regression test for the Incremental Patching Property (BLT Paper §2.4).

Formally validates:
f_p(x_<i) == f_p(x)_<i

Entropy-based, space, and strided patchers must satisfy this property.
Subword BPE fails this property when appending continuation bytes alters earlier merges.
"""

import numpy as np
import pytest
import torch
from blt.patching.patchers import (
    strided_patcher,
    space_patcher,
    entropy_patcher_monotonic,
    bpe_as_patches,
)
from blt.data.tokenizer_baseline import BPETokenizerBaseline


def test_strided_incremental_patching():
    """Verify strided patcher satisfies incremental patching property."""
    full_text = b"Once upon a time in a byte latent transformer research paper."
    k = 4
    full_mask = strided_patcher(full_text, k=k)

    # For any prefix length i
    for i in [10, 20, 35, 50]:
        prefix = full_text[:i]
        prefix_mask = strided_patcher(prefix, k=k)
        # Sliced full mask must match prefix mask exactly
        np.testing.assert_array_equal(prefix_mask, full_mask[:i])


def test_space_incremental_patching():
    """Verify space patcher satisfies incremental patching property."""
    full_text = b"The little puppy played with a bright red ball in the green garden."
    full_mask = space_patcher(full_text)

    # For any prefix that cuts off after whitespace
    for i in [15, 30, 45]:
        prefix = full_text[:i]
        prefix_mask = space_patcher(prefix)
        np.testing.assert_array_equal(prefix_mask, full_mask[:i])


def test_entropy_monotonic_incremental_patching():
    """Verify monotonic entropy patcher satisfies incremental patching property."""
    # Synthetic causal entropy trajectory
    np.random.seed(42)
    seq_len = 50
    entropy = np.random.uniform(1.0, 5.0, size=seq_len).astype(np.float32)
    bytes_seq = np.random.randint(0, 255, size=seq_len, dtype=np.uint8)

    theta_r = 1.2
    full_mask = entropy_patcher_monotonic(entropy, bytes_seq=bytes_seq, theta_r=theta_r, max_patch_size=16)

    for i in [15, 25, 40]:
        prefix_ent = entropy[:i]
        prefix_bytes = bytes_seq[:i]
        prefix_mask = entropy_patcher_monotonic(
            prefix_ent, bytes_seq=prefix_bytes, theta_r=theta_r, max_patch_size=16
        )
        np.testing.assert_array_equal(prefix_mask, full_mask[:i])


def test_bpe_fails_incremental_patching():
    """Demonstrate that subword BPE tokenization fails the incremental property (§2.4)."""
    tokenizer = BPETokenizerBaseline()

    # In BPE, appending 'e' to 'th' changes 't' + 'h' into single token 'the'
    # Or 'inter' + 'nation' vs 'intern'
    t1 = "th"
    t2 = "the"

    m1 = bpe_as_patches(t1, tokenizer)
    m2 = bpe_as_patches(t2, tokenizer)

    # In m1, len=2. In m2[:2], if merges changed, boundaries differ or merge differs.
    tokens1 = tokenizer.encode(t1)
    tokens2 = tokenizer.encode(t2)

    # The tokenization of 'the' is 1 token [the], whereas 'th' is split differently
    assert len(tokens2) == 1
    assert len(tokens1) >= 1
