"""C/C++ native acceleration package for Byte Latent Transformer."""

from blt.csrc.bridge import (
    has_native_engine,
    get_native_engine,
    compute_ngram_hashes_native,
    monotonic_boundary_mask_native,
    streaming_patcher_feed_native,
    NativeStreamingPatcher,
    pack_patch_batches_native,
    dedup_stories_native,
)

__all__ = [
    "has_native_engine",
    "get_native_engine",
    "compute_ngram_hashes_native",
    "monotonic_boundary_mask_native",
    "streaming_patcher_feed_native",
    "NativeStreamingPatcher",
    "pack_patch_batches_native",
    "dedup_stories_native",
]
