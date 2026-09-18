"""Patching mechanisms and entropy estimation for Byte Latent Transformer."""

from blt.patching.entropy_model import ByteEntropyModel
from blt.patching.boundary_rules import (
    monotonic_boundary_rule,
    global_boundary_rule,
    calibrate_monotonic_threshold,
    compute_average_patch_size,
)
from blt.patching.patchers import (
    strided_patcher,
    space_patcher,
    entropy_patcher_global,
    entropy_patcher_monotonic,
    bpe_as_patches,
    StreamingPatcher,
)

__all__ = [
    "ByteEntropyModel",
    "monotonic_boundary_rule",
    "global_boundary_rule",
    "calibrate_monotonic_threshold",
    "compute_average_patch_size",
    "strided_patcher",
    "space_patcher",
    "entropy_patcher_global",
    "entropy_patcher_monotonic",
    "bpe_as_patches",
    "StreamingPatcher",
]
