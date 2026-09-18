"""Patch boundary rules, monotonicity checks with context resets, and threshold calibration."""

from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from blt.csrc.bridge import monotonic_boundary_mask_native
from blt.data.dataset import DOC_BOUNDARY_TOKEN


def monotonic_boundary_rule(
    entropy: Union[torch.Tensor, np.ndarray],
    bytes_seq: Optional[Union[torch.Tensor, np.ndarray, bytes]] = None,
    theta_r: float = 1.0,
    reset_on_newline: bool = True,
    doc_boundary_token: int = DOC_BOUNDARY_TOKEN,
    max_patch_size: int = 32,
) -> Tuple[np.ndarray, int]:
    """Compute patch boundary mask using the monotonic entropy jump rule (Equation §2.3).

    Accelerated via native C++ DLL (blt_native.dll) with bit-exact NumPy fallback.

    A boundary is placed at position i (out[i] = 1) if:
    1. i == 0
    2. bytes_seq[i] == doc_boundary_token
    3. H(x_i) - H(x_{i-1}) > theta_r
    4. current patch length >= max_patch_size

    Context resets: previous entropy is cleared when bytes_seq[i] is newline ('\n') or doc boundary.

    Returns:
        Tuple of (boundary_mask (uint8 ndarray of shape (seq_len,)), num_patches (int)).
    """
    if isinstance(entropy, torch.Tensor):
        entropy = entropy.detach().cpu().numpy()
    if isinstance(bytes_seq, torch.Tensor):
        bytes_seq = bytes_seq.detach().cpu().numpy()
    elif isinstance(bytes_seq, bytes):
        bytes_seq = np.frombuffer(bytes_seq, dtype=np.uint8)

    return monotonic_boundary_mask_native(
        entropy=entropy,
        bytes_arr=bytes_seq,
        theta_r=theta_r,
        reset_on_newline=reset_on_newline,
        doc_boundary_token=doc_boundary_token,
        max_patch_size=max_patch_size,
    )


def global_boundary_rule(
    entropy: Union[torch.Tensor, np.ndarray],
    theta_g: float = 3.5,
) -> Tuple[np.ndarray, int]:
    """Global threshold boundary rule: boundary if H(x_i) > theta_g or i == 0."""
    if isinstance(entropy, torch.Tensor):
        entropy = entropy.detach().cpu().numpy()

    mask = (entropy > theta_g).astype(np.uint8)
    if len(mask) > 0:
        mask[0] = 1

    return mask, int(mask.sum())


def compute_average_patch_size(boundary_mask: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate average patch size in bytes given a binary boundary mask."""
    if isinstance(boundary_mask, torch.Tensor):
        boundary_mask = boundary_mask.detach().cpu().numpy()

    total_bytes = len(boundary_mask)
    num_patches = int(np.sum(boundary_mask))
    if num_patches == 0:
        return 0.0
    return float(total_bytes) / float(num_patches)


def calibrate_monotonic_threshold(
    sample_entropies: List[np.ndarray],
    sample_bytes: Optional[List[np.ndarray]] = None,
    target_avg_patch_size: float = 4.5,
    tolerance: float = 0.1,
    min_theta: float = 0.1,
    max_theta: float = 6.0,
    max_iter: int = 25,
) -> float:
    """Calibrate theta_r via binary search over calibration sequences to achieve target average patch size."""
    low = min_theta
    high = max_theta
    best_theta = (low + high) / 2.0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        total_bytes = 0
        total_patches = 0

        for i, ent in enumerate(sample_entropies):
            b = sample_bytes[i] if sample_bytes is not None else None
            mask, num_p = monotonic_boundary_rule(ent, bytes_seq=b, theta_r=mid)
            total_bytes += len(mask)
            total_patches += num_p

        current_avg = total_bytes / max(1, total_patches)

        if abs(current_avg - target_avg_patch_size) <= tolerance:
            return mid

        # Higher theta_r means fewer boundaries -> larger patch size
        if current_avg < target_avg_patch_size:
            # Need larger patches -> increase threshold
            low = mid
        else:
            # Need smaller patches -> decrease threshold
            high = mid
        best_theta = mid

    return best_theta
