"""Patch Compression, Statistics, and Entropy Correlation (evaluation.md §8 - §11, §16).

Implements:
- Patch Length Statistics (Mean, Median, Std, P25, P75, Min, Max) (§9)
- Compression Ratio and Latent Ratio (§8, §16)
- Local Byte Entropy vs Patch Length Correlation (§10)
- Patch Boundary Context Analysis (§11)
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def compute_patch_lengths_from_boundaries(
    boundaries: Union[torch.Tensor, np.ndarray],
) -> List[int]:
    """Compute list of patch lengths from a binary boundary mask."""
    if isinstance(boundaries, torch.Tensor):
        b_np = boundaries.detach().cpu().numpy()
    else:
        b_np = np.asarray(boundaries)

    if b_np.ndim == 2:
        lengths = []
        for row in b_np:
            lengths.extend(compute_patch_lengths_from_boundaries(row))
        return lengths

    # 1D array
    indices = np.where(b_np > 0)[0]
    if len(indices) == 0:
        return [len(b_np)] if len(b_np) > 0 else []

    if indices[0] != 0:
        indices = np.insert(indices, 0, 0)

    # Patch lengths are diffs between consecutive boundary positions
    starts = indices
    ends = np.append(indices[1:], len(b_np))
    patch_lengths = (ends - starts).tolist()
    return patch_lengths


def compute_patch_statistics(
    boundaries_or_lengths: Union[List[int], torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Compute detailed distribution statistics of patch lengths (§9, §16).

    Args:
        boundaries_or_lengths: Either a list of patch lengths, or binary boundaries array.

    Returns:
        Dictionary containing mean, median, std, min, max, p25, p75, compression ratio, latent ratio.
    """
    if isinstance(boundaries_or_lengths, list) and all(isinstance(x, (int, float, np.integer)) for x in boundaries_or_lengths):
        lengths = np.array(boundaries_or_lengths, dtype=np.float64)
    else:
        lens = compute_patch_lengths_from_boundaries(boundaries_or_lengths)
        lengths = np.array(lens, dtype=np.float64)

    if len(lengths) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "p25": 0.0,
            "p75": 0.0,
            "total_bytes": 0,
            "total_patches": 0,
            "compression_ratio": 1.0,
            "latent_ratio": 1.0,
        }

    total_bytes = int(lengths.sum())
    total_patches = len(lengths)
    compression_ratio = total_bytes / total_patches
    latent_ratio = total_patches / total_bytes

    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "total_bytes": total_bytes,
        "total_patches": total_patches,
        "compression_ratio": compression_ratio,
        "latent_ratio": latent_ratio,
    }


def compute_entropy_patch_correlation(
    local_entropies: Union[torch.Tensor, np.ndarray],
    patch_boundaries: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Calculate Pearson correlation between local byte entropy and patch length (§10).

    In an information-dense patching scheme, high-entropy bytes should produce shorter
    patches (negative correlation: r < 0).
    """
    if isinstance(local_entropies, torch.Tensor):
        H = local_entropies.detach().cpu().numpy().flatten()
    else:
        H = np.asarray(local_entropies).flatten()

    if isinstance(patch_boundaries, torch.Tensor):
        B = patch_boundaries.detach().cpu().numpy().flatten()
    else:
        B = np.asarray(patch_boundaries).flatten()

    # Assign each byte the length of the patch it belongs to
    indices = np.where(B > 0)[0]
    if len(indices) == 0:
        indices = np.array([0])
    elif indices[0] != 0:
        indices = np.insert(indices, 0, 0)

    ends = np.append(indices[1:], len(B))
    patch_lengths_per_byte = np.zeros(len(B), dtype=np.float64)

    for start, end in zip(indices, ends):
        length = end - start
        patch_lengths_per_byte[start:end] = length

    # Pearson correlation coefficient
    valid_len = min(len(H), len(patch_lengths_per_byte))
    H_val = H[:valid_len]
    L_val = patch_lengths_per_byte[:valid_len]

    std_H = np.std(H_val)
    std_L = np.std(L_val)

    if std_H > 1e-8 and std_L > 1e-8:
        corr_matrix = np.corrcoef(H_val, L_val)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = 0.0

    return {
        "pearson_correlation": pearson_r,
        "mean_entropy": float(np.mean(H_val)),
        "mean_patch_length": float(np.mean(L_val)),
    }


def analyze_patch_boundaries(
    bytes_seq: Union[bytes, str, torch.Tensor, np.ndarray],
    patch_boundaries: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Analyze character types at detected patch boundary positions (§11).

    Measures fraction of boundaries occurring at:
    - Whitespace (spaces, newlines, tabs)
    - Punctuation
    - Alphanumeric characters
    - Non-ASCII / Unicode bytes
    """
    if isinstance(bytes_seq, str):
        raw_bytes = list(bytes_seq.encode("utf-8"))
    elif isinstance(bytes_seq, bytes):
        raw_bytes = list(bytes_seq)
    elif isinstance(bytes_seq, torch.Tensor):
        raw_bytes = bytes_seq.detach().cpu().numpy().flatten().tolist()
    else:
        raw_bytes = list(bytes_seq.flatten())

    if isinstance(patch_boundaries, torch.Tensor):
        b_np = patch_boundaries.detach().cpu().numpy().flatten()
    else:
        b_np = np.asarray(patch_boundaries).flatten()

    boundary_indices = [i for i in range(min(len(raw_bytes), len(b_np))) if b_np[i] > 0]
    total_boundaries = len(boundary_indices)

    if total_boundaries == 0:
        return {
            "whitespace_pct": 0.0,
            "punctuation_pct": 0.0,
            "alphanumeric_pct": 0.0,
            "unicode_pct": 0.0,
            "total_boundaries": 0,
        }

    whitespace_count = 0
    punctuation_count = 0
    alphanumeric_count = 0
    unicode_count = 0

    whitespace_bytes = {32, 10, 9, 13}
    # ASCII punctuation: 33-47, 58-64, 91-96, 123-126
    punct_bytes = set(range(33, 48)) | set(range(58, 65)) | set(range(91, 97)) | set(range(123, 127))

    for idx in boundary_indices:
        b = int(raw_bytes[idx])
        if b in whitespace_bytes:
            whitespace_count += 1
        elif b in punct_bytes:
            punctuation_count += 1
        elif (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122):
            alphanumeric_count += 1
        elif b >= 128:
            unicode_count += 1

    return {
        "whitespace_pct": (whitespace_count / total_boundaries) * 100.0,
        "punctuation_pct": (punctuation_count / total_boundaries) * 100.0,
        "alphanumeric_pct": (alphanumeric_count / total_boundaries) * 100.0,
        "unicode_pct": (unicode_count / total_boundaries) * 100.0,
        "total_boundaries": total_boundaries,
    }
