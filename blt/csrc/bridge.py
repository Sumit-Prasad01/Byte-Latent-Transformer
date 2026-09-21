"""Ctypes bridge to BLT C++ native dynamic library (blt_native.dll).

Provides zero-copy direct array passing between Python/NumPy and native C++
kernels, with automatic, bit-exact fallback to optimized NumPy/Python if the
native library is uncompiled or unavailable.
"""

import os
import ctypes
from typing import List, Tuple, Optional, Union
import numpy as np
from utils.logger import get_logger

logger = get_logger("BLT.NativeBridge")

_DLL_HANDLE: Optional[ctypes.CDLL] = None
_HAS_NATIVE: bool = False

# Candidate locations for blt_native.dll
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CANDIDATE_PATHS = [
    os.path.join(_REPO_ROOT, "build", "csrc", "bin", "blt_native.dll"),
    os.path.join(_REPO_ROOT, "blt", "csrc", "blt_native.dll"),
    os.path.join(_REPO_ROOT, "csrc", "blt_native.dll"),
    "blt_native.dll",
]


def _init_native_library() -> bool:
    """Attempt to find and load blt_native.dll via ctypes."""
    global _DLL_HANDLE, _HAS_NATIVE

    if _DLL_HANDLE is not None:
        return _HAS_NATIVE

    dll_path = None
    for p in _CANDIDATE_PATHS:
        if os.path.isabs(p) and os.path.exists(p):
            dll_path = p
            break

    if not dll_path:
        # Check relative or PATH
        for p in _CANDIDATE_PATHS:
            if not os.path.isabs(p):
                try:
                    _DLL_HANDLE = ctypes.CDLL(p)
                    dll_path = p
                    break
                except Exception:
                    continue

    if dll_path and not _DLL_HANDLE:
        try:
            _DLL_HANDLE = ctypes.CDLL(dll_path)
        except OSError as e:
            logger.warning(f"Found native DLL at {dll_path} but could not load: {e}")
            _HAS_NATIVE = False
            return False

    if _DLL_HANDLE:
        try:
            # Configure signatures
            _DLL_HANDLE.blt_sanity_check.restype = ctypes.c_int32
            _DLL_HANDLE.blt_sanity_check.argtypes = []

            _DLL_HANDLE.blt_get_version.restype = ctypes.c_char_p
            _DLL_HANDLE.blt_get_version.argtypes = []

            # Rolling hash
            _DLL_HANDLE.blt_roll_hash_single.restype = ctypes.c_int64
            _DLL_HANDLE.blt_roll_hash_single.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_uint64,
                ctypes.c_int64,
            ]

            _DLL_HANDLE.blt_compute_ngram_hashes.restype = ctypes.c_int32
            _DLL_HANDLE.blt_compute_ngram_hashes.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_size_t,
                ctypes.c_uint64,
                ctypes.POINTER(ctypes.c_int64),
            ]

            # Boundary rules
            _DLL_HANDLE.blt_monotonic_boundary_mask.restype = ctypes.c_int32
            _DLL_HANDLE.blt_monotonic_boundary_mask.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int64),
            ]

            # Story dedup
            _DLL_HANDLE.blt_hash_bytes.restype = ctypes.c_uint64
            _DLL_HANDLE.blt_hash_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

            _DLL_HANDLE.blt_dedup_story_offsets.restype = ctypes.c_int32
            _DLL_HANDLE.blt_dedup_story_offsets.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int64),
            ]

            # Streaming patcher
            _DLL_HANDLE.blt_streaming_patcher_create.restype = ctypes.c_void_p
            _DLL_HANDLE.blt_streaming_patcher_create.argtypes = [
                ctypes.c_float,
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_int32,
            ]

            _DLL_HANDLE.blt_streaming_patcher_feed.restype = ctypes.c_int32
            _DLL_HANDLE.blt_streaming_patcher_feed.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int32,
                ctypes.c_float,
            ]

            _DLL_HANDLE.blt_streaming_patcher_get_current_patch_len.restype = ctypes.c_int32
            _DLL_HANDLE.blt_streaming_patcher_get_current_patch_len.argtypes = [ctypes.c_void_p]

            _DLL_HANDLE.blt_streaming_patcher_destroy.restype = None
            _DLL_HANDLE.blt_streaming_patcher_destroy.argtypes = [ctypes.c_void_p]

            # Batch packing
            _DLL_HANDLE.blt_pack_patch_batches_c.restype = ctypes.c_int32
            _DLL_HANDLE.blt_pack_patch_batches_c.argtypes = [
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
            ]

            sanity = _DLL_HANDLE.blt_sanity_check()
            if sanity == 42:
                version = _DLL_HANDLE.blt_get_version().decode("utf-8")
                logger.info(f"Loaded BLT native C++ engine ({version}) from: {dll_path}")
                _HAS_NATIVE = True
                return True
        except Exception as e:
            logger.warning(f"Error initializing native C++ functions: {e}")
            _HAS_NATIVE = False
            return False

    logger.info("BLT native C++ DLL not loaded; running with high-performance NumPy/Python fallback.")
    _HAS_NATIVE = False
    return False


# Initialize on module import
_init_native_library()


def has_native_engine() -> bool:
    """Check if native C++ engine (.dll) is active."""
    return _HAS_NATIVE


def get_native_engine() -> Optional[ctypes.CDLL]:
    """Retrieve raw ctypes handle to blt_native.dll."""
    return _DLL_HANDLE


# =========================================================================
# 1. RollPolyHash n-gram embedding calculation
# =========================================================================

def compute_ngram_hashes_native(
    byte_seq: Union[bytes, np.ndarray],
    ngram_sizes: List[int],
    vocab_sizes: List[int],
    prime: int = 31337,
) -> np.ndarray:
    """Compute rolling polynomial hash for multiple n-gram sizes over a byte sequence.

    Args:
        byte_seq: Array or bytes sequence of length seq_len.
        ngram_sizes: List of n-gram lengths (e.g. [3, 4, 5]).
        vocab_sizes: List of bucket counts for each n-gram size.
        prime: Polynomial base prime (default 31337).

    Returns:
        np.ndarray of shape (num_sizes, seq_len) with int64 bucket IDs.
    """
    if isinstance(byte_seq, bytes):
        bytes_arr = np.frombuffer(byte_seq, dtype=np.uint8)
    elif isinstance(byte_seq, np.ndarray):
        bytes_arr = np.ascontiguousarray(byte_seq, dtype=np.uint8)
    else:
        bytes_arr = np.array(byte_seq, dtype=np.uint8)

    seq_len = len(bytes_arr)
    num_sizes = len(ngram_sizes)
    if seq_len == 0 or num_sizes == 0:
        return np.zeros((num_sizes, seq_len), dtype=np.int64)

    ngram_sizes_arr = np.ascontiguousarray(ngram_sizes, dtype=np.int32)
    vocab_sizes_arr = np.ascontiguousarray(vocab_sizes, dtype=np.int64)

    if _HAS_NATIVE and _DLL_HANDLE is not None:
        out_hashes = np.empty((num_sizes, seq_len), dtype=np.int64)
        status = _DLL_HANDLE.blt_compute_ngram_hashes(
            bytes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(seq_len),
            ngram_sizes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            vocab_sizes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(num_sizes),
            ctypes.c_uint64(prime),
            out_hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        if status == 0:
            return out_hashes

    # Python / NumPy Fallback (Bit-Exact)
    out_hashes = np.zeros((num_sizes, seq_len), dtype=np.int64)
    for s_idx, (n, v) in enumerate(zip(ngram_sizes, vocab_sizes)):
        p_pow = np.array([pow(prime, j, v) for j in range(n)], dtype=np.int64)
        for i in range(seq_len):
            avail = min(i + 1, n)
            window = bytes_arr[i - avail + 1 : i + 1][::-1]  # b_i, b_{i-1}, ...
            h = 0
            for j in range(avail):
                h = (h + int(window[j]) * int(p_pow[j])) % v
            out_hashes[s_idx, i] = h

    return out_hashes


# =========================================================================
# 2. Boundary rules (monotonicity rule + context reset)
# =========================================================================

def monotonic_boundary_mask_native(
    entropy: np.ndarray,
    bytes_arr: Optional[np.ndarray] = None,
    theta_r: float = 1.0,
    reset_on_newline: bool = True,
    doc_boundary_token: int = 256,
    max_patch_size: int = 32,
) -> Tuple[np.ndarray, int]:
    """Compute patch boundary mask using the monotonic entropy jump rule.

    Returns:
        Tuple of (boundary_mask (uint8 array), num_patches (int)).
    """
    entropy = np.ascontiguousarray(entropy, dtype=np.float32)

    # Support 2D batched input: (batch_size, seq_len)
    if entropy.ndim == 2:
        batch_size, seq_len = entropy.shape
        masks = np.empty((batch_size, seq_len), dtype=np.uint8)
        total_patches = 0
        for b in range(batch_size):
            b_bytes = bytes_arr[b] if bytes_arr is not None else None
            m, p = monotonic_boundary_mask_native(
                entropy[b],
                bytes_arr=b_bytes,
                theta_r=theta_r,
                reset_on_newline=reset_on_newline,
                doc_boundary_token=doc_boundary_token,
                max_patch_size=max_patch_size,
            )
            masks[b] = m
            total_patches += p
        return masks, total_patches

    entropy = np.asarray(entropy, dtype=np.float32).reshape(-1)
    seq_len = len(entropy)
    if seq_len == 0:
        return np.zeros(0, dtype=np.uint8), 0

    if bytes_arr is not None:
        raw_bytes_for_py = np.asarray(bytes_arr).reshape(-1)
        bytes_arr_uint8 = np.ascontiguousarray(np.clip(raw_bytes_for_py, 0, 255), dtype=np.uint8)
    else:
        raw_bytes_for_py = None
        bytes_arr_uint8 = None

    if _HAS_NATIVE and _DLL_HANDLE is not None:
        out_boundaries = np.empty(seq_len, dtype=np.uint8)
        num_patches = ctypes.c_int64(0)

        bytes_ptr = bytes_arr_uint8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if bytes_arr_uint8 is not None else None

        status = _DLL_HANDLE.blt_monotonic_boundary_mask(
            entropy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bytes_ptr,
            ctypes.c_size_t(seq_len),
            ctypes.c_float(theta_r),
            ctypes.c_int32(1 if reset_on_newline else 0),
            ctypes.c_int32(doc_boundary_token),
            ctypes.c_int32(max_patch_size),
            out_boundaries.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(num_patches),
        )
        if status == 0:
            return out_boundaries, num_patches.value

    # Python Fallback (Bit-Exact)
    out_boundaries = np.zeros(seq_len, dtype=np.uint8)
    patch_count = 0
    prev_entropy = 0.0
    has_prev = False
    current_patch_len = 0

    for i in range(seq_len):
        is_boundary = False
        byte_val = int(raw_bytes_for_py[i]) if raw_bytes_for_py is not None else 0

        if i == 0:
            is_boundary = True
        elif raw_bytes_for_py is not None and doc_boundary_token >= 0 and byte_val == doc_boundary_token:
            is_boundary = True
            has_prev = False
        elif max_patch_size > 0 and current_patch_len >= max_patch_size:
            is_boundary = True
        elif has_prev and (entropy[i] - prev_entropy > theta_r):
            is_boundary = True

        if is_boundary:
            out_boundaries[i] = 1
            patch_count += 1
            current_patch_len = 1
        else:
            out_boundaries[i] = 0
            current_patch_len += 1

        if reset_on_newline and raw_bytes_for_py is not None and byte_val == 10:  # ord('\n') == 10
            has_prev = False
        else:
            prev_entropy = float(entropy[i])
            has_prev = True

    return out_boundaries, patch_count


# =========================================================================
# 3. Story deduplication
# =========================================================================

def dedup_stories_native(
    corpus_bytes: bytes,
    story_offsets: np.ndarray,
    story_lengths: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Deduplicate stories given offsets in a contiguous byte corpus.

    Returns:
        Tuple of (keep_mask (int8 array), unique_count (int)).
    """
    corpus_arr = np.frombuffer(corpus_bytes, dtype=np.uint8)
    num_stories = len(story_offsets)
    if num_stories == 0:
        return np.zeros(0, dtype=np.int8), 0

    story_offsets = np.ascontiguousarray(story_offsets, dtype=np.int64)
    story_lengths = np.ascontiguousarray(story_lengths, dtype=np.int64)

    if _HAS_NATIVE and _DLL_HANDLE is not None:
        keep_mask = np.empty(num_stories, dtype=np.int8)
        unique_count = ctypes.c_int64(0)
        status = _DLL_HANDLE.blt_dedup_story_offsets(
            corpus_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            story_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            story_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(num_stories),
            keep_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            ctypes.byref(unique_count),
        )
        if status == 0:
            return keep_mask, unique_count.value

    # Python Fallback
    seen = set()
    keep_mask = np.zeros(num_stories, dtype=np.int8)
    unique_count = 0
    for i in range(num_stories):
        off = story_offsets[i]
        ln = story_lengths[i]
        if off < 0 or ln <= 0:
            keep_mask[i] = 0
            continue
        story_chunk = bytes(corpus_arr[off : off + ln])
        h = hash(story_chunk)
        if h not in seen:
            seen.add(h)
            keep_mask[i] = 1
            unique_count += 1
        else:
            keep_mask[i] = 0

    return keep_mask, unique_count


# =========================================================================
# 4. Batch packing
# =========================================================================

def pack_patch_batches_native(
    patch_counts: List[int],
    max_patches_per_batch: int = 512,
) -> Tuple[np.ndarray, int]:
    """Greedy batch assignment balancing patch counts.

    Returns:
        Tuple of (batch_ids (int32 array), num_batches (int)).
    """
    num_seqs = len(patch_counts)
    if num_seqs == 0:
        return np.zeros(0, dtype=np.int32), 0

    patch_counts_arr = np.ascontiguousarray(patch_counts, dtype=np.int32)

    if _HAS_NATIVE and _DLL_HANDLE is not None:
        batch_ids = np.empty(num_seqs, dtype=np.int32)
        num_batches = ctypes.c_int32(0)
        status = _DLL_HANDLE.blt_pack_patch_batches_c(
            patch_counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(num_seqs),
            ctypes.c_int32(max_patches_per_batch),
            batch_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(num_batches),
        )
        if status == 0:
            return batch_ids, num_batches.value

    # Python Fallback
    batch_ids = np.zeros(num_seqs, dtype=np.int32)
    cur_id = 0
    cur_patches = 0
    for i in range(num_seqs):
        p = max(1, patch_counts[i])
        if cur_patches + p > max_patches_per_batch and cur_patches > 0:
            cur_id += 1
            cur_patches = 0
        batch_ids[i] = cur_id
        cur_patches += p

    return batch_ids, cur_id + 1


# =========================================================================
# 5. Stateful streaming patcher for generation
# =========================================================================

class NativeStreamingPatcher:
    """Stateful streaming patcher for autoregressive generation."""

    def __init__(
        self,
        theta_r: float = 0.8,
        max_patch_size: int = 16,
        reset_on_newline: bool = True,
        doc_boundary_token: int = 256,
    ):
        self.theta_r = theta_r
        self.max_patch_size = max_patch_size
        self.reset_on_newline = reset_on_newline
        self.doc_boundary_token = doc_boundary_token

        self._handle = None
        if _HAS_NATIVE and _DLL_HANDLE is not None:
            self._handle = _DLL_HANDLE.blt_streaming_patcher_create(
                ctypes.c_float(theta_r),
                ctypes.c_int32(max_patch_size),
                ctypes.c_int32(1 if reset_on_newline else 0),
                ctypes.c_int32(doc_boundary_token),
            )

        # Python fallback state
        self._prev_entropy = 0.0
        self._has_prev = False
        self._current_patch_len = 0
        self._total_bytes = 0

    def feed(self, byte_val: int, entropy_val: float) -> int:
        """Feed a generated byte and its entropy, returns 1 if new patch, 0 if continuation."""
        if self._handle and _DLL_HANDLE is not None:
            return _DLL_HANDLE.blt_streaming_patcher_feed(
                self._handle, ctypes.c_int32(byte_val), ctypes.c_float(entropy_val)
            )

        # Python Fallback
        is_boundary = False
        if self._total_bytes == 0:
            is_boundary = True
        elif self.doc_boundary_token >= 0 and byte_val == self.doc_boundary_token:
            is_boundary = True
            self._has_prev = False
        elif self.max_patch_size > 0 and self._current_patch_len >= self.max_patch_size:
            is_boundary = True
        elif self._has_prev and (entropy_val - self._prev_entropy > self.theta_r):
            is_boundary = True

        if is_boundary:
            self._current_patch_len = 1
        else:
            self._current_patch_len += 1

        if self.reset_on_newline and byte_val == 10:
            self._has_prev = False
        else:
            self._prev_entropy = entropy_val
            self._has_prev = True

        self._total_bytes += 1
        return 1 if is_boundary else 0

    def current_patch_len(self) -> int:
        if self._handle and _DLL_HANDLE is not None:
            return _DLL_HANDLE.blt_streaming_patcher_get_current_patch_len(self._handle)
        return self._current_patch_len

    def __del__(self):
        if self._handle and _DLL_HANDLE is not None:
            _DLL_HANDLE.blt_streaming_patcher_destroy(self._handle)
            self._handle = None


def streaming_patcher_feed_native(
    byte_seq: Union[bytes, List[int], np.ndarray],
    entropy_seq: Union[np.ndarray, List[float]],
    theta_r: float = 0.8,
    max_patch_size: int = 16,
    reset_on_newline: bool = True,
    doc_boundary_token: int = 256,
) -> List[int]:
    """Feed a sequence of bytes and entropies into a streaming patcher and collect boundary decisions."""
    patcher = NativeStreamingPatcher(
        theta_r=theta_r,
        max_patch_size=max_patch_size,
        reset_on_newline=reset_on_newline,
        doc_boundary_token=doc_boundary_token,
    )
    return [patcher.feed(int(b), float(h)) for b, h in zip(byte_seq, entropy_seq)]

