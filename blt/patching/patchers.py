"""Multi-strategy patchers implementing the 4 core schemes from BLT paper (§2).

1. Strided Patcher (§2.1): Fixed k-byte stride baseline.
2. Space Patcher (§2.2): Whitespace-delimited words.
3. Entropy Global Patcher (§2.3): Threshold on raw entropy H(x_i) > theta_g.
4. Entropy Monotonic Patcher (§2.3): Delta jump H(x_i) - H(x_{i-1}) > theta_r with context reset.
5. BPE-as-Patches (§2.4): Token boundaries as patches (control comparison).
6. StreamingEntropyPatcher: Stateful streaming patcher for generation loop.
"""

from typing import Union, List, Optional, Callable, Tuple
import numpy as np
import torch
from blt.patching.boundary_rules import monotonic_boundary_rule, global_boundary_rule
from blt.csrc.bridge import NativeStreamingPatcher


def strided_patcher(
    byte_seq: Union[torch.Tensor, np.ndarray, bytes, str],
    k: int = 4,
) -> np.ndarray:
    """Fixed-stride patcher: new patch every k bytes (§2.1)."""
    if isinstance(byte_seq, (str, bytes)):
        seq_len = len(byte_seq.encode("utf-8") if isinstance(byte_seq, str) else byte_seq)
    else:
        seq_len = len(byte_seq)

    mask = np.zeros(seq_len, dtype=np.uint8)
    if seq_len > 0 and k > 0:
        mask[::k] = 1
    return mask


def space_patcher(
    byte_seq: Union[torch.Tensor, np.ndarray, bytes, str],
) -> np.ndarray:
    """Space-aware patcher: starts a new patch after space-like delimiter bytes (§2.2).

    Delimiters include space (32), newline (10), tab (9), carriage return (13).
    """
    if isinstance(byte_seq, str):
        byte_arr = np.frombuffer(byte_seq.encode("utf-8"), dtype=np.uint8)
    elif isinstance(byte_seq, bytes):
        byte_arr = np.frombuffer(byte_seq, dtype=np.uint8)
    elif isinstance(byte_seq, torch.Tensor):
        byte_arr = byte_seq.detach().cpu().numpy().astype(np.uint8)
    else:
        byte_arr = np.ascontiguousarray(byte_seq, dtype=np.uint8)

    seq_len = len(byte_arr)
    mask = np.zeros(seq_len, dtype=np.uint8)
    if seq_len == 0:
        return mask

    mask[0] = 1
    space_bytes = {32, 10, 9, 13}

    for i in range(1, seq_len):
        prev_byte = int(byte_arr[i - 1])
        curr_byte = int(byte_arr[i])
        # Start new patch after whitespace if current byte is non-whitespace
        if prev_byte in space_bytes and curr_byte not in space_bytes:
            mask[i] = 1

    return mask


def entropy_patcher_global(
    entropy: Union[torch.Tensor, np.ndarray],
    theta_g: float = 3.5,
) -> np.ndarray:
    """Global constraint patcher: boundary if H(x_i) > theta_g (§2.3)."""
    mask, _ = global_boundary_rule(entropy, theta_g=theta_g)
    return mask


def entropy_patcher_monotonic(
    entropy: Union[torch.Tensor, np.ndarray],
    bytes_seq: Optional[Union[torch.Tensor, np.ndarray, bytes]] = None,
    theta_r: float = 1.0,
    reset_on_newline: bool = True,
    max_patch_size: int = 32,
) -> np.ndarray:
    """Approx. Monotonic Constraint patcher with context reset (§2.3).

    Accelerated via native C++ DLL with NumPy fallback.
    """
    mask, _ = monotonic_boundary_rule(
        entropy=entropy,
        bytes_seq=bytes_seq,
        theta_r=theta_r,
        reset_on_newline=reset_on_newline,
        max_patch_size=max_patch_size,
    )
    return mask


def bpe_as_patches(
    byte_seq: Union[torch.Tensor, np.ndarray, bytes, str],
    tokenizer=None,
) -> np.ndarray:
    """BPE-as-patches: marks token starts as patch boundaries (§2.4).

    Used to verify the failure of the incremental patching property in BPE.
    """
    if isinstance(byte_seq, str):
        text = byte_seq
        raw_bytes = text.encode("utf-8")
    elif isinstance(byte_seq, bytes):
        raw_bytes = byte_seq
        text = raw_bytes.decode("utf-8", errors="replace")
    elif isinstance(byte_seq, torch.Tensor):
        raw_bytes = bytes(byte_seq.detach().cpu().numpy().tolist())
        text = raw_bytes.decode("utf-8", errors="replace")
    else:
        raw_bytes = bytes(byte_seq.tolist())
        text = raw_bytes.decode("utf-8", errors="replace")

    seq_len = len(raw_bytes)
    mask = np.zeros(seq_len, dtype=np.uint8)
    if seq_len == 0:
        return mask

    mask[0] = 1

    if tokenizer is not None and hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text)
        if hasattr(tokenizer, "count_bytes_per_token"):
            byte_counts = tokenizer.count_bytes_per_token(token_ids)
            cur = 0
            for count in byte_counts:
                if cur < seq_len:
                    mask[cur] = 1
                cur += count
    return mask


class StreamingPatcher:
    """Stateful streaming patcher for the generation hot path."""

    def __init__(
        self,
        theta_r: float = 1.0,
        max_patch_size: int = 16,
        reset_on_newline: bool = True,
        doc_boundary_token: int = 256,
    ):
        self._native = NativeStreamingPatcher(
            theta_r=theta_r,
            max_patch_size=max_patch_size,
            reset_on_newline=reset_on_newline,
            doc_boundary_token=doc_boundary_token,
        )

    def feed(self, byte_val: int, entropy_val: float) -> int:
        """Feed next byte and its estimated entropy. Returns 1 if new patch, 0 if continuation."""
        return self._native.feed(byte_val, entropy_val)

    def current_patch_len(self) -> int:
        return self._native.current_patch_len()
