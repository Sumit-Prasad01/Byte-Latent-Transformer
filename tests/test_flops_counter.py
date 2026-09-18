"""Unit tests for FLOPs counter and complexity metrics."""

import pytest
from blt.flops.counter import (
    count_linear_flops,
    count_swiglu_flops,
    count_attention_flops,
    compute_blt_flops,
    compute_baseline_flops,
)
from blt.flops.report import generate_flop_report


def test_primitive_flops():
    # Linear: 2 * num_tokens * in * out
    # 10 tokens, 16 in, 32 out -> 2 * 10 * 16 * 32 = 10,240
    assert count_linear_flops(16, 32, 10) == 10240

    # SwiGLU: 3 * (2 * 10 * 16 * 64) + (2 * 10 * 64)
    # Proj: 3 * 20480 = 61440, elem: 1280 -> 62720
    assert count_swiglu_flops(16, 64, 10) == 62720

    # Attention: 4 * (2 * 10 * 16 * 16) + 4 * 10 * 10 * 16
    # Proj: 4 * 5120 = 20480, sdpa: 4 * 100 * 16 = 6400 -> 26880
    assert count_attention_flops(16, 10) == 26880


def test_compute_blt_flops():
    res = compute_blt_flops(
        seq_len_bytes=1024,
        avg_patch_size=4.5,
        byte_dim=256,
        patch_dim=512,
        encoder_layers=1,
        latent_layers=8,
        decoder_layers=4,
    )

    assert "total_forward_flops" in res
    assert res["total_forward_flops"] > 0
    assert res["flops_per_byte"] > 0
    assert res["patches_count"] == 228  # ceil(1024 / 4.5) = 228


def test_blt_vs_baseline_efficiency():
    """BLT should use significantly fewer FLOPs than a full-depth stride-1 byte model."""
    seq_len = 2048
    blt = compute_blt_flops(seq_len_bytes=seq_len, avg_patch_size=4.5, byte_dim=256, patch_dim=512)
    base = compute_baseline_flops(seq_len_bytes=seq_len, dim=512, n_layers=8)

    # BLT operates the 8-layer transformer on patches instead of bytes, saving significant FLOPs
    assert blt["total_forward_flops"] < base["total_forward_flops"]
    ratio = base["total_forward_flops"] / blt["total_forward_flops"]
    assert ratio > 1.5, f"Expected BLT to be >1.5x more FLOP efficient, got {ratio:.2f}x"


def test_generate_flop_report():
    report = generate_flop_report(seq_lengths=[512, 1024])
    assert "Sequence (Bytes)" in report
    assert "BLT GFLOPs" in report
    assert "faster" in report
