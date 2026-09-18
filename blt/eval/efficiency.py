"""Computational & Hardware Efficiency Benchmarking (evaluation.md §12 - §15).

Measures:
- Inference Throughput (bytes/sec) and Latency (§13)
- Time to First Byte (TTFB) (§13)
- Peak GPU Memory (VRAM in MB) (§14)
- FLOPs per Byte and Total GFLOPs (§15)
"""

import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from blt.flops.counter import compute_blt_flops


@torch.no_grad()
def benchmark_inference_efficiency(
    model: nn.Module,
    prompt: str = "Once upon a time there was a brave knight",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Benchmark inference throughput, latency, TTFB, and peak VRAM."""
    model.eval()
    model.to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    prompt_bytes = list(prompt.encode("utf-8"))
    prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

    # 1. Measure Time-to-First-Byte (TTFB)
    t0 = time.perf_counter()
    first_token_out = model.generate(prompt_tensor, max_new_tokens=1, temperature=temperature, top_p=top_p)
    if device == "cuda":
        torch.cuda.synchronize()
    ttfb = time.perf_counter() - t0

    # 2. Measure Full Generation
    t1 = time.perf_counter()
    full_out = model.generate(prompt_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t1

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    generated_bytes = max_new_tokens
    throughput_bytes_sec = generated_bytes / max(1e-6, total_time)
    ms_per_byte = (total_time / generated_bytes) * 1000.0

    return {
        "ttfb_ms": ttfb * 1000.0,
        "total_time_sec": total_time,
        "generated_bytes": generated_bytes,
        "throughput_bytes_sec": throughput_bytes_sec,
        "ms_per_byte": ms_per_byte,
        "peak_vram_mb": peak_vram_mb,
    }


def compute_model_efficiency_summary(
    seq_len_bytes: int = 1024,
    avg_patch_size: float = 4.5,
    byte_dim: int = 256,
    patch_dim: int = 512,
    latent_layers: int = 8,
    peak_vram_mb: float = 0.0,
    inference_throughput: float = 0.0,
) -> Dict[str, Any]:
    """Calculate FLOPs, parameter summary, and memory footprint."""
    flop_stats = compute_blt_flops(
        seq_len_bytes=seq_len_bytes,
        avg_patch_size=avg_patch_size,
        byte_dim=byte_dim,
        patch_dim=patch_dim,
        latent_layers=latent_layers,
    )

    return {
        "seq_len_bytes": seq_len_bytes,
        "avg_patch_size": avg_patch_size,
        "gflops": flop_stats["total_forward_flops"] / 1e9,
        "flops_per_byte": flop_stats["flops_per_byte"],
        "peak_vram_mb": peak_vram_mb,
        "inference_throughput_bytes_sec": inference_throughput,
    }
