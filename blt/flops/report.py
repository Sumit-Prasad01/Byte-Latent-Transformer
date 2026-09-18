"""FLOPs comparative reporting table between BLT and Baseline models (Paper §4.1, Table 1)."""

from typing import List, Dict, Any
from blt.flops.counter import compute_blt_flops, compute_baseline_flops


def generate_flop_report(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    avg_patch_size: float = 4.5,
    byte_dim: int = 256,
    patch_dim: int = 512,
    latent_layers: int = 8,
) -> str:
    """Generate markdown table comparing FLOPs between BLT and Byte Baseline."""
    headers = [
        "Sequence (Bytes)",
        "Patches (M)",
        "BLT GFLOPs",
        "BLT FLOPs/Byte",
        "Baseline GFLOPs",
        "Baseline FLOPs/Byte",
        "FLOP Efficiency",
    ]
    rows = []

    for seq_len in seq_lengths:
        blt_stats = compute_blt_flops(
            seq_len_bytes=seq_len,
            avg_patch_size=avg_patch_size,
            byte_dim=byte_dim,
            patch_dim=patch_dim,
            latent_layers=latent_layers,
        )
        base_stats = compute_baseline_flops(
            seq_len_bytes=seq_len,
            dim=patch_dim,
            n_layers=latent_layers,
        )

        blt_gflops = blt_stats["total_forward_flops"] / 1e9
        base_gflops = base_stats["total_forward_flops"] / 1e9
        ratio = base_stats["total_forward_flops"] / blt_stats["total_forward_flops"]

        rows.append([
            f"{seq_len:,}",
            f"{blt_stats['patches_count']:,}",
            f"{blt_gflops:.3f}",
            f"{blt_stats['flops_per_byte']:,.0f}",
            f"{base_gflops:.3f}",
            f"{base_stats['flops_per_byte']:,.0f}",
            f"{ratio:.2f}x faster",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + data_lines)
