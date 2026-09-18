"""Evaluation metrics, robustness suites, patch statistics, and efficiency benchmarks."""

from blt.eval.bpb import (
    compute_bpb,
    evaluate_byte_model_bpb,
    evaluate_bpe_model_bpb,
)
from blt.eval.metrics import (
    loss_to_bpb,
    loss_to_ppl,
    compute_byte_accuracy,
    LanguageModelMetricTracker,
)
from blt.eval.patch_metrics import (
    compute_patch_lengths_from_boundaries,
    compute_patch_statistics,
    compute_entropy_patch_correlation,
    analyze_patch_boundaries,
)
from blt.eval.robustness import (
    evaluate_robustness_suite,
    format_robustness_table,
    get_curated_robustness_data,
    corrupt_antspeak,
    corrupt_drop,
    corrupt_random_case,
    corrupt_repeat,
    corrupt_uppercase,
)
from blt.eval.efficiency import (
    benchmark_inference_efficiency,
    compute_model_efficiency_summary,
)

__all__ = [
    "compute_bpb",
    "evaluate_byte_model_bpb",
    "evaluate_bpe_model_bpb",
    "loss_to_bpb",
    "loss_to_ppl",
    "compute_byte_accuracy",
    "LanguageModelMetricTracker",
    "compute_patch_lengths_from_boundaries",
    "compute_patch_statistics",
    "compute_entropy_patch_correlation",
    "analyze_patch_boundaries",
    "evaluate_robustness_suite",
    "format_robustness_table",
    "get_curated_robustness_data",
    "corrupt_antspeak",
    "corrupt_drop",
    "corrupt_random_case",
    "corrupt_repeat",
    "corrupt_uppercase",
    "benchmark_inference_efficiency",
    "compute_model_efficiency_summary",
]
