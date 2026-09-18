"""Unit tests for evaluation metrics, patch statistics, and robustness suites."""

import math
import pytest
import torch
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
    corrupt_antspeak,
    corrupt_drop,
    corrupt_random_case,
    corrupt_repeat,
    corrupt_uppercase,
    get_curated_robustness_data,
    evaluate_robustness_suite,
    format_robustness_table,
)
from blt.model.baseline_bpe_model import BaselineTransformer


def test_lm_metric_conversions():
    loss = 2.0
    bpb = loss_to_bpb(loss)
    assert abs(bpb - (2.0 / math.log(2))) < 1e-6

    ppl = loss_to_ppl(loss)
    assert abs(ppl - math.exp(2.0)) < 1e-6


def test_compute_byte_accuracy():
    # 4 positions: predictions [65, 66, 67, 68], targets [65, 66, 100, 68]
    # Position 3 target is 100 vs pred 68 -> 3 out of 4 correct
    logits = torch.zeros(1, 4, 260)
    logits[0, 0, 65] = 10.0
    logits[0, 1, 66] = 10.0
    logits[0, 2, 67] = 10.0
    logits[0, 3, 68] = 10.0

    targets = torch.tensor([[65, 66, 100, 68]], dtype=torch.long)
    correct, total = compute_byte_accuracy(logits, targets)
    assert correct == 3
    assert total == 4

    # Test ignore index
    targets_with_ignore = torch.tensor([[65, -100, 100, 68]], dtype=torch.long)
    c, tot = compute_byte_accuracy(logits, targets_with_ignore, ignore_index=-100)
    assert c == 2
    assert tot == 3


def test_language_model_metric_tracker():
    tracker = LanguageModelMetricTracker()

    logits = torch.randn(2, 8, 260)
    targets = torch.randint(0, 256, (2, 8))

    tracker.update(logits, targets)
    metrics = tracker.compute()

    assert "loss" in metrics
    assert "bpb" in metrics
    assert "ppl" in metrics
    assert "accuracy" in metrics
    assert metrics["total_bytes"] == 16
    assert metrics["loss"] > 0
    assert metrics["bpb"] > 0


def test_patch_statistics():
    # Boundary mask with patches of lengths [4, 4, 2]
    # indices: 0, 4, 8 -> seq_len = 10
    boundaries = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    lens = compute_patch_lengths_from_boundaries(boundaries)
    assert lens == [4, 4, 2]

    stats = compute_patch_statistics(boundaries)
    assert stats["mean"] == (4 + 4 + 2) / 3
    assert stats["min"] == 2
    assert stats["max"] == 4
    assert stats["total_bytes"] == 10
    assert stats["total_patches"] == 3
    assert abs(stats["compression_ratio"] - 10 / 3) < 1e-5


def test_entropy_patch_correlation():
    # High entropy on bytes 0..3 (entropy=5.0), low on 4..7 (entropy=1.0)
    # Boundaries: byte 0 and byte 4 -> patch 1 length 4, patch 2 length 4
    entropy = torch.tensor([5.0, 5.0, 1.0, 1.0, 1.0, 1.0])
    # Boundaries at 0, 1, 2: patch lengths [1, 1, 4]
    boundaries = torch.tensor([1, 1, 1, 0, 0, 0])

    res = compute_entropy_patch_correlation(entropy, boundaries)
    assert "pearson_correlation" in res
    assert -1.0 <= res["pearson_correlation"] <= 1.0


def test_analyze_patch_boundaries():
    text = "Hello world! Test."
    # len is 18. Boundaries at 0 ('H'), 5 (' '), 12 ('!'), 13 (' ')
    bounds = torch.zeros(len(text), dtype=torch.uint8)
    bounds[0] = 1   # 'H' (alphanumeric)
    bounds[5] = 1   # ' ' (whitespace)
    bounds[11] = 1  # '!' (punctuation)

    res = analyze_patch_boundaries(text, bounds)
    assert res["total_boundaries"] == 3
    assert res["whitespace_pct"] == pytest.approx(33.33, abs=0.1)
    assert res["punctuation_pct"] == pytest.approx(33.33, abs=0.1)
    assert res["alphanumeric_pct"] == pytest.approx(33.33, abs=0.1)


def test_robustness_corruptions():
    text = "Hello World"
    antspeak = corrupt_antspeak(text)
    assert antspeak == "H e l l o   W o r l d"

    upper = corrupt_uppercase(text)
    assert upper == "HELLO WORLD"

    data = get_curated_robustness_data()
    assert "Normal English" in data
    assert "Noisy Text" in data
    assert "Code & Structured" in data
    assert "Numbers & Math" in data
    assert "URLs & Paths" in data
    assert "Unicode & Emojis" in data


def test_evaluate_robustness_suite():
    tiny_model = BaselineTransformer(
        vocab_size=260,
        dim=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=64,
    )
    benchmarks = {
        "Test English": ["A cute little puppy.", "The sun rose early."],
        "Test Code": ["x = 10\ny = 20"],
    }
    results = evaluate_robustness_suite(tiny_model, benchmarks=benchmarks, device="cpu")
    assert "Test English" in results
    assert "Test Code" in results
    assert results["Test English"]["bpb"] > 0

    table = format_robustness_table(results)
    assert "Dataset Category" in table
    assert "Test English" in table
