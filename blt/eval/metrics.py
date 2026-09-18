"""Language Modeling Evaluation Metrics (evaluation.md §3 - §7).

Implements:
- Cross-Entropy Loss (nats) (§3)
- Bits-Per-Byte (BPB) (§4)
- Perplexity (PPL) (§5)
- Byte-Level Accuracy (§6)
- Negative Log-Likelihood (NLL) (§7)
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_to_bpb(loss_nats: float) -> float:
    """Convert cross-entropy in nats to Bits-Per-Byte (BPB = L_CE / ln(2))."""
    return loss_nats / math.log(2)


def loss_to_ppl(loss_nats: float) -> float:
    """Convert cross-entropy in nats to Perplexity (PPL = exp(L_CE))."""
    try:
        return math.exp(min(100.0, loss_nats))
    except OverflowError:
        return float("inf")


def compute_byte_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    max_byte_val: int = 255,
) -> Tuple[int, int]:
    """Compute number of correctly predicted byte tokens and total valid byte tokens.

    Args:
        logits: (batch_size, seq_len, vocab_size) or flattened (N, vocab_size).
        targets: (batch_size, seq_len) or flattened (N,).
        ignore_index: Target index to ignore.
        max_byte_val: Upper bound on valid raw byte token value (255 for raw bytes).

    Returns:
        (correct_count, total_count)
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)

    preds = logits.argmax(dim=-1)
    mask = (targets != ignore_index) & (targets >= 0) & (targets <= max_byte_val)

    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct, total


class LanguageModelMetricTracker:
    """Accumulates cross-entropy loss, byte accuracy, and token counts across evaluation batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss_nats = 0.0
        self.total_correct_bytes = 0
        self.total_valid_bytes = 0
        self.num_batches = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ):
        """Update running metrics with a batch of predictions and targets."""
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)

        # Sum cross-entropy reduction for exact information accumulation
        loss_sum = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="sum",
            ignore_index=ignore_index,
        )

        correct, count = compute_byte_accuracy(
            logits_flat, targets_flat, ignore_index=ignore_index
        )

        self.total_loss_nats += loss_sum.item()
        self.total_correct_bytes += correct
        self.total_valid_bytes += count
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """Compute final aggregated metrics."""
        if self.total_valid_bytes == 0:
            return {
                "loss": 0.0,
                "bpb": 0.0,
                "ppl": 1.0,
                "accuracy": 0.0,
                "nll": 0.0,
                "total_bytes": 0,
            }

        avg_loss = self.total_loss_nats / self.total_valid_bytes
        bpb = loss_to_bpb(avg_loss)
        ppl = loss_to_ppl(avg_loss)
        accuracy = self.total_correct_bytes / self.total_valid_bytes

        return {
            "loss": avg_loss,
            "bpb": bpb,
            "ppl": ppl,
            "accuracy": accuracy,
            "nll": self.total_loss_nats,
            "total_bytes": self.total_valid_bytes,
        }
