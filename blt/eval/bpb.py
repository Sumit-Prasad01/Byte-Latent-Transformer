"""Bits-Per-Byte (BPB) evaluation metric (Equation 18).

BPB(x) = L_CE(x) / (ln(2) * num_bytes)

Provides mathematically rigorous, apples-to-apples compression evaluation
between byte-level models (vocab=260) and subword tokenized models (BPE/LLaMA).
"""

import math
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def loss_to_bpb(loss_nats: float) -> float:
    """Convert average cross-entropy loss (in nats per byte) to Bits-Per-Byte.

    When loss_nats is already per-byte:
        BPB = loss_nats / ln(2)
    """
    return loss_nats / math.log(2)


def compute_bpb(
    total_loss_nats: float,
    num_bytes: int,
) -> float:
    """Compute Bits-Per-Byte given total cross-entropy loss (in nats) and exact UTF-8 byte count.

    Args:
        total_loss_nats: Sum of cross-entropy losses across all targets (sum reduction).
        num_bytes: Total number of raw UTF-8 bytes represented by the target sequence.

    Returns:
        Bits-Per-Byte (float).
    """
    if num_bytes <= 0:
        return 0.0
    return total_loss_nats / (math.log(2) * num_bytes)


@torch.no_grad()
def evaluate_byte_model_bpb(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[Union[str, torch.device]] = None,
    max_batches: Optional[int] = None,
) -> float:
    """Evaluate Bits-Per-Byte for a byte-level model over a DataLoader.

    For byte models where vocab=260 and each token represents 1 byte (excluding special tokens):
    Target sequence byte count equals the number of evaluated token positions.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss_nats = 0.0
    total_bytes = 0

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        # Autoregressive target: predict x_{t+1} from x_t
        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits, _ = model(inputs)
        # Sum reduction to get exact nats
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )

        # Valid byte count: count targets that are actual byte values (< 256)
        valid_byte_mask = (targets < 256) & (targets >= 0)
        num_valid_bytes = int(valid_byte_mask.sum().item())

        total_loss_nats += loss.item()
        total_bytes += num_valid_bytes

    return compute_bpb(total_loss_nats, total_bytes)


@torch.no_grad()
def evaluate_bpe_model_bpb(
    model: nn.Module,
    encoded_token_batches: List[torch.Tensor],
    byte_lengths: List[int],
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """Evaluate BPB for a subword tokenized BPE model normalized by exact byte lengths.

    Crucial: dividing by true UTF-8 byte count, NOT token count, ensuring fair comparison.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss_nats = 0.0
    total_bytes = sum(byte_lengths)

    for x in encoded_token_batches:
        x = x.to(device)
        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits, _ = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        total_loss_nats += loss.item()

    return compute_bpb(total_loss_nats, total_bytes)
