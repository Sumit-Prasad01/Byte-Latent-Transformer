"""Optimizer and learning rate scheduler for BLT pretraining.

Optimized for consumer GPUs (RTX 3050 4GB VRAM):
- 8-bit AdamW via bitsandbytes (saves ~300MB VRAM) with fallback to PyTorch AdamW.
- Parameter grouping separating 2D weights (decay) from 1D biases/norms (no-decay).
- Linear warmup with cosine decay schedule down to lr_floor_fraction.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from utils.logger import get_logger

logger = get_logger("BLT.Optim")


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.15,
    lr: float = 3.0e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    optimizer_type: str = "adamw_8bit",
) -> torch.optim.Optimizer:
    """Create optimizer with weight decay parameter grouping.

    2D parameters (weights in Linear and Embedding layers) receive weight decay.
    1D parameters (biases, RMSNorm scale parameters) have weight decay set to 0.0.
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    logger.info(f"Optimizer parameter groups: {len(decay_params)} decayed ({num_decay:,} params), {len(nodecay_params)} non-decayed ({num_nodecay:,} params)")

    if optimizer_type == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            logger.info("Using bitsandbytes AdamW8bit optimizer for memory efficiency.")
            return bnb.optim.AdamW8bit(optim_groups, lr=lr, betas=betas, eps=eps)
        except Exception as e:
            logger.warning(f"Failed to initialize bitsandbytes AdamW8bit ({e}). Falling back to torch.optim.AdamW.")

    # Fallback to standard PyTorch AdamW (with fused=True if on CUDA)
    use_fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
    except Exception:
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a schedule with linear warmup and cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
