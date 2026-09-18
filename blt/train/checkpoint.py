"""Resumable and rotating checkpoint manager for BLT training."""

import os
import glob
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("BLT.Checkpoint")


class CheckpointManager:
    """Manages rotating periodic checkpoints and best-model preservation."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_rotating: int = 2,
        keep_best_val: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_rotating = keep_rotating
        self.keep_best_val = keep_best_val
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        step: int,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        val_bpb: Optional[float] = None,
        is_best: bool = False,
    ) -> str:
        """Save a complete training checkpoint."""
        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "val_bpb": val_bpb,
            "rng_torch": torch.get_rng_state(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if torch.cuda.is_available():
            checkpoint["rng_cuda"] = torch.cuda.get_rng_state()

        step_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, step_path)
        logger.info(f"Saved step checkpoint: {step_path}")

        if is_best and self.keep_best_val:
            best_path = os.path.join(self.checkpoint_dir, "best_val_bpb.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Updated best validation checkpoint (BPB={val_bpb:.4f}): {best_path}")

        # Rotate older step checkpoints
        self._rotate_checkpoints()
        return step_path

    def _rotate_checkpoints(self):
        """Keep only the latest `keep_rotating` step checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        files = glob.glob(pattern)
        if len(files) <= self.keep_rotating:
            return

        # Sort by step number
        def _get_step(filename):
            try:
                base = os.path.basename(filename)
                step_str = base.replace("checkpoint_step_", "").replace(".pt", "")
                return int(step_str)
            except ValueError:
                return -1

        sorted_files = sorted(files, key=_get_step)
        files_to_remove = sorted_files[: -self.keep_rotating]
        for f in files_to_remove:
            try:
                os.remove(f)
                logger.debug(f"Removed older rotating checkpoint: {f}")
            except OSError as e:
                logger.warning(f"Could not remove checkpoint {f}: {e}")

    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Dict[str, Any]:
        """Load checkpoint state and restore weights and training progress."""
        logger.info(f"Loading checkpoint from {checkpoint_path} to {device}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_to_load = model.module if hasattr(model, "module") else model
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_to_load.load_state_dict(state_dict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state dict: {e}")

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scheduler state dict: {e}")

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "val_bpb": checkpoint.get("val_bpb", None),
        }
