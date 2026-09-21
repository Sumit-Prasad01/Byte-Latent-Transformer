"""Core Training Engine for Byte Latent Transformer (BLT).

Engineered strictly for consumer GPUs (RTX 3050 Laptop GPU 4GB VRAM):
- Physical batch size 4 with 16 gradient accumulation steps (effective batch 64).
- Automatic mixed precision (bf16/fp16 autocast) with PyTorch SDPA.
- Dynamic context length ramp-up (384 -> 768 bytes).
- 8-bit AdamW optimizer with cosine learning rate schedule and warmup.
- Live VRAM telemetry, rotating checkpoints, and periodic BPB evaluation.
"""

import os
import time
from typing import Dict, Any, Optional, List
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from blt.model.blt_model import ByteLatentTransformer
from blt.train.optim import configure_optimizers, get_cosine_schedule_with_warmup
from blt.train.checkpoint import CheckpointManager
from blt.eval.bpb import evaluate_byte_model_bpb, loss_to_bpb
from utils.logger import get_logger

logger = get_logger("BLT.Trainer")


class BLTTrainer:
    """Trainer coordinating optimization, checkpointing, and evaluation of BLT."""

    def __init__(
        self,
        model: ByteLatentTransformer,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)

        t_cfg = config.get("training", {})
        self.physical_batch_size = t_cfg.get("physical_batch_size", 4)
        self.grad_accum_steps = t_cfg.get("grad_accum_steps", 16)
        self.precision = t_cfg.get("precision", "bf16")
        self.grad_clip_norm = t_cfg.get("grad_clip_norm", 1.0)
        self.peak_lr = float(t_cfg.get("peak_lr", 3.0e-4))
        self.weight_decay = float(t_cfg.get("weight_decay", 0.15))
        self.warmup_steps = int(t_cfg.get("warmup_steps", 2000))
        self.lr_floor_fraction = float(t_cfg.get("lr_floor_fraction", 0.1))
        self.planned_epochs = int(t_cfg.get("planned_epochs", 20))
        self.context_start = int(t_cfg.get("context_length_start", 384))
        self.context_final = int(t_cfg.get("context_length_final", 768))

        # Precision autocast setup
        self.use_amp = self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # Optimizer
        self.optimizer = configure_optimizers(
            model=self.model,
            weight_decay=self.weight_decay,
            lr=self.peak_lr,
            betas=tuple(t_cfg.get("betas", [0.9, 0.95])),
            eps=float(t_cfg.get("eps", 1e-8)),
            optimizer_type=t_cfg.get("optimizer", "adamw_8bit"),
        )

        # Learning rate schedule
        self.total_steps = len(self.train_loader) * self.planned_epochs // self.grad_accum_steps
        self.total_steps = max(self.total_steps, self.warmup_steps + 1000)
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.lr_floor_fraction,
        )

        # Checkpointing
        ckpt_cfg = config.get("checkpointing", {})
        self.save_every_steps = ckpt_cfg.get("save_every_steps", 1000)
        if checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=ckpt_cfg.get("dir", "checkpoints"),
                keep_rotating=ckpt_cfg.get("keep_rotating", 2),
                keep_best_val=ckpt_cfg.get("keep_best_val", True),
            )
        else:
            self.checkpoint_manager = checkpoint_manager

        # Logging / Eval schedule
        log_cfg = config.get("logging", {})
        self.eval_every_steps = log_cfg.get("eval_every_steps", 1000)
        self.qualitative_sample_every = log_cfg.get("qualitative_sample_every_steps", 2000)

        self.global_step = 0
        self.best_val_bpb = float("inf")

    def _get_current_context_length(self) -> int:
        """Calculate ramped context length based on current training step."""
        if self.global_step >= self.warmup_steps:
            return self.context_final
        progress = self.global_step / max(1, self.warmup_steps)
        return int(self.context_start + (self.context_final - self.context_start) * progress)

    def train_epoch(self, epoch: int) -> float:
        """Execute one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        accum_loss = 0.0
        start_time = time.perf_counter()

        logger.info(f"Starting epoch {epoch + 1}/{self.planned_epochs} ({num_batches} batches)...")
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for b_idx, batch in enumerate(pbar):
            # Resolve batch tokens
            if isinstance(batch, (tuple, list)):
                tokens = batch[0]
            else:
                tokens = batch

            # Context length slicing (ramp-up schedule)
            cur_ctx_len = self._get_current_context_length()
            if tokens.shape[1] > cur_ctx_len + 1:
                tokens = tokens[:, : cur_ctx_len + 1]

            tokens = tokens.to(self.device, non_blocking=True)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            # Forward pass with AMP autocast
            with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=self.amp_dtype, enabled=self.use_amp):
                logits, loss = self.model(inputs, targets=targets)
                scaled_loss = loss / self.grad_accum_steps

            # Backward pass
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += loss.item()

            # Optimizer step on accumulation boundary
            if (b_idx + 1) % self.grad_accum_steps == 0 or (b_idx + 1) == num_batches:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                step_loss = accum_loss / self.grad_accum_steps
                step_bpb = loss_to_bpb(step_loss)
                accum_loss = 0.0

                # Telemetry
                lr_curr = self.optimizer.param_groups[0]["lr"]
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if self.device == "cuda" else 0.0
                pbar.set_postfix({
                    "loss": f"{step_loss:.4f}",
                    "bpb": f"{step_bpb:.3f}",
                    "lr": f"{lr_curr:.2e}",
                    "ctx": cur_ctx_len,
                    "vram": f"{vram_mb:.0f}MB",
                })

                # Evaluation check
                if self.val_loader is not None and self.global_step % self.eval_every_steps == 0:
                    val_bpb = self.evaluate()
                    is_best = val_bpb < self.best_val_bpb
                    if is_best:
                        self.best_val_bpb = val_bpb
                    self.checkpoint_manager.save(
                        step=self.global_step,
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        val_bpb=val_bpb,
                        is_best=is_best,
                    )
                    self.model.train()

                # Periodic step checkpoint
                elif self.global_step % self.save_every_steps == 0:
                    self.checkpoint_manager.save(
                        step=self.global_step,
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )

                # Qualitative text generation check
                if self.global_step % self.qualitative_sample_every == 0:
                    self.generate_sample("Once upon a time")

        elapsed = time.perf_counter() - start_time
        logger.info(f"Epoch {epoch + 1} completed in {elapsed:.1f}s. Global step: {self.global_step}")
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> float:
        """Run validation evaluation to compute exact Bits-Per-Byte."""
        self.model.eval()
        logger.info(f"Running validation evaluation (step {self.global_step})...")
        val_bpb = evaluate_byte_model_bpb(
            model=self.model,
            dataloader=self.val_loader,
            device=self.device,
            max_batches=max_batches,
        )
        logger.info(f"Validation BPB @ step {self.global_step}: {val_bpb:.4f} (best: {self.best_val_bpb:.4f})")
        return val_bpb

    @torch.no_grad()
    def generate_sample(self, prompt: str = "Once upon a time", max_new_tokens: int = 60) -> str:
        """Sample generation during training to monitor qualitative text coherence."""
        self.model.eval()
        prompt_bytes = list(prompt.encode("utf-8"))
        prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device)

        out_tokens = self.model.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
        )
        raw_bytes = bytes([b for b in out_tokens[0].cpu().tolist() if b < 256])
        text = raw_bytes.decode("utf-8", errors="replace")
        logger.info(f"[Sample @ Step {self.global_step}]: {text}")
        return text

    def train(self):
        """Execute full training across planned epochs."""
        logger.info(f"Starting BLT pretraining for {self.planned_epochs} epochs (~{self.total_steps} optimizer steps)...")
        for epoch in range(self.planned_epochs):
            self.train_epoch(epoch)
        logger.info("Training complete!")
