"""Pretraining script for Baseline Transformer models (MegaByte Stride-1 and BPE control).

Matches the parameter count and training conditions of the 50M BLT model
to enable fair, rigorous comparative evaluation per evaluation.md §17 & §27.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from blt.data.dataset import ByteDataset, ByteDataLoader
from blt.model.baseline_bpe_model import BaselineTransformer
from blt.train.optim import configure_optimizers, get_cosine_schedule_with_warmup
from blt.train.checkpoint import CheckpointManager
from blt.eval.bpb import evaluate_byte_model_bpb, loss_to_bpb
from utils.logger import get_logger
from utils.seeding import seed_everything

logger = get_logger("Baseline.Train")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Baseline Transformer control model.")
    parser.add_argument("--config", type=str, default="configs/baseline_bpe_llama.yaml", help="Config YAML path")
    parser.add_argument("--train-data", type=str, default="data/processed/train.bin", help="Path to train.bin")
    parser.add_argument("--val-data", type=str, default="data/processed/val.bin", help="Path to val.bin")
    parser.add_argument("--mode", type=str, default="byte", choices=["byte", "bpe"], help="Baseline mode: byte or bpe")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--steps", type=int, default=3500, help="Total training steps (default: 3500 for ~1hr run)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Execute 5 steps to verify memory without full training")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)

    logger.info(f"Loading configuration from {args.config} (Mode: {args.mode.upper()})...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # In byte mode, vocab_size is 260
    vocab_size = 260 if args.mode == "byte" else config.get("vocab_size", 32000)
    dim = config.get("dim", 384)
    layers = config.get("layers", 8)
    heads = config.get("heads", 6)
    max_seq_len = config.get("max_seq_len", 512)
    grad_checkpointing = config.get("grad_checkpointing", True)

    t_cfg = config.get("training", {})
    batch_size = t_cfg.get("batch_size", 4)
    grad_accum_steps = t_cfg.get("grad_accum_steps", 16)
    learning_rate = float(t_cfg.get("learning_rate", 3.0e-4))
    weight_decay = float(t_cfg.get("weight_decay", 0.1))
    precision = t_cfg.get("precision", "bf16")

    # 1. Dataset & DataLoader
    logger.info(f"Loading datasets (train={args.train_data}, val={args.val_data})...")
    train_dataset = ByteDataset(
        data=args.train_data,
        sequence_length=max_seq_len + 1,
        stride=max_seq_len,
    )
    val_dataset = ByteDataset(
        data=args.val_data,
        sequence_length=max_seq_len + 1,
        stride=max_seq_len,
    )

    train_loader = ByteDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = ByteDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model
    logger.info("Initializing BaselineTransformer...")
    model = BaselineTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=layers,
        n_heads=heads,
        max_seq_len=max_seq_len,
        grad_checkpointing=grad_checkpointing,
    )
    model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Baseline model initialized with {n_params:,} trainable parameters.")

    # 3. Precision & Optimizer
    use_amp = args.device == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    optimizer = configure_optimizers(
        model=model,
        weight_decay=weight_decay,
        lr=learning_rate,
        optimizer_type="adamw_8bit",
    )

    planned_epochs = 5
    total_steps = args.steps if args.steps is not None else (len(train_loader) * planned_epochs // grad_accum_steps)
    warmup_steps = min(300, total_steps // 10)
    steps_per_epoch = max(1, total_steps // planned_epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    ckpt_dir = os.path.join("checkpoints", f"baseline_{args.mode}")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        keep_rotating=2,
        keep_best_val=True,
    )

    global_step = 0
    best_val_bpb = float("inf")

    if args.resume is not None:
        info = CheckpointManager.load(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, device=args.device)
        global_step = info["step"]
        logger.info(f"Resumed from step {global_step}")

    if args.dry_run:
        logger.info("Executing dry run (5 batches on physical batch size 4)...")
        model.train()
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            tokens = batch.to(args.device)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            with torch.autocast(device_type="cuda" if args.device == "cuda" else "cpu", dtype=amp_dtype, enabled=use_amp):
                logits, loss = model(inputs, targets=targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"Dry-run step {i + 1}/5 completed. Loss: {loss.item():.4f}")

        if args.device == "cuda":
            alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logger.info(f"Peak VRAM allocated: {alloc_mb:.1f} MB")
        logger.info("Dry run succeeded without errors!")
        return

    # 4. Training Loop
    logger.info(f"Starting Baseline pretraining for {planned_epochs} epochs (~{total_steps} steps)...")
    train_iter = None

    for epoch in range(planned_epochs):
        if global_step >= total_steps:
            break
        model.train()
        accum_loss = 0.0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{planned_epochs}", leave=True)
        step_in_epoch = 0
        micro_step = 0

        while step_in_epoch < steps_per_epoch and global_step < total_steps:
            if train_iter is None:
                train_iter = iter(train_loader)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            tokens = batch.to(args.device)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]

            with torch.autocast(device_type="cuda" if args.device == "cuda" else "cpu", dtype=amp_dtype, enabled=use_amp):
                logits, loss = model(inputs, targets=targets)
                scaled_loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_in_epoch += 1
                pbar.update(1)

                step_loss = accum_loss / grad_accum_steps
                step_bpb = loss_to_bpb(step_loss)
                accum_loss = 0.0

                pbar.set_postfix({
                    "step": f"{global_step}/{total_steps}",
                    "loss": f"{step_loss:.4f}",
                    "bpb": f"{step_bpb:.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                if global_step % 500 == 0:
                    model.eval()
                    val_bpb = evaluate_byte_model_bpb(model, val_loader, device=args.device, max_batches=30)
                    is_best = val_bpb < best_val_bpb
                    if is_best:
                        best_val_bpb = val_bpb
                    checkpoint_manager.save(
                        step=global_step,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        val_bpb=val_bpb,
                        is_best=is_best,
                    )
                    model.train()

        pbar.close()

    logger.info("Baseline training complete!")


if __name__ == "__main__":
    main()
