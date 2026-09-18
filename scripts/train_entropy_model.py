"""Training script for the byte-level entropy model on TinyStories.

Lightweight next-byte LM (~1.5M parameters) trained to convergence to estimate
Shannon entropy H(x_i) for BLT patch boundary creation.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from blt.patching.entropy_model import ByteEntropyModel
from blt.data.dataset import ByteDataset, ByteDataLoader
from blt.eval.bpb import compute_bpb
from utils.logger import get_logger
from utils.seeding import seed_everything

logger = get_logger("BLT.TrainEntropy")


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_entropy_model(
    config_path: str = "configs/entropy_model_tinystories.yaml",
    train_data_path: str = "data/processed/train.bin",
    val_data_path: str = "data/processed/val.bin",
    output_checkpoint: str = "checkpoints/entropy_model.pt",
    max_steps_override: Optional[int] = None,
    device_str: Optional[str] = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(42)

    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Training ByteEntropyModel on device: {device}")

    # Dataset & Dataloaders
    seq_len = cfg.get("window_size", 256)
    batch_size = cfg.get("training", {}).get("batch_size", 16)

    train_ds = ByteDataset(train_data_path, sequence_length=seq_len + 1, stride=seq_len)
    val_ds = ByteDataset(val_data_path, sequence_length=seq_len + 1, stride=seq_len)

    train_loader = ByteDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = ByteDataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Train dataset samples: {len(train_ds)}, Val dataset samples: {len(val_ds)}")

    # Model
    model = ByteEntropyModel(
        vocab_size=cfg.get("vocab_size", 260),
        dim=cfg.get("hidden", 128),
        n_layers=cfg.get("layers", 3),
        n_heads=cfg.get("heads", 4),
        sliding_window=cfg.get("window_size", 256),
        max_seq_len=cfg.get("max_seq_len", 1024),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ByteEntropyModel total parameters: {total_params:,}")

    # Training hyperparameters
    t_cfg = cfg.get("training", {})
    max_steps = max_steps_override if max_steps_override is not None else t_cfg.get("max_steps", 2000)
    max_lr = t_cfg.get("learning_rate", 5.0e-4)
    min_lr = t_cfg.get("min_lr", 5.0e-5)
    warmup_steps = t_cfg.get("warmup_steps", 200)
    weight_decay = t_cfg.get("weight_decay", 0.1)
    eval_every = t_cfg.get("eval_every", 250)

    # Use fused AdamW if available on CUDA
    use_fused = (device.type == "cuda")
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, fused=use_fused)

    # Precision
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    use_autocast = (device.type == "cuda")

    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)

    model.train()
    step = 0
    running_loss = 0.0
    best_val_bpb = float("inf")

    data_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, desc="Training Entropy Model")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Schedule LR
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, loss = model(inputs, targets=targets)
        else:
            logits, loss = model(inputs, targets=targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        step += 1
        pbar.update(1)

        # Periodic evaluation
        if step % eval_every == 0 or step == max_steps:
            model.eval()
            val_loss_nats = 0.0
            val_bytes = 0

            with torch.no_grad():
                for v_idx, v_batch in enumerate(val_loader):
                    if v_idx >= 50:  # evaluate on up to 50 batches for speed
                        break
                    v_batch = v_batch.to(device)
                    v_in = v_batch[:, :-1]
                    v_tgt = v_batch[:, 1:]

                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            v_logits, _ = model(v_in)
                    else:
                        v_logits, _ = model(v_in)

                    v_loss = F.cross_entropy(
                        v_logits.view(-1, model.vocab_size),
                        v_tgt.reshape(-1),
                        reduction="sum",
                    )
                    val_loss_nats += v_loss.item()
                    val_bytes += (v_tgt < 256).sum().item()

            val_bpb = compute_bpb(val_loss_nats, val_bytes)
            train_avg_loss = running_loss / eval_every
            running_loss = 0.0

            pbar.set_postfix({"train_loss": f"{train_avg_loss:.3f}", "val_bpb": f"{val_bpb:.3f}", "lr": f"{lr:.2e}"})
            logger.info(f"Step {step}/{max_steps} - Train Loss: {train_avg_loss:.4f}, Val BPB: {val_bpb:.4f}, LR: {lr:.2e}")

            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "best_val_bpb": best_val_bpb,
                    },
                    output_checkpoint,
                )
                logger.info(f"Saved new best checkpoint to {output_checkpoint} (Val BPB: {val_bpb:.4f})")

            model.train()

    pbar.close()
    logger.info(f"Entropy model training complete. Best Val BPB: {best_val_bpb:.4f}")
    return best_val_bpb


def main():
    parser = argparse.ArgumentParser(description="Train ByteEntropyModel on TinyStories")
    parser.add_argument("--config", type=str, default="configs/entropy_model_tinystories.yaml")
    parser.add_argument("--train-data", type=str, default="data/processed/train.bin")
    parser.add_argument("--val-data", type=str, default="data/processed/val.bin")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/entropy_model.pt")
    parser.add_argument("--steps", type=int, default=None, help="Override training max steps")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_entropy_model(
        config_path=args.config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_checkpoint=args.checkpoint,
        max_steps_override=args.steps,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
