"""CLI Pretraining Script for Byte Latent Transformer (BLT).

Pretrains the ~50M parameter BLT model on TinyStories binary shards with:
- 8-bit AdamW optimizer via bitsandbytes
- bf16/fp16 automatic mixed precision
- Gradient accumulation (physical batch 4, accumulation 16)
- Rotating and best validation BPB checkpointing
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blt.data.dataset import ByteDataset, ByteDataLoader
from blt.model.blt_model import ByteLatentTransformer
from blt.patching.entropy_model import ByteEntropyModel
from blt.train.trainer import BLTTrainer
from blt.train.checkpoint import CheckpointManager
from utils.logger import get_logger
from utils.seeding import seed_everything

logger = get_logger("BLT.Train")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Byte Latent Transformer on TinyStories.")
    parser.add_argument("--config", type=str, default="configs/blt_tinystories_50m.yaml", help="Path to YAML configuration")
    parser.add_argument("--train-data", type=str, default="data/processed/train.bin", help="Path to train.bin")
    parser.add_argument("--val-data", type=str, default="data/processed/val.bin", help="Path to val.bin")
    parser.add_argument("--entropy-checkpoint", type=str, default=None, help="Optional trained entropy model checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Optional BLT checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Execute 5 steps to verify memory and pipeline without full training")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed_everything(args.seed)

    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Datasets & DataLoaders
    t_cfg = config.get("training", {})
    batch_size = t_cfg.get("physical_batch_size", 4)
    init_seq_len = t_cfg.get("context_length_start", 384)

    logger.info(f"Loading datasets (train={args.train_data}, val={args.val_data})...")
    train_dataset = ByteDataset(
        data_path=args.train_data,
        seq_len=init_seq_len + 1,  # +1 for autoregressive target shift
        stride=init_seq_len,
    )
    val_dataset = ByteDataset(
        data_path=args.val_data,
        seq_len=init_seq_len + 1,
        stride=init_seq_len,
    )

    train_loader = ByteDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = ByteDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 2. Entropy Model (optional dynamic patching)
    entropy_model = None
    if args.entropy_checkpoint is not None and os.path.exists(args.entropy_checkpoint):
        logger.info(f"Loading trained entropy model from {args.entropy_checkpoint}...")
        e_cfg = config.get("entropy_model", {})
        entropy_model = ByteEntropyModel(
            vocab_size=config["model"].get("vocab_size", 260),
            dim=e_cfg.get("hidden", 128),
            n_layers=e_cfg.get("layers", 3),
            n_heads=e_cfg.get("heads", 4),
            window_size=e_cfg.get("window", 256),
        )
        ckpt = torch.load(args.entropy_checkpoint, map_location=args.device)
        entropy_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        entropy_model.to(args.device)
        entropy_model.eval()

    # 3. BLT Model
    logger.info("Initializing Byte Latent Transformer (~50M parameter configuration)...")
    model = ByteLatentTransformer.from_config(config, entropy_model=entropy_model)
    model.to(args.device)

    params = model.count_parameters()
    logger.info(f"Model initialized with {params['total']:,} trainable parameters.")

    # 4. Checkpoint Manager & Resumption
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.get("checkpointing", {}).get("dir", "checkpoints"),
        keep_rotating=config.get("checkpointing", {}).get("keep_rotating", 2),
        keep_best_val=config.get("checkpointing", {}).get("keep_best_val", True),
    )

    # 5. Trainer
    trainer = BLTTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=checkpoint_manager,
        device=args.device,
    )

    if args.resume is not None:
        info = CheckpointManager.load(
            args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=args.device,
        )
        trainer.global_step = info["step"]
        logger.info(f"Resumed from step {trainer.global_step}")

    if args.dry_run:
        logger.info("Executing dry run (5 batches on physical batch size 4)...")
        trainer.model.train()
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            tokens = batch.to(args.device)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            with torch.autocast(device_type="cuda" if args.device == "cuda" else "cpu", dtype=trainer.amp_dtype):
                logits, loss = trainer.model(inputs, targets=targets)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            logger.info(f"Dry-run step {i + 1}/5 completed. Loss: {loss.item():.4f}")

        if args.device == "cuda":
            alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logger.info(f"Peak VRAM allocated during dry run: {alloc_mb:.1f} MB (Target < 3500 MB)")
        logger.info("Dry run succeeded without OOM or errors!")
        return

    # 6. Pretraining execution
    trainer.train()


if __name__ == "__main__":
    main()
