# Byte Latent Transformer (BLT)

Implementation of **"Byte Latent Transformer: Patches Scale Better Than Tokens"** (Pagnoni et al., FAIR at Meta, Dec 2024).

## Overview

Byte Latent Transformer (BLT) is a byte-level language model architecture that eliminates tokenization artifacts by dynamically grouping bytes into variable-length patches based on entropy and uncertainty.

Key architectural pillars:
1. **Dynamic Patching:** A lightweight entropy model dynamically identifies patch boundaries based on entropy spikes (global threshold or monotonicity rule).
2. **Local Byte Encoder:** Encodes raw byte sequences + roll-poly hash n-gram representations with local block-causal attention and cross-attention into patch representations.
3. **Latent (Global) Transformer:** Autoregressively operates over patch representations with block-causal attention.
4. **Local Byte Decoder:** Decodes patch representations back to next-byte predictions via cross-attention and local causal transformer layers.

---

## Repository Structure

```
blt/
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   ├── blt_tinystories_40m.yaml  # Config for RTX 3050 4GB (~40M params)
│   ├── blt_400m.yaml             # Config for 400M scale
│   ├── blt_1b.yaml               # Config for 1B scale
│   ├── entropy_model_100m.yaml   # Small entropy model
│   └── baseline_bpe_llama.yaml   # Standard BPE decoder-only control
├── blt/
│   ├── data/                     # Byte-level dataset, packing, preprocessing & BPE baseline
│   ├── patching/                 # Entropy model, strided/space/entropy patchers & boundary rules
│   ├── modules/                  # Transformer primitives, hash n-grams, cross-attention, local encoder/decoder, latent transformer
│   ├── model/                    # Full BLT model and baseline BPE model
│   ├── flops/                    # Per-module FLOP accounting (Appendix B)
│   ├── train/                    # Trainer, AdamW / 8-bit AdamW optimizer, BPB loss
│   ├── eval/                     # Bits-per-byte (BPB), downstream harness, CUTE, noise robustness, FLORES
│   └── utils/                    # Checkpointing, logging, reproducible seeding
├── scripts/                      # Training, evaluation, ablations, and scaling study entrypoints
└── tests/                        # Comprehensive unit and integration tests
```

---

## Quickstart

### Installation

```bash
# Clone and install in editable mode
pip install -e .
```

### Running Tests

```bash
pytest tests/
```

### Training

1. **Train Entropy Model:**
   ```bash
   python scripts/train_entropy_model.py --config configs/blt_tinystories_40m.yaml
   ```

2. **Train Full BLT:**
   ```bash
   python scripts/train_blt.py --config configs/blt_tinystories_40m.yaml
   ```

3. **Evaluation:**
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/blt_best.pt --suite bpb
   ```
# Byte-Latent-Transformer
