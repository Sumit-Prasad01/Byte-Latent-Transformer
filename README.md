# Byte Latent Transformer (BLT)

A research-grade, from-scratch implementation of **"Byte Latent Transformer: Patches Scale Better Than Tokens"** (*Pagnoni et al., Meta FAIR, Dec 2024*).

Engineered to train a **~50M parameter BLT model** from raw bytes on consumer hardware (**NVIDIA GeForce RTX 3050 Laptop GPU with 4GB VRAM**) using the curated **TinyStories** corpus, complete with native C++ acceleration engines (`.dll`, `.a`, `.exe`).

---

## Architecture Overview

```
Raw Bytes (UTF-8, Vocab 260)
  │
  ├──► Byte & Rolling Hash n-gram Embeddings (n=3,4,5 via native RollPolyHash C++ engine)
  │
  ├──► Local Byte Encoder (1 Transformer Block, 256 hidden, local sliding window 256)
  │      │
  │      └──► Encoder Cross-Attention (pools constituent bytes into latent patch queries)
  │
  ├──► Latent Transformer (8 Transformer Blocks, 512 hidden, block-causal attention across patches)
  │      │
  │      └──► Decoder Cross-Attention (byte queries attend causally to latent patch states)
  │
  └──► Local Byte Decoder (4 Transformer Blocks, 256 hidden, local sliding window 256)
         │
         └──► Linear LM Projection Head ──► Next-Byte Logits (260 classes)
```

---

## Quick Reference: Commands Cheat Sheet

All commands below are ready to copy-paste directly into your PowerShell or terminal (using the active virtual environment `.\.venv\Scripts\python.exe`).

### 1. Environment & C++ Native Build

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Rebuild all native C++ binaries (.dll, .a, .exe)
powershell -ExecutionPolicy Bypass -File .\csrc\build_native.ps1 -Clean -BuildAll
```

---

### 2. Testing & Parity Verification

```powershell
# Run the complete test suite (all 57 unit tests)
.\.venv\Scripts\pytest.exe -v

# Run fast / quiet test suite
.\.venv\Scripts\pytest.exe -q

# Run specific module test suites
.\.venv\Scripts\pytest.exe tests/test_c_native_parity.py -v         # C++ DLL vs NumPy parity
.\.venv\Scripts\pytest.exe tests/test_blt_forward_backward.py -v    # 50M BLT forward/backward/overfitting
.\.venv\Scripts\pytest.exe tests/test_eval_metrics.py -v            # BPB, PPL, Accuracy & Robustness
.\.venv\Scripts\pytest.exe tests/test_flops_counter.py -v           # Appendix B analytical FLOPs

# Run native C++ standalone test executables directly (.exe)
.\build\csrc\bin\blt_native.exe --test-all
.\build\csrc\bin\blt_dedup.exe --test
.\build\csrc\bin\blt_rolling_hash_bench.exe --verify
.\build\csrc\bin\blt_boundary_rules_test.exe
.\build\csrc\bin\blt_streaming_patcher_cli.exe --test
```

---

### 3. Data Pipeline & Preprocessing

```powershell
# Download TinyStories raw text
.\.venv\Scripts\python.exe scripts/download_tinystories.py --target-mb 280

# Preprocess raw text with native 64-bit story deduplication into binary shards (.bin)
.\.venv\Scripts\python.exe scripts/preprocess_data.py --raw-file data/raw/tinystories_raw.txt --output-dir data/processed --val-fraction 0.04
```

---

### 4. Training Entrypoints

#### A. Pretrain Byte Entropy Model (~1.5M params)
The lightweight entropy model evaluates local uncertainty $H(x_t)$ to guide dynamic patch boundary creation.

```powershell
# Train Entropy Model on TinyStories
.\.venv\Scripts\python.exe scripts/train_entropy_model.py --config configs/entropy_model_tinystories.yaml
```

#### B. Pretrain 50M Byte Latent Transformer (BLT)
Trained with 8-bit AdamW, mixed precision (`bf16`/`fp16`), and context ramp-up (384 $\rightarrow$ 768 bytes).

```powershell
# 1. Quick dry-run (executes 5 batches to verify GPU memory & pipeline without full training)
.\.venv\Scripts\python.exe scripts/train_blt.py --dry-run

# 2. Launch full BLT pretraining on RTX 3050 GPU
.\.venv\Scripts\python.exe scripts/train_blt.py --config configs/blt_tinystories_50m.yaml

# 3. Pretrain BLT with trained dynamic entropy model
.\.venv\Scripts\python.exe scripts/train_blt.py --config configs/blt_tinystories_50m.yaml --entropy-checkpoint checkpoints/entropy_model.pt

# 4. Resume training from an existing checkpoint
.\.venv\Scripts\python.exe scripts/train_blt.py --resume checkpoints/checkpoint_step_1000.pt
```

#### C. Pretrain Baseline Transformer (Control Model)
Parameter-matched baseline to reproduce the side-by-side comparative experiments from `evaluation.md` §17 & §27.

```powershell
# 1. Quick dry-run
.\.venv\Scripts\python.exe scripts/train_baseline.py --dry-run

# 2. Train Stride-1 Byte Baseline (direct character-level control)
.\.venv\Scripts\python.exe scripts/train_baseline.py --mode byte

# 3. Train BPE Baseline (subword tokenizer control)
.\.venv\Scripts\python.exe scripts/train_baseline.py --mode bpe
```

---

### 5. Autoregressive Text Generation

Generate text from any prompt using nucleus sampling ($p=0.9$) decoded directly into UTF-8 text:

```powershell
# Generate text with default prompt (on CUDA)
.\.venv\Scripts\python.exe scripts/generate.py --max-new-tokens 80

# Custom prompt with temperature & top-p
.\.venv\Scripts\python.exe scripts/generate.py --prompt "Once upon a time, there was a little girl named Lily." --max-new-tokens 120 --temperature 0.8 --top-p 0.9

# Generate using trained checkpoint weights
.\.venv\Scripts\python.exe scripts/generate.py --checkpoint checkpoints/best_val_bpb.pt --prompt "The brave puppy ran into the woods and found"
```

---

### 6. Comprehensive Evaluation Suite (`evaluation.md`)

```powershell
# Run complete evaluation suite across all categories
.\.venv\Scripts\python.exe scripts/evaluate.py --suite all

# Run Language Modeling quality evaluation (Loss, BPB, PPL, Byte Accuracy, NLL) (§3 - §7)
.\.venv\Scripts\python.exe scripts/evaluate.py --suite lm

# Run Patch Compression & Distribution statistics (§8 - §11, §16)
.\.venv\Scripts\python.exe scripts/evaluate.py --suite patching

# Run 6-Category Robustness Suite (Normal, Noisy, Code, Math, URLs, Unicode) (§19)
.\.venv\Scripts\python.exe scripts/evaluate.py --suite robustness

# Run Hardware & Computational Efficiency benchmark (TTFB, Throughput, VRAM, FLOPs) (§12 - §15)
.\.venv\Scripts\python.exe scripts/evaluate.py --suite efficiency

# Print Final BLT vs. Baseline Comparison Table (§27)
.\.venv\Scripts\python.exe scripts/evaluate.py --suite compare

# Evaluate a specific trained checkpoint
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite all
```

---

### 7. Analytical FLOPs Report (Paper Appendix B)

Generate the analytical FLOP efficiency comparison table comparing 50M BLT against a stride-1 byte baseline across context lengths:

```powershell
.\.venv\Scripts\python.exe -c "from blt.flops.report import generate_flop_report; print(generate_flop_report())"
```

Output:
```
| Sequence (Bytes) | Patches (M) | BLT GFLOPs | BLT FLOPs/Byte | Baseline GFLOPs | Baseline FLOPs/Byte | FLOP Efficiency |
|------------------|-------------|------------|----------------|-----------------|---------------------|-----------------|
| 512              | 114         | 11.302     | 22,073,891     | 30.208          | 59,000,144          | 2.67x faster    |
| 1,024            | 228         | 23.089     | 22,548,131     | 69.006          | 67,388,752          | 2.99x faster    |
| 2,048            | 456         | 48.121     | 23,496,611     | 172.372         | 84,165,968          | 3.58x faster    |
| 4,096            | 911         | 103.926    | 25,372,641     | 482.183         | 117,720,400         | 4.64x faster    |
```

---

### 8. Publication Plots Generator (`evaluation.md` §28)

Generate all **15 publication-ready plots** (at 300 DPI) into `plots/`:

```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py --output-dir plots
```

Generated plots:
1. `01_training_loss_vs_steps.png` — Training cross-entropy loss over steps
2. `02_validation_loss_vs_steps.png` — Validation loss progression
3. `03_validation_bpb_vs_steps.png` — Validation Bits-Per-Byte convergence
4. `04_train_vs_val_loss.png` — Overfitting/underfitting diagnostic overlay
5. `05_patch_length_distribution.png` — Histogram with mean & median markers
6. `06_local_entropy_vs_patch_length.png` — Scatter plot with Pearson $r$ trendline
7. `07_compression_ratio_distribution.png` — Window-wise compression histogram
8. `08_bpb_vs_training_compute.png` — Pareto efficiency (BPB vs GFLOPs)
9. `09_bpb_vs_training_time.png` — BPB vs training wall-clock time
10. `10_bpb_vs_peak_vram.png` — Memory footprint vs compression quality
11. `11_blt_vs_baseline_bpb.png` — Comparative bar chart: BLT vs Baseline BPB
12. `12_blt_vs_baseline_throughput.png` — Comparative inference throughput (bytes/sec)
13. `13_blt_vs_baseline_memory.png` — Comparative peak VRAM allocation
14. `14_context_length_vs_bpb.png` — Context length scaling curve
15. `15_context_length_vs_vram.png` — Memory scaling curve vs 4GB safety ceiling

---

## Memory Optimization Stack (RTX 3050 4GB GPU)

| Technique | Implementation | Memory Benefit |
|---|---|---|
| **8-bit AdamW** | `bitsandbytes.optim.AdamW8bit` via `blt/train/optim.py` | Reduces optimizer memory from ~370 MB to **~92 MB** |
| **Gradient Checkpointing** | PyTorch activation checkpointing on Latent Transformer layers | Reduces latent activation cache by **~65%** |
| **Mixed Precision (AMP)** | `torch.autocast(dtype=torch.bfloat16)` | Cuts weight and activation footprint in half |
| **SDPA FlashAttention** | `torch.nn.functional.scaled_dot_product_attention` | $O(N)$ linear memory attention without full $N \times N$ matrix |
| **Gradient Accumulation** | Physical batch 64 $\times$ 1 accumulation step = 64 effective batch | Peak training memory stays around **2.7 to 2.9 GB** |
| **Context Length Ramping** | 384 bytes $\rightarrow$ 768 bytes over 2,000 warmup steps | Avoids early memory spikes while stabilizing convergence |

---

## Directory Layout

```
Byte-Latent-Transformer/
├── configs/
│   ├── blt_tinystories_50m.yaml         # Primary 50M BLT config for RTX 3050 4GB
│   ├── entropy_model_tinystories.yaml   # Small byte entropy model config
│   └── baseline_bpe_llama.yaml          # Parameter-matched control baseline
├── csrc/                                # C++ Native Triad engine sources & CLI benchmarks
├── blt/
│   ├── csrc/bridge.py                   # Python ctypes zero-copy bridge + fallbacks
│   ├── data/                            # ByteDataset, ByteDataLoader, packing, tokenizer
│   ├── patching/                        # EntropyModel, monotonic rules, streaming patcher
│   ├── modules/                         # Hash n-grams, cross-attentions, encoder, latent, decoder
│   ├── model/                           # ByteLatentTransformer & BaselineTransformer assemblies
│   ├── flops/                           # Analytical Appendix B FLOP counter & reporting
│   ├── train/                           # 8-bit AdamW, cosine schedule, checkpointing, trainer
│   └── eval/                            # BPB, accuracy, patch statistics, robustness, efficiency
├── scripts/
│   ├── download_tinystories.py          # TinyStories dataset downloader
│   ├── preprocess_data.py               # Shard creation & 64-bit story deduplication
│   ├── train_entropy_model.py           # Entropy model pretraining
│   ├── train_blt.py                     # 50M BLT pretraining CLI
│   ├── train_baseline.py                # Baseline control pretraining CLI
│   ├── generate.py                      # Autoregressive text generation with nucleus sampling
│   ├── evaluate.py                      # Master evaluation CLI (evaluation.md)
│   └── plot_results.py                  # 15 publication plots generator (evaluation.md §28)
├── tests/                               # 57 automated pytest unit & integration tests
├── evaluation.md                        # Formal evaluation blueprint & benchmark criteria
└── README.md                            # Documentation and cheat sheet
```
