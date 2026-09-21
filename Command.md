# Byte Latent Transformer (BLT) — Command Reference Guide

This document contains all operational commands required to build, test, preprocess, train, evaluate, and visualize the **Byte Latent Transformer (BLT)** model on an **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)** using PowerShell.

---

## Table of Contents
1. [Environment & Test Verification](#1-environment--test-verification)
2. [Data Acquisition & Preprocessing](#2-data-acquisition--preprocessing)
3. [Model Pretraining](#3-model-pretraining)
   - [3.1 Pretrain Byte Entropy Model (~1.5M params)](#31-pretrain-byte-entropy-model-15m-params)
   - [3.2 Pretrain 50M Byte Latent Transformer (BLT)](#32-pretrain-50m-byte-latent-transformer-blt)
   - [3.3 Pretrain Baseline Control Model (Optional)](#33-pretrain-baseline-control-model-optional)
4. [Autoregressive Text Generation](#4-autoregressive-text-generation)
5. [Evaluation & Benchmarking Suite](#5-evaluation--benchmarking-suite)
6. [Analytical FLOPs Reporting](#6-analytical-flops-reporting)
7. [Publication Plots Generation](#7-publication-plots-generation)
8. [Quick Reference Summary Table](#8-quick-reference-summary-table)

---

## 1. Environment & Test Verification

### 1.1 Activate Python Virtual Environment
Activates the project virtual environment containing PyTorch, bitsandbytes, and dependencies.
```powershell
.\.venv\Scripts\Activate.ps1
```
- **What it does:** Sets up the Python environment paths in the current terminal session.
- **Runtime:** < 1 second.
- **Output:** PowerShell prompt prefixes with `(.venv)`.

---

### 1.2 Run Complete Automated Test Suite (57 Tests)
Executes all unit and integration tests covering native C++ bridge parity, embedding math, encoder/decoder cross-attention, SDPA causality, and evaluation metrics.
```powershell
.\.venv\Scripts\pytest.exe -q
```
- **What it does:** Verifies that all 57 test cases pass with a 100% success rate.
- **Runtime:** ~6 to 7 seconds.
- **VRAM / Memory:** CPU & GPU (< 400 MB).
- **Output:** `57 passed in 6.xx s`.

---

### 1.3 (Optional) Rebuild 64-bit Native C++ DLL (MSVC)
Recompiles the native C++ acceleration engines into a true 64-bit Windows DLL matching 64-bit Python.
```powershell
powershell -ExecutionPolicy Bypass -File .\csrc\build_native_msvc.ps1
```
- **What it does:** Uses Microsoft Visual Studio C++ Compiler (`cl.exe` for `x64`) to build `build/csrc/bin/blt_native.dll`.
- **Runtime:** ~5 to 8 seconds.
- **Output:** `build/csrc/bin/blt_native.dll` and `build/csrc/bin/libblt_native.lib`.

---

## 2. Data Acquisition & Preprocessing

### 2.1 Download TinyStories Corpus
Streams and extracts the curated ~280 MB TinyStories subset from Hugging Face into raw text format.
```powershell
.\.venv\Scripts\python.exe scripts/download_tinystories.py --target-mb 280 --output-file data/raw/tinystories_raw.txt
```
- **What it does:** Downloads stories until the raw byte size reaches ~280 MB.
- **Flags:**
  - `--target-mb 280`: Target uncompressed dataset volume in megabytes.
  - `--output-file`: Destination file path for raw text.
- **Runtime:** ~1 to 3 minutes (depending on network speed).
- **Output:** `data/raw/tinystories_raw.txt` (~280 MB raw UTF-8 text).

---

### 2.2 Preprocess, Deduplicate, and Shard Data
Deduplicates stories using exact 64-bit Rolling Polynomial Hashes and partitions the data into binary byte-level shards (`uint16`).
```powershell
.\.venv\Scripts\python.exe scripts/preprocess_data.py --raw-file data/raw/tinystories_raw.txt --output-dir data/processed --val-fraction 0.04
```
- **What it does:**
  1. Identifies individual stories delimited by `<|endoftext|>`.
  2. Computes 64-bit hash fingerprints to eliminate duplicated stories.
  3. Inserts document boundary tokens (`DOC_BOUNDARY_TOKEN = 256`).
  4. Encodes tokens as `uint16` memory-mapped arrays.
  5. Splits into a 96% training shard and a 4% validation shard.
- **Flags:**
  - `--raw-file` (or `--input-file`): Path to downloaded raw text.
  - `--output-dir`: Directory where binary shards are stored.
  - `--val-fraction 0.04`: Fraction of data reserved for validation (4%).
- **Runtime:** ~15 to 25 seconds.
- **Outputs:**
  - `data/processed/train.bin` (~553.8 MB memmapped binary shard).
  - `data/processed/val.bin` (~23.1 MB memmapped binary shard).

---

## 3. Model Pretraining

### 3.1 Pretrain Byte Entropy Model (~1.5M params)
Trains a lightweight next-byte language model to estimate Shannon entropy $H(x_t)$ for dynamic patch boundary creation.
```powershell
.\.venv\Scripts\python.exe scripts/train_entropy_model.py --config configs/entropy_model_tinystories.yaml
```
- **What it does:**
  - Optimizes a 3-layer, 128-dim, 4-head byte-level transformer on raw bytes.
  - Computes next-byte probability distributions and Shannon entropy $H(x_t) = -\sum p \log_2 p$.
  - Saves the best checkpoint based on validation Bits-Per-Byte (BPB).
- **Specifications:**
  - **Parameters:** 1,515,648 parameters.
  - **Batch Size:** 16 sequences $\times$ 256 bytes.
  - **Optimizer:** Fused AdamW with cosine decay (max LR: `5.0e-4`).
  - **VRAM Usage:** ~450 MB.
  - **Runtime:** ~3 to 5 minutes (2,000 steps).
- **Output:** `checkpoints/entropy_model.pt`.

---

### 3.2 Pretrain 50M Byte Latent Transformer (BLT)
The primary pretraining pipeline for the full research-grade Byte Latent Transformer model.

#### Main Training Command:
```powershell
.\.venv\Scripts\python.exe scripts/train_blt.py --config configs/blt_tinystories_50m.yaml --entropy-checkpoint checkpoints/entropy_model.pt
```
- **What it does:**
  - Instantiates the complete ~50M parameter BLT architecture:
    - **Embeddings:** Byte embeddings (256 dims) + Rolling-hash n-gram embeddings ($n=3,4,5$; 20,000 hash bins each).
    - **Local Byte Encoder:** 1 layer, 256 hidden, local sliding window 256.
    - **Encoder Cross-Attention:** Pools constituent byte states into latent patch queries.
    - **Latent Transformer:** 8 layers, 512 hidden, 8 heads, block-causal attention across patches.
    - **Decoder Cross-Attention:** Byte queries causally attend to latent patch states.
    - **Local Byte Decoder:** 4 layers, 256 hidden, predicting next-byte logits (260 vocab).
  - Uses the pretrained entropy model to dynamically discover variable-sized patch boundaries (approx. monotonic entropy jump rule).
  - Employs 8-bit AdamW optimizer (`bitsandbytes`) and Automatic Mixed Precision (`bf16`/`fp16`).
  - Automatically evaluates validation BPB periodically and preserves the best checkpoint.
- **Flags:**
  - `--config`: Path to the YAML model and training configuration.
  - `--entropy-checkpoint`: Trained entropy model checkpoint used for dynamic patch discovery.
  - `--dry-run`: (Optional) Runs 5 test batches to check VRAM and verify pipeline integrity without training.
  - `--resume`: (Optional) Resumes training from a saved checkpoint file.
- **Hardware Profile:**
  - **VRAM:** ~2.7 to 2.9 GB (configured for batch size 64) or ~750 MB (safe mode batch size 8).
  - **GPU Utilization:** 95–99% Tensor Core saturation.
- **Outputs:**
  - `checkpoints/best_val_bpb.pt` (best validation checkpoint).
  - `checkpoints/checkpoint_step_XXXX.pt` (rotating periodic checkpoints).

#### Quick Dry-Run Verification:
```powershell
.\.venv\Scripts\python.exe scripts/train_blt.py --dry-run
```
- **What it does:** Tests 5 micro-batches and logs peak allocated VRAM. Exits immediately.
- **Runtime:** ~5 seconds.

#### Resuming Training:
```powershell
.\.venv\Scripts\python.exe scripts/train_blt.py --config configs/blt_tinystories_50m.yaml --entropy-checkpoint checkpoints/entropy_model.pt --resume checkpoints/checkpoint_step_1400.pt
```
- **What it does:** Restores model weights, optimizer states, scheduler step, and RNG states to resume uninterrupted.

---

### 3.3 Pretrain Baseline Control Model (Optional)
Trains an equivalent parameter-matched (~40M–46M) standard Transformer to conduct fair comparative experiments.
```powershell
.\.venv\Scripts\python.exe scripts/train_baseline.py --mode byte --steps 3500
```
- **What it does:** Trains a stride-1 byte transformer (no patching) on identical data to reproduce the speedup and compression comparisons from `evaluation.md`.
- **Flags:**
  - `--mode byte`: Character-level raw byte baseline (vocab 260).
  - `--mode bpe`: Subword BPE tokenizer baseline (vocab 32,000).
  - `--steps 3500`: Number of training steps.
- **Runtime:** ~60 minutes.
- **Output:** `checkpoints/baseline_byte/best_val_bpb.pt`.

---

## 4. Autoregressive Text Generation

Generates text from raw bytes using nucleus sampling ($p=0.9$) decoded directly into UTF-8.

### 4.1 Generate with Default Prompt
```powershell
.\.venv\Scripts\python.exe scripts/generate.py --checkpoint checkpoints/best_val_bpb.pt --max-new-tokens 80
```
- **What it does:** Prompts the model with `"Once upon a time"` and generates 80 consecutive bytes autoregressively.
- **Runtime:** ~2 to 5 seconds.
- **Output:** Decoded UTF-8 text printed to console.

---

### 4.2 Custom Prompt with Temperature and Top-P Controls
```powershell
.\.venv\Scripts\python.exe scripts/generate.py --checkpoint checkpoints/best_val_bpb.pt --prompt "One sunny day, a little dog found a big stick" --max-new-tokens 150 --temperature 0.7 --top-p 0.9
```
- **Flags:**
  - `--checkpoint`: Path to trained model weights.
  - `--prompt`: Input seed string to start generation.
  - `--max-new-tokens`: Number of byte tokens to generate.
  - `--temperature`: Softmax temperature (lower = more focused, higher = more creative).
  - `--top-p`: Nucleus sampling cumulative probability threshold.
- **Runtime:** ~5 to 8 seconds.

---

## 5. Evaluation & Benchmarking Suite

Implements the formal benchmarking protocol described in `evaluation.md`.

### 5.1 Run Master Evaluation Suite (All Benchmarks)
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite all
```
- **What it does:** Executes the complete test battery:
  1. **LM Metrics:** Cross-Entropy Loss, Bits-Per-Byte (BPB), Perplexity (PPL), Byte Accuracy (%), Negative Log-Likelihood.
  2. **Patching Metrics:** Average patch size, patch length distribution, compression ratio, entropy-patch Pearson correlation.
  3. **Robustness Suite:** Evaluates degradation across 6 data regimes (Clean, Character Noise, Synthetic Code, Math Equations, Raw URLs, Non-ASCII Unicode).
  4. **Efficiency Benchmark:** Measures Time-To-First-Byte (TTFB), generation latency (ms/token), generation throughput (bytes/sec), and peak VRAM.
  5. **Comparison Table:** Renders Table 27 comparing BLT against baseline.
- **Runtime:** ~2 to 3 minutes.
- **Output:** Formatted markdown tables logged to stdout and saved to evaluation summary.

---

### 5.2 Targeted Sub-Suites:

#### Language Modeling Metrics Only:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite lm
```

#### Patch Distribution & Compression Statistics Only:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite patching
```

#### 6-Category Noise & Robustness Evaluation Only:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite robustness
```

#### Hardware & Throughput Benchmarks Only:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite efficiency
```

#### Direct Model Comparison Only:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite compare
```

---

## 6. Analytical FLOPs Reporting

Computes the theoretical computational cost (FLOPs per byte and GFLOPs per sequence) based on the exact formulas from **Paper Appendix B**.
```powershell
.\.venv\Scripts\python.exe -c "from blt.flops.report import generate_flop_report; print(generate_flop_report())"
```
- **What it does:** Compares analytical FLOP consumption between BLT and standard Byte Transformer across context lengths 512, 1024, 2048, and 4096.
- **Runtime:** < 1 second.
- **Output:**
```
| Sequence (Bytes) | Patches (M) | BLT GFLOPs | BLT FLOPs/Byte | Baseline GFLOPs | Baseline FLOPs/Byte | FLOP Efficiency |
|------------------|-------------|------------|----------------|-----------------|---------------------|-----------------|
| 512              | 114         | 11.302     | 22,073,891     | 30.208          | 59,000,144          | 2.67x faster    |
| 1,024            | 228         | 23.089     | 22,548,131     | 69.006          | 67,388,752          | 2.99x faster    |
| 2,048            | 456         | 48.121     | 23,496,611     | 172.372         | 84,165,968          | 3.58x faster    |
| 4,096            | 911         | 103.926    | 25,372,641     | 482.183         | 117,720,400         | 4.64x faster    |
```

---

## 7. Publication Plots Generation

Renders all **15 publication-quality figures** (at 300 DPI) as specified in `evaluation.md` §28.
```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py --output-dir plots
```
- **What it does:** Visualizes training progression, distribution curves, efficiency frontiers, and memory scaling.
- **Flags:**
  - `--output-dir plots`: Destination folder for `.png` image files.
- **Runtime:** ~5 to 8 seconds.
- **Generated Figures:**
  1. `01_training_loss_vs_steps.png` — Training loss progression curve.
  2. `02_validation_loss_vs_steps.png` — Validation loss evaluation curve.
  3. `03_validation_bpb_vs_steps.png` — Validation Bits-Per-Byte convergence.
  4. `04_train_vs_val_loss.png` — Overfitting diagnostic overlay.
  5. `05_patch_length_distribution.png` — Patch size histogram with mean/median markers.
  6. `06_local_entropy_vs_patch_length.png` — Entropy vs patch length correlation with Pearson $r$.
  7. `07_compression_ratio_distribution.png` — Sequence-wise compression ratio histogram.
  8. `08_bpb_vs_training_compute.png` — Pareto efficiency curve (BPB vs GFLOPs).
  9. `09_bpb_vs_training_time.png` — Compression quality vs wall-clock time.
  10. `10_bpb_vs_peak_vram.png` — Memory footprint vs compression quality.
  11. `11_blt_vs_baseline_bpb.png` — Bar chart comparing BLT vs Baseline BPB.
  12. `12_blt_vs_baseline_throughput.png` — Bar chart comparing inference throughput.
  13. `13_blt_vs_baseline_memory.png` — Bar chart comparing peak VRAM allocation.
  14. `14_context_length_vs_bpb.png` — Context length scaling curve.
  15. `15_context_length_vs_vram.png` — Memory scaling curve vs 4GB safety threshold.

---

## 8. Quick Reference Summary Table

| Task | Primary Command | Est. Runtime | Peak VRAM | Key Output |
|---|---|---|---|---|
| **Test Verification** | `.\.venv\Scripts\pytest.exe -q` | ~6s | < 400 MB | 57 passed tests |
| **Download Data** | `.\.venv\Scripts\python.exe scripts/download_tinystories.py` | ~2 min | < 100 MB | `tinystories_raw.txt` |
| **Preprocess Data** | `.\.venv\Scripts\python.exe scripts/preprocess_data.py --raw-file data/raw/tinystories_raw.txt` | ~20s | < 300 MB | `train.bin`, `val.bin` |
| **Train Entropy Model** | `.\.venv\Scripts\python.exe scripts/train_entropy_model.py` | ~3–5 min | ~450 MB | `checkpoints/entropy_model.pt` |
| **Train 50M BLT** | `.\.venv\Scripts\python.exe scripts/train_blt.py --entropy-checkpoint checkpoints/entropy_model.pt` | ~1–1.5 hr | ~2.7 GB | `checkpoints/best_val_bpb.pt` |
| **Generate Story** | `.\.venv\Scripts\python.exe scripts/generate.py --checkpoint checkpoints/best_val_bpb.pt` | ~5s | ~600 MB | Console output text |
| **Full Evaluation** | `.\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best_val_bpb.pt --suite all` | ~2 min | ~1.1 GB | Benchmark tables |
| **Generate Plots** | `.\.venv\Scripts\python.exe scripts/plot_results.py --output-dir plots` | ~6s | < 300 MB | 15 figures in `plots/` |
