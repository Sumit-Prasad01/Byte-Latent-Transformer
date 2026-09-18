"""CLI generation script for Byte Latent Transformer (BLT).

Generates text autoregressively from a prompt using nucleus sampling,
decoding raw bytes directly into UTF-8 text.
"""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import torch

from blt.model.blt_model import ByteLatentTransformer
from blt.patching.entropy_model import ByteEntropyModel
from utils.logger import get_logger

logger = get_logger("BLT.Generate")


def load_model(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ByteLatentTransformer:
    """Instantiate and optionally load weights into ByteLatentTransformer."""
    logger.info(f"Loading BLT model from config {config_path} on {device}...")
    model = ByteLatentTransformer.from_config(config_path)

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def generate_text(
    model: ByteLatentTransformer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Generate text from a prompt string."""
    prompt_bytes = list(prompt.encode("utf-8"))
    prompt_tensor = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

    start_t = time.perf_counter()
    with torch.no_grad():
        out_tokens = model.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    elapsed_t = time.perf_counter() - start_t

    raw_bytes = bytes([b for b in out_tokens[0].cpu().tolist() if b < 256])
    generated_text = raw_bytes.decode("utf-8", errors="replace")

    speed = max_new_tokens / max(1e-5, elapsed_t)
    logger.info(f"Generated {max_new_tokens} bytes in {elapsed_t:.2f}s ({speed:.1f} bytes/sec)")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Autoregressive text generation with BLT.")
    parser.add_argument("--config", type=str, default="configs/blt_tinystories_50m.yaml", help="Model YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little girl named Lily.", help="Seed prompt")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Max new byte tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    model = load_model(args.config, args.checkpoint, device=args.device)
    output = generate_text(
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print("\n" + "=" * 50)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 50)
    print("GENERATION:")
    print(output)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
