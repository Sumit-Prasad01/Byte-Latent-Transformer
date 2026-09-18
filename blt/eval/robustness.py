"""Robustness Evaluation Suite (evaluation.md §19 & BLT Paper §6.1).

Evaluates byte-level model resilience against tokenizer-breaking phenomena:
1. Natural Language (normal English sentences)
2. Noisy Text (AntSpeak, random drops, casing noise, character repeats, irregular whitespace)
3. Structured Text (Python code, JSON, markdown)
4. Numbers & Arithmetic
5. URLs and File Paths
6. Unicode and Emojis (accented Latin, non-Latin scripts, emoji strings)
"""

import random
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from blt.eval.metrics import LanguageModelMetricTracker


def corrupt_antspeak(text: str) -> str:
    """AntSpeak: space-separate characters (e.g. 'hello' -> 'h e l l o')."""
    return " ".join(list(text))


def corrupt_drop(text: str, drop_prob: float = 0.1, seed: int = 42) -> str:
    """Randomly drop characters with probability drop_prob."""
    rng = random.Random(seed)
    return "".join(c for c in text if rng.random() > drop_prob)


def corrupt_random_case(text: str, flip_prob: float = 0.5, seed: int = 42) -> str:
    """Randomly toggle case with probability flip_prob."""
    rng = random.Random(seed)
    res = []
    for c in text:
        if rng.random() < flip_prob:
            res.append(c.lower() if c.isupper() else c.upper())
        else:
            res.append(c)
    return "".join(res)


def corrupt_repeat(text: str, repeat_prob: float = 0.2, max_repeat: int = 4, seed: int = 42) -> str:
    """Randomly repeat characters up to max_repeat times."""
    rng = random.Random(seed)
    res = []
    for c in text:
        if rng.random() < repeat_prob:
            res.append(c * rng.randint(2, max_repeat))
        else:
            res.append(c)
    return "".join(res)


def corrupt_uppercase(text: str) -> str:
    """Capitalize all text."""
    return text.upper()


def get_curated_robustness_data() -> Dict[str, List[str]]:
    """Return standardized evaluation texts across the 6 categories from evaluation.md §19."""
    normal_english = [
        "Once upon a time, a small rabbit lived in a sunny forest near a crystal river.",
        "The girl decided to bake an apple pie for her grandmother on Sunday afternoon.",
        "Every morning the rooster crows to awaken the villagers and herald the sunrise.",
        "They walked along the winding cobblestone path until reaching the old castle gate.",
    ]

    # Noisy variations derived from English sentences
    noisy_text = [
        corrupt_antspeak("The quick brown fox jumps over the lazy dog."),
        corrupt_drop("A delightful afternoon spent reading books in the library garden.", drop_prob=0.15),
        corrupt_random_case("Computational linguistics explores the interface between languages and algorithms."),
        corrupt_repeat("Look at that incredible rainbow stretching across the wide valley!"),
        "A    strange    sentence   with   lots    of    irregular    spaces   and   tabs.",
        "Wait... What?! Is this really happening??? Yes!!! No way...",
    ]

    # Structured code and data
    code_json = [
        '{"user_id": 4912, "name": "Lily", "scores": [98.5, 92.0, 100.0], "active": true}',
        'def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)',
        'class TransformerBlock(nn.Module):\n    def __init__(self, dim: int):\n        super().__init__()\n        self.norm = RMSNorm(dim)',
        '# Header\n- Item 1\n- Item 2\n```python\nprint("Hello")\n```',
    ]

    # Numbers and arithmetic
    numbers_math = [
        "The recipe calls for 2.5 cups of flour, 3/4 teaspoon of salt, and 150ml of milk.",
        "Calculate: 1492 * 38 = 56696, and 123456789 + 987654321 = 1111111110.",
        "Latitude 37.7749 N, Longitude 122.4194 W, Altitude 16.5 meters above sea level.",
        "In the year 2026, total quarterly revenue increased by 14.85% reaching $4,582,100.",
    ]

    # URLs and file paths
    urls_paths = [
        "https://github.com/facebookresearch/byte-latent-transformer/tree/main/csrc",
        "C:\\Users\\sumit\\OneDrive\\Desktop\\Code_PlayGround\\LLM_Engineering\\Byte-Latent-Transformer\\blt\\eval",
        "/usr/local/bin/python3.12 -m pytest tests/test_transformer_layers.py --color=yes",
        "https://arxiv.org/abs/2412.09871?download_pdf=true#section.4.2",
    ]

    # Unicode, accents, non-Latin, and emojis
    unicode_emojis = [
        "Café, naïve, façade, crème brûlée, Noël, and über-cool piñatas! 🎉✨",
        "Bonjour le monde! こんにちは世界！你好世界！Привет, мир! مرحبا بالعالم 🌍",
        "The magical forest was full of fairies 🧚, unicorns 🦄, dragons 🐉, and mushrooms 🍄.",
        "Mathematical symbols: ∀x ∈ ℝ, ∃y: x² + y² = r², ∑_{i=1}^n i = n(n+1)/2, ∇ × B = μ₀J.",
    ]

    return {
        "Normal English": normal_english,
        "Noisy Text": noisy_text,
        "Code & Structured": code_json,
        "Numbers & Math": numbers_math,
        "URLs & Paths": urls_paths,
        "Unicode & Emojis": unicode_emojis,
    }


@torch.no_grad()
def evaluate_robustness_suite(
    model: nn.Module,
    benchmarks: Optional[Dict[str, List[str]]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Dict[str, float]]:
    """Evaluate language modeling metrics across all 6 robustness categories (§19).

    Returns:
        Dictionary mapping category name to {"loss", "bpb", "ppl", "accuracy", "total_bytes"}.
    """
    if benchmarks is None:
        benchmarks = get_curated_robustness_data()

    model.eval()
    model.to(device)
    results = {}

    for category, texts in benchmarks.items():
        tracker = LanguageModelMetricTracker()

        for text in texts:
            raw_bytes = list(text.encode("utf-8"))
            if len(raw_bytes) < 2:
                continue

            tokens = torch.tensor([raw_bytes], dtype=torch.long, device=device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits, _ = model(inputs)
            tracker.update(logits, targets)

        metrics = tracker.compute()
        results[category] = metrics

    return results


def format_robustness_table(results: Dict[str, Dict[str, float]]) -> str:
    """Format robustness evaluation results into markdown table matching evaluation.md §19."""
    headers = ["Dataset Category", "Bits Per Byte (BPB)", "Perplexity (PPL)", "Byte Accuracy (%)", "Evaluated Bytes"]
    rows = []

    for cat, m in results.items():
        rows.append([
            cat,
            f"{m['bpb']:.3f}",
            f"{m['ppl']:.2f}",
            f"{m['accuracy'] * 100.0:.1f}%",
            f"{int(m['total_bytes']):,}",
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, col_widths)) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + data_lines)
