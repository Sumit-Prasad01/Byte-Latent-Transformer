"""Preprocessing script for TinyStories corpus.

Performs:
1. Story extraction and UTF-8 validation
2. High-performance story deduplication via native C++ dedup engine
3. Clean, leakage-free train/validation split (default 4% val)
4. Document boundary token (256) insertion
5. Serialization to memory-mapped uint16 binary shards (train.bin, val.bin)
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
from utils.logger import get_logger
from utils.custom_exception import DataPipelineError
from utils.seeding import seed_everything
from blt.csrc.bridge import dedup_stories_native

logger = get_logger("BLT.Preprocess")

DOC_BOUNDARY_TOKEN = 256
STORY_DELIMITER = "<|endoftext|>"


def load_raw_stories(input_path: str) -> List[str]:
    """Read raw corpus and split into individual stories."""
    if not os.path.exists(input_path):
        raise DataPipelineError(f"Raw input file not found: {input_path}", error_detail=sys)

    logger.info(f"Reading raw corpus from {input_path}...")
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    raw_stories = content.split(STORY_DELIMITER)
    cleaned_stories = []
    for s in raw_stories:
        s_clean = s.strip()
        if len(s_clean) > 20:  # ignore micro-fragments or pure whitespace
            cleaned_stories.append(s_clean)

    logger.info(f"Loaded {len(cleaned_stories)} candidate stories from {input_path}.")
    return cleaned_stories


def deduplicate_stories(stories: List[str]) -> List[str]:
    """Deduplicate stories using the native C++ 64-bit hashing engine."""
    logger.info(f"Starting deduplication over {len(stories)} stories...")

    # Pack into contiguous bytes with offsets and lengths
    corpus_parts = []
    offsets = []
    lengths = []
    cur_offset = 0

    for s in stories:
        s_bytes = s.encode("utf-8")
        corpus_parts.append(s_bytes)
        offsets.append(cur_offset)
        lengths.append(len(s_bytes))
        cur_offset += len(s_bytes)

    corpus_bytes = b"".join(corpus_parts)
    offsets_arr = np.array(offsets, dtype=np.int64)
    lengths_arr = np.array(lengths, dtype=np.int64)

    keep_mask, unique_count = dedup_stories_native(corpus_bytes, offsets_arr, lengths_arr)

    unique_stories = [stories[i] for i in range(len(stories)) if keep_mask[i] == 1]
    duplicate_count = len(stories) - unique_count
    dup_pct = (duplicate_count / len(stories) * 100.0) if len(stories) > 0 else 0.0

    logger.info(
        f"Deduplication complete: {unique_count} unique stories kept, "
        f"{duplicate_count} duplicates removed ({dup_pct:.1f}% duplicate rate)."
    )
    return unique_stories


def create_splits(
    stories: List[str],
    val_fraction: float = 0.04,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Shuffle stories and create disjoint train and validation splits."""
    seed_everything(seed)
    indices = np.random.permutation(len(stories))

    num_val = max(1, int(len(stories) * val_fraction))
    val_indices = set(indices[:num_val])
    train_indices = set(indices[num_val:])

    # Strict no-leakage assertion
    assert train_indices.isdisjoint(val_indices), "Data leakage detected between train and val splits!"

    train_stories = [stories[i] for i in indices[num_val:]]
    val_stories = [stories[i] for i in indices[:num_val]]

    logger.info(f"Split data: {len(train_stories)} train stories, {len(val_stories)} validation stories.")
    return train_stories, val_stories


def serialize_split_to_bin(
    stories: List[str],
    output_bin_path: str,
    doc_boundary_token: int = DOC_BOUNDARY_TOKEN,
) -> Dict:
    """Encode stories to uint16 tokens with document boundary markers and write to binary shard."""
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)

    token_chunks = []
    total_tokens = 0
    story_lens = []

    for s in tqdm(stories, desc=f"Encoding {os.path.basename(output_bin_path)}"):
        s_bytes = s.encode("utf-8")
        story_lens.append(len(s_bytes))
        # Raw bytes (0-255) followed by doc boundary token (256)
        story_tokens = np.frombuffer(s_bytes, dtype=np.uint8).astype(np.uint16)
        tokens_with_boundary = np.append(story_tokens, np.uint16(doc_boundary_token))
        token_chunks.append(tokens_with_boundary)
        total_tokens += len(tokens_with_boundary)

    logger.info(f"Writing {total_tokens:,} tokens to {output_bin_path}...")
    # Flatten and save as contiguous binary uint16 file
    full_array = np.concatenate(token_chunks) if token_chunks else np.zeros(0, dtype=np.uint16)

    # Use memmap to write efficiently
    mmap = np.memmap(output_bin_path, dtype=np.uint16, mode="w+", shape=(len(full_array),))
    mmap[:] = full_array[:]
    mmap.flush()
    del mmap

    file_size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    logger.info(f"Saved {output_bin_path} ({file_size_mb:.2f} MB).")

    return {
        "num_stories": len(stories),
        "total_tokens": int(total_tokens),
        "file_size_bytes": int(os.path.getsize(output_bin_path)),
        "file_size_mb": round(file_size_mb, 2),
        "avg_story_bytes": round(float(np.mean(story_lens)), 2) if story_lens else 0,
        "max_story_bytes": int(np.max(story_lens)) if story_lens else 0,
        "min_story_bytes": int(np.min(story_lens)) if story_lens else 0,
    }


def preprocess_corpus(
    raw_input_path: str,
    output_dir: str,
    val_fraction: float = 0.04,
    seed: int = 42,
) -> Dict:
    """Full preprocessing pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    raw_stories = load_raw_stories(raw_input_path)
    unique_stories = deduplicate_stories(raw_stories)
    train_stories, val_stories = create_splits(unique_stories, val_fraction=val_fraction, seed=seed)

    train_bin = os.path.join(output_dir, "train.bin")
    val_bin = os.path.join(output_dir, "val.bin")

    train_stats = serialize_split_to_bin(train_stories, train_bin)
    val_stats = serialize_split_to_bin(val_stories, val_bin)

    metadata = {
        "source_raw_file": raw_input_path,
        "doc_boundary_token": DOC_BOUNDARY_TOKEN,
        "total_raw_stories": len(raw_stories),
        "unique_stories": len(unique_stories),
        "val_fraction": val_fraction,
        "seed": seed,
        "train": train_stats,
        "val": val_stats,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Preprocessing finished successfully. Metadata saved to {meta_path}.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess TinyStories raw text into binary shards")
    parser.add_argument(
        "--input-file",
        "--raw-file",
        type=str,
        default=os.path.join("data", "raw", "tinystories_raw.txt"),
        help="Path to raw input text file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("data", "processed"),
        help="Destination directory for binary shards",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.04,
        help="Fraction of unique stories reserved for validation (default: 0.04)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split",
    )
    args = parser.parse_args()

    preprocess_corpus(
        raw_input_path=args.input_file,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
