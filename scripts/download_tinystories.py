"""Download script for TinyStories dataset subset.

Streams the specified byte budget (default ~280MB) directly from HuggingFace
and ensures the downloaded file cleanly cuts off at story boundaries (<|endoftext|>).
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import urllib.request
from typing import Optional
from tqdm import tqdm
from utils.logger import get_logger
from utils.custom_exception import DataPipelineError

logger = get_logger("BLT.Download")

DEFAULT_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
DEFAULT_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
STORY_DELIMITER = "<|endoftext|>"


def download_stream(
    url: str,
    output_path: str,
    target_bytes: int,
    chunk_size: int = 65536,
) -> int:
    """Download up to target_bytes from url, continuing until the next story boundary.

    Args:
        url: Remote file URL
        output_path: Local destination file path
        target_bytes: Target byte size threshold
        chunk_size: Streaming chunk size in bytes

    Returns:
        Total bytes written
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info(f"Connecting to {url}...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BLT-Research/1.0"},
    )

    try:
        response = urllib.request.urlopen(req)
    except Exception as e:
        raise DataPipelineError(f"Failed to open connection to {url}: {e}", error_detail=sys)

    total_written = 0
    buffer = bytearray()
    hit_target = False

    pbar = tqdm(
        total=target_bytes,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(output_path)}",
    )

    with open(output_path, "wb") as f_out:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                # EOF reached
                if buffer:
                    f_out.write(buffer)
                    total_written += len(buffer)
                break

            buffer.extend(chunk)

            if not hit_target:
                if total_written + len(buffer) >= target_bytes:
                    hit_target = True

            if hit_target:
                # Search for the nearest story delimiter to cut cleanly
                delim_bytes = STORY_DELIMITER.encode("utf-8")
                last_idx = buffer.rfind(delim_bytes)
                if last_idx != -1:
                    end_pos = last_idx + len(delim_bytes)
                    to_write = buffer[:end_pos]
                    f_out.write(to_write)
                    total_written += len(to_write)
                    pbar.update(len(to_write))
                    logger.info(
                        f"Clean cutoff found at story boundary after {total_written / (1024 * 1024):.2f} MB."
                    )
                    break

            # Flush full chunks
            if len(buffer) >= chunk_size * 2:
                write_len = len(buffer) - chunk_size
                f_out.write(buffer[:write_len])
                total_written += write_len
                pbar.update(write_len)
                buffer = buffer[write_len:]

    pbar.close()
    logger.info(f"Finished downloading to {output_path} ({total_written / (1024 * 1024):.2f} MB written).")
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Download TinyStories dataset subset")
    parser.add_argument(
        "--target-mb",
        type=float,
        default=280.0,
        help="Target size in megabytes to download (default: 280MB)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Directory to save raw text data",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Quick download of 5MB for testing",
    )
    args = parser.parse_args()

    target_mb = 5.0 if args.sample else args.target_mb
    target_bytes = int(target_mb * 1024 * 1024)

    raw_path = os.path.join(args.output_dir, "tinystories_raw.txt")
    logger.info(f"Target download: {target_mb:.1f} MB -> {raw_path}")

    download_stream(DEFAULT_TRAIN_URL, raw_path, target_bytes)


if __name__ == "__main__":
    main()
