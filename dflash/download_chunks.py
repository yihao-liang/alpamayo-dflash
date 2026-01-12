#!/usr/bin/env python3
"""Download PhysicalAI-AV dataset by CHUNK.

Downloads N chunks directly, then use ALL clips in those chunks for training.
Each chunk contains ~100 clips.

Usage:
    python download_chunks.py --num-chunks 400 --workers 4
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Features we need (egomotion + 4 cameras)
FEATURES = [
    'labels/egomotion',
    'camera/camera_cross_left_120fov',
    'camera/camera_front_wide_120fov',
    'camera/camera_cross_right_120fov',
    'camera/camera_front_tele_30fov',
]

REPO_ID = 'nvidia/PhysicalAI-Autonomous-Vehicles'


def get_chunks_to_download(num_chunks: int) -> list[int]:
    """Return list of chunk indices to download (0 to num_chunks-1)."""
    chunks = list(range(num_chunks))
    logger.info(f"Will download {num_chunks} chunks (indices 0-{num_chunks-1})")
    logger.info(f"Estimated: ~{num_chunks * 100} clips, ~{num_chunks * 100_000:,} training blocks")
    return chunks


def get_files_to_download(chunks_needed: list[int]) -> list[tuple[str, int]]:
    """Get list of (file_path, size) for all chunks × features."""
    logger.info("Listing repository files...")
    api = HfApi()
    files = list(api.list_repo_tree(REPO_ID, repo_type='dataset', recursive=True))

    chunks_set = set(chunks_needed)
    to_download = []

    for f in files:
        # Check if this file is one of our features
        if not any(feat in f.path for feat in FEATURES):
            continue

        # Extract chunk number
        match = re.search(r'chunk_(\d+)', f.path)
        if not match:
            continue

        chunk_num = int(match.group(1))
        if chunk_num in chunks_set:
            size = f.size if hasattr(f, 'size') and f.size else 0
            to_download.append((f.path, size))

    return to_download


def download_file(file_path: str, cache_dir: str, max_retries: int = 3) -> tuple[str, bool, str]:
    """Download a single file with retries."""
    for attempt in range(max_retries):
        try:
            hf_hub_download(
                REPO_ID,
                file_path,
                repo_type='dataset',
                cache_dir=cache_dir
            )
            return (file_path, True, "")
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                if "429" in error_str or "rate limit" in error_str.lower():
                    wait = 30 * (2 ** attempt)
                    time.sleep(wait)
                else:
                    time.sleep(5)
            else:
                return (file_path, False, error_str)

    return (file_path, False, "Max retries exceeded")


def main():
    parser = argparse.ArgumentParser(description="Download PhysicalAI-AV by chunk")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data/physicalai_av/hf_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output-log",
        type=str,
        default="/data/physicalai_av/chunk_download_log.json",
        help="Log file tracking download progress",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=400,
        help="Number of chunks to download (default: 400, each chunk ~100 clips)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers",
    )
    args = parser.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Determine chunks to download
    chunks_needed = get_chunks_to_download(args.num_chunks)

    # Step 2: Get file list
    files_to_download = get_files_to_download(chunks_needed)
    total_size = sum(size for _, size in files_to_download)

    logger.info(f"Files to download: {len(files_to_download)}")
    logger.info(f"Total size: {total_size / 1e9:.1f} GB")

    # Step 3: Load existing progress
    log_path = Path(args.output_log)
    successful = set()
    failed = []
    if log_path.exists():
        with open(log_path) as f:
            log_data = json.load(f)
            successful = set(log_data.get("successful", []))
            failed = log_data.get("failed", [])
        logger.info(f"Resuming: {len(successful)} already downloaded")

    # Filter out already downloaded
    files_remaining = [(p, s) for p, s in files_to_download if p not in successful]
    remaining_size = sum(s for _, s in files_remaining)
    logger.info(f"Files remaining: {len(files_remaining)} ({remaining_size / 1e9:.1f} GB)")

    if not files_remaining:
        logger.info("All files already downloaded!")
        return

    # Step 4: Download with progress bar
    successful_list = list(successful)
    failed_list = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all downloads
        futures = {
            executor.submit(download_file, path, args.cache_dir): (path, size)
            for path, size in files_remaining
        }

        # Process completions
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading chunks",
            unit="file"
        )

        downloaded_size = 0
        for future in pbar:
            path, size = futures[future]
            file_path, success, error = future.result()

            if success:
                successful_list.append(file_path)
                downloaded_size += size
            else:
                failed_list.append({"path": file_path, "error": error})
                logger.warning(f"Failed: {file_path}: {error}")

            pbar.set_postfix({
                "done": len(successful_list),
                "failed": len(failed_list),
                "GB": f"{downloaded_size/1e9:.1f}"
            })

            # Save progress periodically
            if len(successful_list) % 10 == 0:
                with open(log_path, 'w') as f:
                    json.dump({
                        "successful": successful_list,
                        "failed": failed_list,
                        "chunks_needed": chunks_needed,
                        "total_files": len(files_to_download),
                    }, f, indent=2)

    # Final save
    with open(log_path, 'w') as f:
        json.dump({
            "successful": successful_list,
            "failed": failed_list,
            "chunks_needed": chunks_needed,
            "total_files": len(files_to_download),
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info(f"  Successful: {len(successful_list)}")
    logger.info(f"  Failed: {len(failed_list)}")
    logger.info(f"  Log: {args.output_log}")


if __name__ == "__main__":
    main()
