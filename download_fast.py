"""High-speed multi-threaded downloader with resume capability (Windows compatible)."""

import os
import sys
import time
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"
DEST = Path("data/raw/abo-images-small.tar")
EXTRACT_DIR = Path("data/raw/abo-images-small")
TOTAL_SIZE = 3253381120  # 3.25 GB
NUM_WORKERS = 8


def download_part(url: str, part_path: Path, start: int, end: int, max_retries: int = 30) -> int:
    """Downloads a byte range with chunk resume directly into a part file."""
    expected_len = end - start + 1
    retries = 0

    while retries < max_retries:
        current_size = part_path.stat().st_size if part_path.exists() else 0
        if current_size >= expected_len:
            print(f"Part {part_path.name} complete ({expected_len / 1e6:.1f} MB).", flush=True)
            return expected_len

        chunk_start = start + current_size
        chunk_end = end

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={chunk_start}-{chunk_end}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "ab") as f:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:
            retries += 1
            cur_mb = (part_path.stat().st_size if part_path.exists() else 0) / 1e6
            print(f"Part {part_path.name} drop at {cur_mb:.1f}/{expected_len / 1e6:.1f} MB (retry {retries}): {e}", flush=True)
            time.sleep(1.0)

    raise RuntimeError(f"Failed to download {part_path.name} after {max_retries} retries.")


def parallel_download():
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists() and DEST.stat().st_size == TOTAL_SIZE:
        print(f"Archive {DEST} already fully downloaded ({TOTAL_SIZE / 1e9:.2f} GB).", flush=True)
    else:
        chunk_size = (TOTAL_SIZE + NUM_WORKERS - 1) // NUM_WORKERS
        tasks = []
        part_paths = []

        for i in range(NUM_WORKERS):
            start = i * chunk_size
            end = min(start + chunk_size - 1, TOTAL_SIZE - 1)
            part_path = DEST.parent / f"abo_part_{i:02d}.tmp"
            part_paths.append(part_path)
            if start <= end:
                tasks.append((part_path, start, end))

        print(f"Starting parallel resumable download with {len(tasks)} workers ({chunk_size / 1e6:.1f} MB per chunk)...", flush=True)
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(download_part, URL, p, s, e)
                for p, s, e in tasks
            ]
            for future in as_completed(futures):
                future.result()

        elapsed = time.time() - start_time
        print(f"All parts downloaded in {elapsed:.1f}s ({TOTAL_SIZE / (1e6 * elapsed):.1f} MB/s).", flush=True)

        print(f"Assembling {DEST} from {len(part_paths)} parts...", flush=True)
        with open(DEST, "wb") as out_f:
            for p in part_paths:
                with open(p, "rb") as in_f:
                    while True:
                        buf = in_f.read(8 * 1024 * 1024)
                        if not buf:
                            break
                        out_f.write(buf)
                p.unlink(missing_ok=True)

        print(f"Assembled {DEST} successfully ({DEST.stat().st_size} bytes).", flush=True)

    # Extract archive
    if not EXTRACT_DIR.exists():
        print(f"Extracting {DEST} to {EXTRACT_DIR}...", flush=True)
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DEST, "r:*") as tar:
            tar.extractall(path=EXTRACT_DIR)
        print(f"Extraction complete: {EXTRACT_DIR}", flush=True)
    else:
        print(f"Extracted directory already exists: {EXTRACT_DIR}", flush=True)


if __name__ == "__main__":
    parallel_download()
