#!/usr/bin/env python3
"""
Mendeley Data Archive Downloader for EcoGate-Audio.
Downloads the full 1.91 GB `bird_sounds.zip` and optional `bird-Images.zip` from Mendeley Data
(DOI: 10.17632/p6m5tmr477.1) into `data/external/` with progress bar and streaming chunks.
"""
import sys
import shutil
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "bird_sounds.zip": {
        "url": "https://data.mendeley.com/public-files/datasets/p6m5tmr477/files/d9c7dd33-ab15-4eb5-985d-d71935ebfc1a/file_downloaded",
        "size_bytes": 2001696389,
        "description": "Full 8,512-file acoustic recordings dataset (~1.91 GB)",
    },
    "bird-Images.zip": {
        "url": "https://data.mendeley.com/public-files/datasets/p6m5tmr477/files/f02afef8-e77e-48e6-9ab3-d5651ee46299/file_downloaded",
        "size_bytes": 21262272,
        "description": "763 camera trap JPEG images (~20.3 MB)",
    },
}


def download_file(url: str, dest_path: Path, expected_size: int, chunk_size: int = 2 * 1024 * 1024) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    headers = {"User-Agent": "EcoGateAudio-ResearchDownloader/1.0"}
    downloaded = 0
    if temp_path.exists():
        downloaded = temp_path.stat().st_size
        if downloaded < expected_size:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"Resuming download from byte {downloaded} ({downloaded / (1024*1024):.1f} MB)...")
        elif downloaded == expected_size:
            print("Download already complete in temporary file. Moving...")
            temp_path.rename(dest_path)
            return

    timeout_cfg = httpx.Timeout(180.0, connect=30.0)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout_cfg) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP download failed with status {resp.status_code}")

        if resp.status_code == 200 and downloaded > 0:
            # Server ignored Range header and sent full payload from start
            downloaded = 0

        mode = "ab" if (resp.status_code == 206 and downloaded > 0) else "wb"
        total_size = expected_size
        last_logged_mb = 0

        with open(temp_path, mode) as f:
            for chunk in resp.iter_bytes(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    cur_mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    pct = (downloaded / total_size) * 100.0 if total_size > 0 else 0
                    if cur_mb - last_logged_mb >= 50.0 or downloaded >= total_size:
                        print(f"Downloaded: {cur_mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)", flush=True)
                        last_logged_mb = cur_mb

    print()
    temp_path.rename(dest_path)
    print(f"Successfully saved to {dest_path}")


def main() -> None:
    include_images = "--with-images" in sys.argv
    keys_to_download = ["bird_sounds.zip"]
    if include_images:
        keys_to_download.append("bird-Images.zip")

    for key in keys_to_download:
        info = ARCHIVES[key]
        dest = EXTERNAL_DIR / key
        if dest.exists() and dest.stat().st_size == info["size_bytes"]:
            print(f"Archive '{key}' already exists and size matches ({info['size_bytes'] / (1024*1024):.1f} MB). Skipping.")
            continue

        print(f"\n--- Downloading {key} ({info['description']}) ---")
        download_file(info["url"], dest, info["size_bytes"])


if __name__ == "__main__":
    main()

