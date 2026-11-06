#!/usr/bin/env python3
"""
Extract Mendeley Dataset Annotations (`scripts/extract_mendeley_annotations.py`).

Extracts the official `bird_sounds/info.labels` from `data/external/bird_sounds.zip`
and writes the authoritative reference annotation CSV (`data/manifests/annotations_reference.csv`).
Contains all 10,723 annotation entries from the Mendeley release, of which 8,512 are present
in `data/raw/` and 2,211 were omitted from the zip archive by the original dataset authors.
"""
from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_ROOT / "data" / "external" / "bird_sounds.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_CSV = PROJECT_ROOT / "data" / "manifests" / "annotations_reference.csv"

LABEL_MAP = {
    "Common_waxbill": "common_waxbill",
    "RedBilledQuelea": "red_billed_quelea",
    "VillageWeaver": "village_weaver",
    "Yellow-fronted Canary": "yellow_fronted_canary",
    "Chubbs_cis": "other_birds",
    "Bulbul": "other_birds",
    "Other_Birds": "other_birds",
    "Noise": "environmental_noise",
}


def main() -> None:
    if not ARCHIVE_PATH.exists():
        print(f"[ERROR] Mendeley archive not found at {ARCHIVE_PATH}")
        sys.exit(1)

    print(f"Reading official annotations from {ARCHIVE_PATH}...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        labels_raw = zf.read("bird_sounds/info.labels").decode("utf-8")
    data = json.loads(labels_raw)
    files = data.get("files", [])
    print(f"Loaded {len(files):,} raw annotation records from info.labels.")

    raw_wav_map = {p.name: str(p.relative_to(RAW_DIR)).replace("\\", "/") for p in RAW_DIR.rglob("*.wav")}
    print(f"Scanned {len(raw_wav_map):,} WAV files in {RAW_DIR}.")

    rows = []
    present_count = 0
    missing_count = 0

    for rec in files:
        rel_path = rec.get("path", "")
        fname = Path(rel_path).name
        label_obj = rec.get("label", {})
        lbl_str = label_obj.get("label", "") if isinstance(label_obj, dict) else str(label_obj)
        norm_label = LABEL_MAP.get(lbl_str, "other_birds")

        if fname in raw_wav_map:
            file_rel = raw_wav_map[fname]
            present_count += 1
        else:
            file_rel = f"{norm_label}/{fname}"
            missing_count += 1

        rows.append({
            "file_path": file_rel,
            "label": norm_label,
            "archive_path": rel_path,
            "original_label": lbl_str,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "label", "archive_path", "original_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} annotations to {OUT_CSV}.")
    print(f"  Files present in data/raw: {present_count:,}")
    print(f"  Files missing from archive by original author: {missing_count:,}")


if __name__ == "__main__":
    main()

