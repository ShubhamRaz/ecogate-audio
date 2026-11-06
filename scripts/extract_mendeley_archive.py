#!/usr/bin/env python3
"""
Extract Mendeley bird_sounds.zip archive into data/raw/ with label normalisation.

Archive structure:
  bird_sounds/training/<Label>.<rest>.wav
  bird_sounds/testing/<Label>.<rest>.wav
  bird_sounds/info.labels  (single JSON object with 'files' list)

Label mapping (archive label → project folder):
  Common_waxbill        → common_waxbill
  RedBilledQuelea       → red_billed_quelea
  VillageWeaver         → village_weaver
  Yellow-fronted Canary → yellow_fronted_canary
  Chubbs_cis            → other_birds   (unlisted 5th species)
  Bulbul                → other_birds   (non-focal species)
  Other_Birds           → other_birds
  Noise                 → environmental_noise

Usage:
  .venv\\Scripts\\python.exe scripts\\extract_mendeley_archive.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_ROOT / "data" / "external" / "bird_sounds.zip"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "raw"

LABEL_MAP: dict[str, str] = {
    "Common_waxbill":         "common_waxbill",
    "RedBilledQuelea":        "red_billed_quelea",
    "VillageWeaver":          "village_weaver",
    "Yellow-fronted Canary":  "yellow_fronted_canary",
    "Chubbs_cis":             "other_birds",   # unlisted 5th species → pooled
    "Bulbul":                 "other_birds",   # non-focal species → pooled
    "Other_Birds":            "other_birds",
    "Noise":                  "environmental_noise",
}

SKIP_PATTERNS = ("__MACOSX", ".DS_Store", ".part")


def _should_skip(name: str) -> bool:
    return any(pat in name for pat in SKIP_PATTERNS)


def _load_labels_from_manifest(zf: zipfile.ZipFile) -> dict[str, str]:
    """
    Read bird_sounds/info.labels (single JSON object with 'files' array).
    Returns mapping: archive filename → project folder name.
    Keys are the basename only (as stored in the zip under training/ or testing/).
    """
    labels_entry = "bird_sounds/info.labels"
    try:
        raw = zf.read(labels_entry).decode("utf-8")
    except KeyError:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    files = data.get("files", [])
    path_to_folder: dict[str, str] = {}
    unknown: set[str] = set()

    for rec in files:
        # rec["path"] is like "testing/<filename>" or "training/<filename>"
        rel_path: str = rec.get("path", "")
        label_obj = rec.get("label", {})
        label_str: str = label_obj.get("label", "") if isinstance(label_obj, dict) else str(label_obj)
        if not rel_path or not label_str:
            continue
        # Build full archive path
        archive_path = f"bird_sounds/{rel_path}"
        mapped = LABEL_MAP.get(label_str)
        if mapped is None:
            unknown.add(label_str)
            mapped = "other_birds"
        path_to_folder[archive_path] = mapped

    if unknown:
        print(f"[WARN] Unknown labels in manifest (mapped → other_birds): {sorted(unknown)}")
    return path_to_folder


def _infer_folder_from_filename(filename: str) -> str | None:
    """
    Fallback: infer label from filename prefix.
    Filenames look like: 'Yellow-fronted Canary.sel.3017.ch01...'
    The label is everything before the first '.' that isn't part of a known label.
    We try each known label as a prefix match.
    """
    # Sort by length descending so longer labels match first (e.g. "Yellow-fronted Canary")
    for label in sorted(LABEL_MAP, key=len, reverse=True):
        if filename.startswith(label + ".") or filename.startswith(label + " "):
            return LABEL_MAP[label]
    return None


def extract(dry_run: bool = False) -> None:
    if not ARCHIVE_PATH.exists():
        print(f"[ERROR] Archive not found: {ARCHIVE_PATH}")
        sys.exit(1)

    print(f"Opening archive: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / 1e9:.2f} GB)")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        all_names = zf.namelist()
        print(f"  Total entries in archive: {len(all_names):,}")

        manifest_map = _load_labels_from_manifest(zf)
        print(f"  Loaded {len(manifest_map):,} label records from info.labels")

        wav_names = [n for n in all_names if n.lower().endswith(".wav") and not _should_skip(n)]
        print(f"  WAV files to process: {len(wav_names):,}\n")

        # Pre-create output dirs
        if not dry_run:
            for folder in set(LABEL_MAP.values()):
                (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

        extracted_counts: Counter[str] = Counter()
        skipped_counts: Counter[str] = Counter()
        label_source: Counter[str] = Counter()

        for idx, archive_path in enumerate(wav_names):
            # 1. Try manifest lookup (exact archive path)
            folder = manifest_map.get(archive_path)
            if folder is not None:
                label_source["manifest"] += 1
            else:
                # 2. Fallback: infer from filename prefix
                filename = Path(archive_path).name
                folder = _infer_folder_from_filename(filename)
                if folder is not None:
                    label_source["filename_prefix"] += 1
                else:
                    label_source["unknown"] += 1
                    skipped_counts["no_label"] += 1
                    print(f"  [SKIP] Cannot determine label: {archive_path}")
                    continue

            # Destination: flat per-class directory, keep only the basename
            filename = Path(archive_path).name
            dest = OUTPUT_DIR / folder / filename

            if dest.exists():
                skipped_counts["already_exists"] += 1
                continue

            if dry_run:
                extracted_counts[folder] += 1
                continue

            try:
                with zf.open(archive_path) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_counts[folder] += 1
            except Exception as exc:
                print(f"  [WARN] Failed to extract {archive_path}: {exc}")
                skipped_counts["error"] += 1

            if (idx + 1) % 1000 == 0:
                total_so_far = sum(extracted_counts.values())
                print(f"  Progress: {idx + 1:,}/{len(wav_names):,} processed  |  {total_so_far:,} written")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    mode = "[DRY RUN] " if dry_run else ""
    print(f"{mode}Extraction Summary")
    print("=" * 60)
    print(f"  Label source breakdown:   {dict(label_source)}")
    print(f"  Skipped:                  {dict(skipped_counts)}")
    print("\n  Counts by project folder:")
    total = 0
    for folder in sorted(extracted_counts):
        count = extracted_counts[folder]
        total += count
        print(f"    {folder:<35} {count:>6,}")
    print(f"    {'TOTAL':<35} {total:>6,}")
    print("=" * 60)
    if not dry_run:
        print(f"\nFiles written to: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Mendeley bird_sounds.zip to data/raw/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files without writing to disk")
    args = parser.parse_args()
    extract(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
