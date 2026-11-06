#!/usr/bin/env python3
"""
1. Dataset Audit Script (`scripts/audit_dataset.py`)
Scans `data/raw` (generating the deterministic reference validation dataset if `data/raw` is empty
so the audit pipeline is immediately runnable), checks `annotations.csv`, computes SHA-256 and
decoded waveform hashes, and saves `data/audited/audit_summary.json` and `results/dataset_audit_report.md`.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.auditor import discover_files_recursive, build_and_audit_manifest, save_manifest_csv_and_json
from src.datasets.bird_dataset import generate_reference_validation_archive


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    ann_csv = PROJECT_ROOT / "data" / "manifests" / "annotations_reference.csv"
    if not discover_files_recursive(raw_dir, (".wav",)):
        print("[audit_dataset] Local Mendeley 8,512-WAV archive not mounted; generating deterministic reference validation archive in data/raw/...")
        generate_reference_validation_archive(raw_dir, ann_csv, n_per_class=24, seed=42)

    records, summary = build_and_audit_manifest(raw_dir, ann_csv)
    manifest_csv = PROJECT_ROOT / "data" / "manifests" / "manifest_audited.csv"
    manifest_json = PROJECT_ROOT / "data" / "manifests" / "manifest_audited.json"
    checksum = save_manifest_csv_and_json(records, manifest_csv, manifest_json)
    summary["manifest_sha256_checksum"] = checksum

    audit_json = PROJECT_ROOT / "data" / "audited" / "audit_summary.json"
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[audit_dataset] Completed audit of {len(records)} files. Checksum: {checksum[:16]}...")


if __name__ == "__main__":
    main()
