#!/usr/bin/env python3
"""
7. Create Group Splits Script (`scripts/create_group_splits.py`)
Partitions dataset into train (60%), val (15%), calibration (10%), and test (15%) splits
preserving duplicate groups and recording sessions within the same partition.
"""
import json
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.auditor import (
    discover_files_recursive,
    build_and_audit_manifest,
    create_group_safe_splits,
    save_manifest_csv_and_json,
    verify_split_integrity,
)
from src.datasets.bird_dataset import generate_reference_validation_archive


def main() -> None:
    cfg_path = PROJECT_ROOT / "configs" / "data" / "dataset.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ratios = cfg["splitting"]["ratios"]
    seed = int(cfg["splitting"]["seed"])

    raw_dir = PROJECT_ROOT / "data" / "raw"
    ann_csv = PROJECT_ROOT / "data" / "manifests" / "annotations_reference.csv"
    if not discover_files_recursive(raw_dir, (".wav",)):
        generate_reference_validation_archive(raw_dir, ann_csv, n_per_class=24, seed=seed)

    records, _ = build_and_audit_manifest(raw_dir, ann_csv)
    split_records = create_group_safe_splits(records, ratios=ratios, seed=seed)
    integrity = verify_split_integrity(split_records)
    if not integrity["passed"]:
        raise RuntimeError(f"Split integrity check failed: {integrity}")

    split_csv = PROJECT_ROOT / "data" / "splits" / "group_splits_manifest.csv"
    split_json = PROJECT_ROOT / "data" / "splits" / "group_splits_manifest.json"
    checksum = save_manifest_csv_and_json(split_records, split_csv, split_json)
    integrity["split_manifest_sha256"] = checksum

    (PROJECT_ROOT / "data" / "splits" / "split_integrity_report.json").write_text(
        json.dumps(integrity, indent=2),
        encoding="utf-8",
    )
    print(f"[create_group_splits] Group splits verified (passed={integrity['passed']}): {integrity['split_counts']}")


if __name__ == "__main__":
    main()
