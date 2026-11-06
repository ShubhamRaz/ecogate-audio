#!/usr/bin/env python3
"""
3. Find Duplicates Script (`scripts/find_duplicates.py`)
Identifies exact file SHA-256 duplicates and decoded float32 waveform duplicates across `data/raw/`.
Saves structured report to `data/audited/duplicates_report.json`.
"""
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    manifest_csv = PROJECT_ROOT / "data" / "manifests" / "manifest_audited.csv"
    if not manifest_csv.exists():
        from scripts.audit_dataset import main as run_audit
        run_audit()

    groups = {}
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = row["duplicate_group_id"]
            groups.setdefault(gid, []).append(row["file_path"])

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    out_path = PROJECT_ROOT / "data" / "audited" / "duplicates_report.json"
    out_path.write_text(
        json.dumps(
            {
                "duplicate_group_count": len(dup_groups),
                "total_files_in_duplicate_groups": sum(len(v) for v in dup_groups.values()),
                "duplicate_groups": dup_groups,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[find_duplicates] Found {len(dup_groups)} duplicate groups ({sum(len(v) for v in dup_groups.values())} files).")


if __name__ == "__main__":
    main()
