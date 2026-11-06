#!/usr/bin/env python3
"""
6. Analyze Class Distribution Script (`scripts/analyze_class_distribution.py`)
Computes per-class and per-semantic-category counts and saves CSV/JSON summaries.
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

    counts = {}
    valid_counts = {}
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lbl = row["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
            if row["is_corrupt"].lower() != "true":
                valid_counts[lbl] = valid_counts.get(lbl, 0) + 1

    out_path = PROJECT_ROOT / "data" / "audited" / "class_distribution.json"
    out_path.write_text(
        json.dumps({"total_by_class": counts, "valid_by_class": valid_counts}, indent=2),
        encoding="utf-8",
    )
    print(f"[analyze_class_distribution] Class distribution saved: {valid_counts}")


if __name__ == "__main__":
    main()
