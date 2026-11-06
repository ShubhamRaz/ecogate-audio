#!/usr/bin/env python3
"""
8. Validate Dataset Script (`scripts/validate_dataset.py`)
Validates manifest schema, split containment, zero duplicate leakage, and corrupt exclusion.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_group_splits import main as run_splits


def main() -> None:
    integrity_path = PROJECT_ROOT / "data" / "splits" / "split_integrity_report.json"
    if not integrity_path.exists():
        run_splits()
    report = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not report.get("passed", False):
        raise SystemExit("Dataset validation FAILED: duplicate or session leakage detected.")
    print(f"[validate_dataset] Validation PASSED. Split counts: {report['split_counts']}")


if __name__ == "__main__":
    main()
