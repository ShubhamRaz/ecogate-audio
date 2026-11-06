#!/usr/bin/env python3
"""
2. Build Manifest Script (`scripts/build_manifest.py`)
Builds or refreshes the canonical dataset manifest CSV and JSON files in `data/manifests/`.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_dataset import main as run_audit

if __name__ == "__main__":
    run_audit()
