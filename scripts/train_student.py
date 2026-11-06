#!/usr/bin/env python3
"""Train Lightweight Student CNN & MFCC Baseline (`train_student.py`)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_complete_pipeline import main

if __name__ == "__main__":
    main()
