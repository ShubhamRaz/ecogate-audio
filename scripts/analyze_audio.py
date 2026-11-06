#!/usr/bin/env python3
"""
5. Analyze Audio Script (`scripts/analyze_audio.py`)
Summarizes sample rate, duration, channel count, SNR, RMS energy, and clipping distributions.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    manifest_csv = PROJECT_ROOT / "data" / "manifests" / "manifest_audited.csv"
    if not manifest_csv.exists():
        from scripts.audit_dataset import main as run_audit
        run_audit()

    snrs, rms_vals, clips, durs = [], [], [], []
    corrupt = 0
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_corrupt"].lower() == "true":
                corrupt += 1
                continue
            snrs.append(float(row["estimated_snr_db"]))
            rms_vals.append(float(row["rms_energy"]))
            clips.append(float(row["clipping_ratio"]))
            durs.append(float(row["duration_sec"]))

    report = {
        "valid_audio_files": len(snrs),
        "corrupt_audio_files": corrupt,
        "snr_db_summary": {
            "mean": round(float(np.mean(snrs)), 3),
            "std": round(float(np.std(snrs)), 3),
            "p10": round(float(np.percentile(snrs, 10)), 3),
            "p90": round(float(np.percentile(snrs, 90)), 3),
        },
        "rms_energy_mean": round(float(np.mean(rms_vals)), 5),
        "max_clipping_ratio": round(float(np.max(clips)), 5),
        "duration_mean_sec": round(float(np.mean(durs)), 4),
    }
    out_path = PROJECT_ROOT / "data" / "audited" / "audio_signal_analysis.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[analyze_audio] Analyzed {len(snrs)} valid clips (mean SNR: {report['snr_db_summary']['mean']} dB).")


if __name__ == "__main__":
    main()
