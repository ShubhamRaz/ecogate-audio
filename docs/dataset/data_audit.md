# Data Audit Methodology & Findings (`docs/dataset/data_audit.md`)

## 1. Preliminary Archive Audit Observations (Historical Reference)
An earlier inspection of the released Mendeley archive (`10.17632/p6m5tmr477.1`) reported:
- **WAV files in released archive**: `8,512`
- **JPEG files**: `763` (`3,875` bounding boxes, class `Bird`)
- **Audio annotation records**: `10,723` (`2,211` missing referenced audio paths)
- **Cross-split duplicate groups**: `14` (`29` file entries involved)
- **Noise filenames matching ESC-50 catalogue**: `2,000` (with 3 sampled recordings verified waveform-identical after decoding; not extrapolated to all 2,000 without per-file verification).

## 2. Automated Verification Pipeline (`src/audit/auditor.py`)
Our audit pipeline reproduces and enforces 15 integrity checks on any mounted raw archive:
1. Recursive WAV/JPEG inventory without modifying raw files.
2. Annotation manifest cross-checking and missing-path logging.
3. Exact file SHA-256 hashing.
4. Decoded float32 waveform SHA-256 hashing (detecting identical audio across distinct WAV headers).
5. Union-Find duplicate group assignment.
6. Sample-rate, duration, and channel distributions.
7. RMS energy, peak amplitude, clipping ratio, and frame-percentile SNR estimation.
8. Corrupt WAV header/frame trapping and exclusion logging (`inclusion_status = excluded`).
9. Regex provenance matching (`^[1-5]-\d{3,6}-[A-Z]-\d{1,2}\.wav$`) for ESC-50 catalogue files.
