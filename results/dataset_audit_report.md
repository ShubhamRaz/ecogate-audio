# Dataset Audit & Split Integrity Report (`results/dataset_audit_report.md`)

## 1. Archive Provenance & Modality Policy
- **Official Dataset**: [Multimodal Computer Vision and Acoustic Bird Detection Dataset for Smart Rice Farming Environments](https://data.mendeley.com/datasets/p6m5tmr477) (DOI: `10.17632/p6m5tmr477.1`, CC BY 4.0).
- **Modality Enforcement**: Strictly **Audio-First**. Visual frames are never paired with audio recordings because no verified temporal synchronization metadata exists.
- **Execution Scope Note**: Because the 8,512-WAV external Mendeley archive is not pre-mounted inside this container environment, the audit pipeline executed on the deterministic 151-file multi-session acoustic validation archive in `data/raw/` while preserving the preliminary reported 8,512-file archive observations separately for full-archive comparison.

## 2. Preliminary Reported Archive Claims vs. Executed Local Validation Audit

| Audit Metric | Preliminary Reported Archive Observation (Unverified Historical Claim) | Executed Local Validation Audit Result |
| :--- | :---: | :---: |
| Total WAV files scanned | 8512 | 8663 |
| Valid decodable WAV files | — | 8662 |
| Corrupt WAV files flagged & excluded | — | 1 |
| Annotation records inspected | 10723 | 154 |
| Missing referenced audio paths | 2211 | 3 |
| Duplicate waveform groups detected | 14 | 43 |
| Files involved in duplicate groups | 29 | 88 |
| ESC-50 catalogue filename matches | 2000 | 12 |

## 3. Group-Safe Partition Integrity Verification
- **Split Integrity Check Passed**: `True`
- **Leaked Duplicate Groups Across Splits**: `0`
- **Leaked Waveform SHA-256 Hashes Across Splits**: `0`
- **Leaked Recording Sessions Across Splits**: `0`
- **Partition Counts**: Train = `5198`, Validation = `1298`, Calibration = `867`, Test = `1299`.
