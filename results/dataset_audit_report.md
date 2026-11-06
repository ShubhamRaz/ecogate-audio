# Dataset Audit & Split Integrity Report (`results/dataset_audit_report.md`)

## 1. Archive Provenance & Modality Policy
- **Official Dataset**: [Multimodal Computer Vision and Acoustic Bird Detection Dataset for Smart Rice Farming Environments](https://data.mendeley.com/datasets/p6m5tmr477) (DOI: `10.17632/p6m5tmr477.1`, CC BY 4.0).
- **Modality Enforcement**: Strictly **Audio-First**. Visual frames are never paired with audio recordings because no verified temporal synchronization metadata exists.
- **Execution Scope**: Audited the authoritative 8,512-WAV field recording archive mounted in `data/raw/` (SHA-256: `9f9715801b2133f9...`). All 151 synthetic fixtures were quarantined to `data/synthetic_fixture/` to guarantee zero synthetic data contamination in research training and evaluation.

## 2. Preliminary Reported Archive Claims vs. Executed Local Validation Audit

| Audit Metric | Preliminary Reported Archive Observation (Unverified Historical Claim) | Executed Local Validation Audit Result |
| :--- | :---: | :---: |
| Total WAV files scanned | 8512 | 8512 |
| Valid decodable WAV files | — | 8512 |
| Corrupt WAV files flagged & excluded | — | 0 |
| Annotation records inspected | 10723 | 10723 |
| Missing referenced audio paths | 2211 | 2211 |
| Duplicate waveform groups detected | 14 | 37 |
| Files involved in duplicate groups | 29 | 76 |
| ESC-50 catalogue filename matches | 2000 | 0 |

## 3. Group-Safe Partition Integrity Verification
- **Split Integrity Check Passed**: `True`
- **Leaked Duplicate Groups Across Splits**: `0`
- **Leaked Waveform SHA-256 Hashes Across Splits**: `0`
- **Leaked Recording Sessions Across Splits**: `0`
- **Partition Counts**: Train = `5108`, Validation = `1276`, Calibration = `852`, Test = `1276`.
