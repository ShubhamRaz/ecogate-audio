# EcoGate-Audio

A research prototype for audio-only bird-sound classification, open-set rejection, and budgeted expert routing in smart-agriculture recordings.

> **Research status:** the repository is an engineering prototype, not a publication-ready result. The historical metrics under `results/` are explicitly marked as stale. Use a freshly audited copy of the authoritative source archive and rerun the complete experiments before citing performance numbers.

## Current implementation scope

- **MFCC baseline:** a classical regularized classifier over handcrafted MFCC statistics.
- **Student compatibility model:** handcrafted pooled spectral statistics and temporal differences, frozen random projections, and trained output heads. Despite the legacy class name `LightweightStudentCNN`, this is **not a CNN**.
- **Expert compatibility model:** handcrafted spectral summaries with fixed random attention projections, a partially ridge-initialized embedding projection, and trained output heads. Despite legacy names, this is **not a trained AST/transformer**.
- **Open-set detection:** six configurable novelty scores and calibration-split thresholds.
- **Routing:** out-of-fold benefit estimation, calibrated expert invocation thresholds, and call-rate/energy caps. The current model assumes a constant expert invocation cost; it is not per-sample variable-cost optimization.
- **Edge weight storage:** the current INT8 wrapper stores compressed weight arrays but dequantizes them to float32 before matrix multiplication. It is **not integer-only inference**.
- **Power and complexity:** physical energy is not measured without a meter. Exact MAC/FLOP accounting is currently reported as unavailable rather than estimated from parameter count.

## Repository structure

- `src/`: audio processing, audit, models, open-set scoring, routing, calibration, API, and benchmarking.
- `scripts/`: audit, split, training, evaluation, and report generation.
- `configs/`: dataset and experiment settings.
- `data/`: local dataset area; the authoritative WAV archive is not tracked in this repository.
- `docs/`: experimental protocol, contributions, and limitations.
- `frontend/`: React monitoring interface.
- `manuscript/`: provisional manuscript scaffold.
- `results/`: historical reports; see each report's status banner.
- `tests/`: nine integration/unit test methods in the current suite (not an 18-step suite).

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

## Real-data workflow

1. Place the authoritative dataset WAVs under `data/raw/` and ensure the matching annotation file is at `data/manifests/annotations_reference.csv`.
2. Confirm that `data/raw/.ecogate_synthetic_reference_fixture` does **not** exist. Use an empty raw directory before mounting the archive; never merge the reference fixture into research data.
3. Run the audit and split scripts:

```bash
python scripts/audit_dataset.py
python scripts/create_group_splits.py
```

4. Run the complete experiment pipeline:

```bash
python scripts/train_complete_pipeline.py
```

The training pipeline refuses marked synthetic data. Feature caches are reused only when their metadata hash matches the current split manifest; stale or unverified caches are rebuilt.

## Synthetic smoke fixture (tests only)

The fixture is for software checks only and must never be used to report research accuracy:

```bash
python scripts/audit_dataset.py --allow-synthetic-fixture
python scripts/create_group_splits.py --allow-synthetic-fixture
```

After a smoke test, remove the synthetic raw directory, marker, manifests, splits, and feature cache before mounting real data. Do not run publication experiments on that fixture.

## Backend and frontend

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

## Publication checklist

Do not submit a paper based on the committed historical reports. First verify the archive provenance, run the updated test suite and full experiment pipeline, implement genuinely trainable neural backbones if a CNN/AST contribution is claimed, perform actual integer inference before claiming INT8 acceleration, and measure the target hardware if reporting energy or Raspberry Pi deployment claims.
