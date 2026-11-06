# Reproducibility Instructions (`docs/experiments/reproducibility_guide.md`)

## 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Step-by-Step Pipeline Execution Commands
```bash
# Step 1: Audit dataset, verify provenance, and build manifests
python3 scripts/audit_dataset.py
python3 scripts/find_duplicates.py
python3 scripts/audit_noise_provenance.py
python3 scripts/analyze_audio.py
python3 scripts/analyze_class_distribution.py

# Step 2: Create duplicate-safe group splits and validate zero leakage
python3 scripts/create_group_splits.py
python3 scripts/validate_dataset.py

# Step 3: Run complete training, OOF routing, open-set rotations, benchmarking & reports
python3 scripts/train_complete_pipeline.py

# Step 4: Run automated test suite
python3 -m unittest discover -s tests -p "test_*.py" -v

# Step 5: Launch FastAPI Monitoring Backend
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
