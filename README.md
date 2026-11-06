# EcoGate-Audio

Source-Robust, Cost-Aware Selective Inference for Open-Set Bird Sound Monitoring in Smart Rice Farming Environments.

## Project Structure
- `src/`: Core Python library (pure NumPy/SciPy audio processing, models, cost-aware router, open-set detector, calibration, API).
- `scripts/`: Data auditing, group-partitioning, training, benchmarking, and reporting scripts.
- `configs/`: YAML configurations for dataset, models, routing policies, and deployment targets.
- `data/`: Reference validation audio dataset (151 clips), audited manifests, and leak-free group splits.
- `docs/`: Comprehensive architecture, dataset cards, research questions, and experimental protocols.
- `frontend/`: React + Vite + Tailwind CSS monitoring interface.
- `manuscript/`: LaTeX publication manuscript scaffold and bibliography.
- `results/`: Verification reports, metrics summaries, hardware benchmarks, and evaluation plots.
- `tests/`: 18-step automated unit test suite.

## Setup & Quickstart

### 1. Python Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend Interface
```bash
cd frontend
npm install
npm run dev
```

### 3. Backend API
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 4. Running the Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

