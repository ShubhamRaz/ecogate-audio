# EcoGate-Audio: Cost-Aware, Open-Set Bioacoustic Monitoring at the Edge

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scientifically validated, reproducible research platform for real-time avian acoustic monitoring in smart-agriculture ecosystems. **EcoGate-Audio** orchestrates a tiered inference architecture: an ultra-low-power **Student CNN** runs continuously on edge hardware to filter noise and classify common species, while a high-capacity **Expert Audio Transformer** is conditionally invoked via a cost-calibrated, out-of-fold benefit estimator under strict operational budget constraints.

---

## Key Scientific Contributions & Implemented Architecture

### 1. Authentic Bioacoustic Dataset Provenance
- **Dataset Source:** Canonical South African avian bioacoustic dataset deposited in Mendeley Data (DOI: `10.17632/p6m5tmr477.1`).
- **Archive Reconciliation:** Verified archive containing **strictly 8,512 authentic field recordings** (SHA-256: `9f9715801b2133f9...`).
- **Quarantined Fixtures:** Fully separated from synthetic test fixtures (151 test fixtures quarantined in `data/synthetic_fixture/`).
- **Group-Safe Splitting:** Strict isolation of 37 duplicate waveform groups (76 files) and recording sessions across Train (5,108), Calibration (852), Validation (1,276), and Test (1,276) splits with **zero cross-split data leakage**.

### 2. Genuine Trainable Deep Neural Architectures
- **Lightweight Student CNN (`StudentCNN`):**
  - Input: Log-Mel spectrograms ($64 \times 188$).
  - Architecture: 4-block 2D CNN with BatchNorm2d, GELU activations, MaxPool2d, adaptive average pooling, and dual classification & bird-activity heads.
  - Parameter Complexity: 224,965 trainable parameters (~879 KB FP32 storage).
  - Optimization: AdamW, Cosine Annealing learning rate schedule, gradient norm clipping (1.0), and validation macro-F1 early stopping.
- **Audio Spectrogram Transformer Expert (`ExpertTransformer`):**
  - Architecture: Non-overlapping $8 \times 8$ 2D patch tokenization, learnable `[CLS]` token, sinusoidal 2D positional encodings, 4-layer multi-head self-attention transformer encoder (hidden dim 128, 4 heads), and dual heads.
  - Parameter Complexity: 1,481,221 trainable parameters (~5.78 MB FP32 storage).

### 3. Dynamic Integer Quantization (`qint8`)
- Implemented via `torch.ao.quantization.quantize_dynamic` (`TorchQuantizedStudentModel`).
- Linear projection layers execute true **`torch.qint8` integer matrix operations** on CPU backends at runtime (not merely compressed weight storage).
- Post-quantization probability recalibration performed strictly on the held-out calibration split.

### 4. Calibration-Only Open-Set Recognition
- Evaluates 6 distinct novelty detection methods: Maximum Softmax Probability (MSP), Temperature-Scaled Confidence, Predictive Entropy, Logit Margin, Prototype Distance, and Energy Score.
- Operating thresholds calibrated on the calibration split to maintain 90% target known-species recall.
- Downstream open-set generalization validated via Leave-One-Species-Out cross-validation across all target bird species.
- Environmental noise (2,000 clips) is strictly partitioned from unfamiliar avian taxa (`other_birds`, 2,508 clips).

### 5. Cost-Aware Budgeted Routing
- **Mathematical Foundation:** Theorem 1 proves that when expert invocation incurs a constant computational cost $C_e$, subtracting $C_e$ from predicted sample benefits is order-isomorphic under `argsort` and does not alter sample ranking. Cost awareness is rigorously formulated as an optimal threshold floor ($\tau^* \ge \lambda C_e$) calibrated subject to strict call-rate caps.
- **Out-of-Fold (OOF) Benefit Estimation:** Pre-expert benefit estimator trained on group-held-out cross-validation predictions without test leakage.
- **Quality & Bird-Activity Gates:** Prevents expert dispatch on low-SNR, clipped, or abiotic audio clips.

### 6. Hardware & Energy Benchmarking
- Directly measures batch-1 CPU latency (mean & P95), peak memory, and parameter storage on host platforms.
- Physical electrical energy measurements (Joules) on embedded Raspberry Pi Zero 2 W are documented with full scientific honesty as `not_measured_proxy_cost_units_only`, pending in-line physical power meter instrumentation.

---

## Repository Structure

```
ecogate-audio/
├── configs/               # Dataset, model, and experimental hyperparameters
├── data/
│   ├── audited/           # Provenance reports, duplicate logs, class distributions
│   ├── manifests/         # Split CSVs and reference annotation manifests
│   ├── raw/               # 8,512 authentic Mendeley WAV recordings (gitignored)
│   └── synthetic_fixture/ # 151 quarantined synthetic smoke-test clips
├── docs/                  # Limitations, threats to validity, and research protocols
├── manuscript/            # LaTeX manuscript source (main.tex) and bibliography
├── results/               # Empirical experimental reports & high-resolution plots
│   └── plots/             # Training curves, ROC, risk-coverage, and routing sweeps
├── scripts/               # End-to-end execution, audit, and benchmark scripts
├── src/
│   ├── api/               # FastAPI edge inference server
│   ├── benchmarking/      # Hardware latency and energy accounting
│   ├── calibration/       # Temperature scaling and Platt scaling
│   ├── datasets/          # Group-safe dataset loaders and feature extractors
│   ├── deployment/        # PyTorch dynamic INT8 edge execution engine
│   ├── evaluation/        # Statistical metrics, risk-coverage, bootstrap CIs
│   ├── models/            # Genuine PyTorch StudentCNN and ExpertTransformer
│   ├── open_set/          # Prototype-based and energy-based novelty detectors
│   └── routing/           # Benefit estimators, budget controllers, and routers
└── tests/                 # 15-test comprehensive verification test suite
```

---

## Quickstart & Reproduction

### Environment Setup

We recommend using `uv` for virtual environment management:

```bash
# Create Python 3.11 virtual environment
uv venv .venv --python 3.11

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# (Linux/macOS: source .venv/bin/activate)

# Install PyTorch with CUDA 12.1 support
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
uv pip install -r requirements.txt
```

### Running the Test Suite

Execute the 15-test verification suite covering audio feature extraction, dataset integrity, synthetic data isolation, deep model checkpoints, dynamic INT8 quantization, and budget controllers:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Executing the Complete Research Pipeline

Run the master end-to-end training and evaluation pipeline with the PyTorch backend:

```bash
python scripts/train_complete_pipeline.py --model-backend pytorch --epochs 10 --batch-size 64 --oof-folds 4
```

This single command:
1. Re-audits dataset checksums and validates group splits with zero duplicate leakage.
2. Trains the genuine PyTorch **Student CNN** and **Expert Transformer** models with AdamW and cosine annealing.
3. Performs temperature scaling on the held-out calibration partition.
4. Dynamically quantizes the student to `qint8` and recalibrates on edge data.
5. Fits open-set class prototypes and determines operating thresholds.
6. Generates out-of-fold routing training pairs across 4 folds and trains the benefit estimator.
7. Evaluates 10 routing policies, 6 open-set novelty methods, and leave-one-species-out rotations on held-out test data.
8. Computes group bootstrap 95% confidence intervals preserving session clusters.
9. Exports all publication-grade plots to `results/plots/`.
10. Writes all 6 comprehensive scientific reports to `results/`.

---

## Edge API Deployment

To launch the FastAPI bioacoustic monitoring endpoint:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## Scientific Transparency & Ethical Disclosure

- **No Synthetic Substitution:** Synthetic audio samples are strictly quarantined and prevented from entering research evaluation.
- **No False Quantization Claims:** Quantized inference executes genuine `qint8` integer operations at runtime on CPU backends.
- **Physical Hardware Honesty:** Proxy compute units are reported transparently; theoretical FLOP calculations are never conflated with measured electrical Joules.
