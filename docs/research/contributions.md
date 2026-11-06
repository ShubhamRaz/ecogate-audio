# Proposed Research Contributions

The current repository contains a reproducible engineering prototype. The following are candidate contributions to validate, not claims of demonstrated superiority.

1. **Audio data audit and provenance-aware partitions:** SHA-256 and decoded waveform hashing, corrupt/missing-path logging, and duplicate/session-connected components for train, validation, calibration, and test partitions. Unverified source metadata must remain labelled unknown.
2. **Four-way acoustic decision interface:** explicit output states for `known_species`, `bird_unfamiliar`, `nonbird`, and `unresolved`, combining a quality gate, bird-activity score, and open-set novelty scores.
3. **Out-of-fold expert-benefit prediction:** a ridge benefit estimator trained on group-held-out predictions, evaluated against confidence, entropy, margin, prototype, and random routing baselines. The current inference cost is constant per expert call; the correct claim is budgeted benefit routing, not per-sample variable-cost optimization.
4. **Reproducible evaluation and service tooling:** calibration-split thresholding, group-aware bootstrap, selective-risk evaluation, FastAPI inference interface, and a React monitoring dashboard.

## What is not yet a contribution

- The current student compatibility class is not a CNN; the expert compatibility class is not a trained AST/transformer. They use handcrafted feature extraction, frozen/random transformations, and fitted output heads.
- The current INT8 path stores quantized weights but converts them back to float32 for inference. It does not demonstrate integer compute acceleration.
- The committed numerical reports are historical and marked non-publication-ready until the corrected pipeline has been rerun on the verified archive.
