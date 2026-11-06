"""
Dataset representations, train-derived feature normalization, and synthetic benchmark/smoke
dataset generator for reproducible pipeline verification when the full 8,512-file Mendeley
archive is not locally mounted.
"""
from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.audio_processor import AudioProcessor, AudioPreprocessConfig

KNOWN_SPECIES = [
    "common_waxbill",
    "red_billed_quelea",
    "village_weaver",
    "yellow_fronted_canary",
]

ALL_DATASET_CATEGORIES = KNOWN_SPECIES + [
    "other_birds",
    "environmental_noise",
]

LABEL_TO_INDEX = {name: idx for idx, name in enumerate(KNOWN_SPECIES)}
INDEX_TO_LABEL = {idx: name for idx, name in enumerate(KNOWN_SPECIES)}


@dataclass
class FeatureSample:
    file_path: str
    label: str
    known_class_idx: int  # 0..3 for known species, -1 for unknown bird / nonbird
    is_bird: int          # 1 for known_species or other_birds, 0 for environmental_noise
    semantic_target: str  # known_species, bird_unfamiliar, nonbird, unresolved
    split: str
    duplicate_group_id: str
    source_id: str
    log_mel: np.ndarray
    mfcc_vector: np.ndarray
    snr_db: float
    clipping_ratio: float
    rms_energy: float
    source_type: str = "local_archive_source_unverified"


class FeatureNormalizer:
    """Train-derived standard scaler for Log-Mel spectrograms and MFCC vectors."""

    def __init__(self) -> None:
        self.mel_mean: Optional[np.ndarray] = None
        self.mel_std: Optional[np.ndarray] = None
        self.mfcc_mean: Optional[np.ndarray] = None
        self.mfcc_std: Optional[np.ndarray] = None

    def fit(self, train_samples: List[FeatureSample]) -> "FeatureNormalizer":
        if not train_samples:
            raise ValueError("Cannot fit FeatureNormalizer on empty train_samples.")
        mels = np.stack([s.log_mel for s in train_samples], axis=0)  # (N, M, T)
        mfccs = np.stack([s.mfcc_vector for s in train_samples], axis=0)  # (N, D)
        self.mel_mean = np.mean(mels, axis=(0, 2), keepdims=True).squeeze(0)  # (M, 1)
        self.mel_std = np.std(mels, axis=(0, 2), keepdims=True).squeeze(0) + 1e-6
        self.mfcc_mean = np.mean(mfccs, axis=0)
        self.mfcc_std = np.std(mfccs, axis=0) + 1e-6
        return self

    def transform_mel(self, log_mel: np.ndarray) -> np.ndarray:
        if self.mel_mean is None or self.mel_std is None:
            return log_mel
        return ((log_mel - self.mel_mean) / self.mel_std).astype(np.float32)

    def transform_mfcc(self, mfcc_vec: np.ndarray) -> np.ndarray:
        if self.mfcc_mean is None or self.mfcc_std is None:
            return mfcc_vec
        return ((mfcc_vec - self.mfcc_mean) / self.mfcc_std).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mel_mean": self.mel_mean.tolist() if self.mel_mean is not None else None,
            "mel_std": self.mel_std.tolist() if self.mel_std is not None else None,
            "mfcc_mean": self.mfcc_mean.tolist() if self.mfcc_mean is not None else None,
            "mfcc_std": self.mfcc_std.tolist() if self.mfcc_std is not None else None,
        }


def map_label_to_semantics(label: str, snr_db: float, clipping_ratio: float) -> Tuple[int, int, str]:
    """
    Map raw 6-category dataset label + quality metrics to:
      (known_class_idx, is_bird, semantic_target)
    """
    if clipping_ratio > 0.15 or snr_db < -8.0:
        return (-1, 0, "unresolved")
    if label in LABEL_TO_INDEX:
        return (LABEL_TO_INDEX[label], 1, "known_species")
    if label == "other_birds":
        return (-1, 1, "bird_unfamiliar")
    return (-1, 0, "nonbird")


def generate_reference_validation_archive(
    raw_dir: Path,
    annotations_csv: Path,
    n_per_class: int = 24,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a realistic, deterministic acoustic validation dataset inside `raw_dir`
    with distinct class-specific harmonic/chirp signatures, realistic field noise,
    deliberate duplicate pairs, ESC-50 formatted noise files, one corrupt WAV, and
    missing annotation entries so the entire audit, OOF routing, open-set, and
    evaluation pipeline can be executed and verified end-to-end when the 8,512-file
    external Mendeley archive is not mounted.
    """
    rng = np.random.RandomState(seed)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Explicit provenance marker: fixture audio must never be mistaken for research data.
    (raw_dir / ".ecogate_synthetic_reference_fixture").write_text(
        "synthetic_reference_validation=true\\nseed=" + str(seed) + "\\n", encoding="utf-8"
    )
    sr = 32000
    dur = 3.0
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)

    # Distinct acoustic synthesis profiles per species (fundamental freq + FM rate + harmonic ratios)
    profiles = {
        "common_waxbill": {"f0": 3400.0, "fm": 18.0, "bw": 450.0, "harm2": 0.45, "burst_hz": 5.0},
        "red_billed_quelea": {"f0": 2200.0, "fm": 32.0, "bw": 600.0, "harm2": 0.65, "burst_hz": 8.0},
        "village_weaver": {"f0": 1650.0, "fm": 25.0, "bw": 500.0, "harm2": 0.55, "burst_hz": 6.5},
        "yellow_fronted_canary": {"f0": 4300.0, "fm": 12.0, "bw": 700.0, "harm2": 0.35, "burst_hz": 4.0},
        "other_birds": {"f0": 5400.0, "fm": 42.0, "bw": 900.0, "harm2": 0.50, "burst_hz": 10.0},
    }

    annotation_rows = []
    created_files = 0

    for label in ALL_DATASET_CATEGORIES:
        cls_dir = raw_dir / label
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_per_class):
            sess_idx = i // 3  # 3 clips per session to test session grouping
            if label == "environmental_noise":
                # Use ESC-50 filename pattern for half of the noise clips
                if i % 2 == 0:
                    fold = (i % 5) + 1
                    fname = f"{fold}-{10000 + i * 10:06d}-A-40.wav"
                else:
                    fname = f"field_noise_sess{sess_idx:02d}_clip{i:03d}.wav"
                # Colored environmental noise + wind/rain low-freq rumble
                white = rng.normal(0.0, 0.15, size=t.size)
                rumble = 0.20 * np.sin(2.0 * np.pi * rng.uniform(80.0, 240.0) * t)
                wav_data = white + rumble
            else:
                p = profiles[label]
                fname = f"{label}_sess{sess_idx:02d}_clip{i:03d}.wav"
                # Add subtle cross-species difficulty on 25% of clips so student makes errors that expert can fix
                hard_case = (i % 4 == 3)
                f0_jitter = rng.normal(0.0, 90.0 if not hard_case else 260.0)
                f0 = p["f0"] + f0_jitter
                inst_freq = f0 + p["bw"] * np.sin(2.0 * np.pi * p["fm"] * t + rng.uniform(0, 2 * np.pi))
                phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
                envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * p["burst_hz"] * t)) ** 2
                # Fine-grained rapid trill modulation present in true species (captured by high-res Expert)
                fine_trill = 0.25 * np.sin(2.0 * np.pi * (f0 * 1.5) * t) * np.cos(2.0 * np.pi * (p["fm"] * 2.3) * t)
                noise_std = 0.06 if not hard_case else 0.18
                bg = rng.normal(0.0, noise_std, size=t.size)
                wav_data = envelope * (0.55 * np.sin(phase) + p["harm2"] * np.sin(2.0 * phase) + fine_trill) + bg

            wav_data = np.clip(wav_data, -0.98, 0.98)
            pcm16 = (wav_data * 32767.0).astype(np.int16)
            out_path = cls_dir / fname
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm16.tobytes())
            created_files += 1
            annotation_rows.append({"file_path": f"{label}/{fname}", "label": label})

            # Inject exact duplicate on i == 0 for each class to test duplicate detection & group splitting
            if i == 0:
                dup_fname = f"{label}_duplicate_copy.wav"
                dup_path = cls_dir / dup_fname
                with wave.open(str(dup_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm16.tobytes())
                created_files += 1
                annotation_rows.append({"file_path": f"{label}/{dup_fname}", "label": label})

    # Add 1 corrupt WAV file to verify corrupt audio detection
    corrupt_path = raw_dir / "environmental_noise" / "corrupt_sensor_drop.wav"
    corrupt_path.write_bytes(b"RIFF_CORRUPT_HEADER_ONLY")
    annotation_rows.append({"file_path": "environmental_noise/corrupt_sensor_drop.wav", "label": "environmental_noise"})

    # Add 3 missing referenced audio paths in the annotations CSV to verify missing-path detection
    for m_idx in range(3):
        annotation_rows.append({
            "file_path": f"common_waxbill/missing_field_clip_{m_idx:03d}.wav",
            "label": "common_waxbill",
        })

    annotations_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(annotations_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "label"])
        writer.writeheader()
        writer.writerows(annotation_rows)

    return {
        "created_wav_files": created_files + 1,
        "annotation_rows": len(annotation_rows),
        "missing_references_injected": 3,
        "corrupt_files_injected": 1,
    }


def load_dataset_from_manifest(
    manifest_csv: Path,
    raw_dir: Path,
    processor: Optional[AudioProcessor] = None,
    allow_synthetic_fixture: bool = False,
) -> List[FeatureSample]:
    """Load all valid records from a manifest CSV and extract their acoustic representations."""
    processor = processor or AudioProcessor()
    samples: List[FeatureSample] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    source_types = {row.get("source_type", "local_archive_source_unverified") for row in rows}
    if "synthetic_reference_fixture" in source_types and not allow_synthetic_fixture:
        raise ValueError("Synthetic reference data are blocked from research loading; pass allow_synthetic_fixture=True only in tests/smoke runs.")
    if (Path(raw_dir) / ".ecogate_synthetic_reference_fixture").exists() and not allow_synthetic_fixture:
        raise ValueError("Synthetic fixture marker found in raw_dir; synthetic clips cannot be used for research metrics.")
    total_rows = len(rows)
    for idx, row in enumerate(rows):
        if (idx + 1) % 1000 == 0 or idx + 1 == total_rows:
            print(f"[load_dataset] Processed {idx + 1}/{total_rows} audio clips ({len(samples)} valid loaded)...", flush=True)
        if row.get("is_corrupt", "False").lower() == "true":
            continue
        split = row.get("split", "unassigned")
        if split == "excluded":
            continue
        wav_path = Path(raw_dir) / row["file_path"]
        if not wav_path.exists():
            continue
        res = processor.process_file(wav_path)
        if not res["valid"]:
            continue
        lbl = row["label"]
        snr = float(row.get("estimated_snr_db", res["quality"]["estimated_snr_db"]))
        clip_r = float(row.get("clipping_ratio", res["quality"]["clipping_ratio"]))
        rms = float(row.get("rms_energy", res["quality"]["rms_energy"]))
        k_idx, is_bird, sem_target = map_label_to_semantics(lbl, snr, clip_r)
        samples.append(
            FeatureSample(
                file_path=row["file_path"],
                label=lbl,
                known_class_idx=k_idx,
                is_bird=is_bird,
                semantic_target=sem_target,
                split=split,
                duplicate_group_id=row.get("duplicate_group_id", "unknown"),
                source_id=row.get("source_id", "unknown"),
                log_mel=res["log_mel"],
                mfcc_vector=res["mfcc_vector"],
                snr_db=snr,
                clipping_ratio=clip_r,
                rms_energy=rms,
                source_type=row.get("source_type", "local_archive_source_unverified"),
            )
        )
    return samples
