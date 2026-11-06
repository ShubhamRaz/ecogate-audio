"""
Dataset auditing, SHA-256 and decoded waveform hashing, duplicate grouping,
ESC-50 provenance verification, and group-safe splitting for EcoGate-Audio.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


KNOWN_SUBDIRS = (
    "common_waxbill",
    "red_billed_quelea",
    "village_weaver",
    "yellow_fronted_canary",
    "other_birds",
    "environmental_noise",
)

def discover_files_recursive(raw_dir: Path, suffixes: Tuple[str, ...] = (".wav",)) -> List[Path]:
    raw_dir = Path(raw_dir)
    found_set = {}
    if raw_dir.exists():
        for r, _, fs in os.walk(raw_dir):
            for f in fs:
                if f.lower().endswith(suffixes):
                    p = Path(r) / f
                    found_set[str(p)] = p
        for sub in KNOWN_SUBDIRS:
            sd = raw_dir / sub
            if sd.exists() and sd.is_dir():
                for f in os.listdir(sd):
                    if f.lower().endswith(suffixes):
                        p = sd / f
                        found_set[str(p)] = p
    return [found_set[k] for k in sorted(found_set.keys())]

ESC50_FILENAME_PATTERN = re.compile(r"^[1-5]-\d{3,6}-[A-Z]-\d{1,2}\.wav$", re.IGNORECASE)

REPORTED_PRELIMINARY_CLAIMS = {
    "wav_files_in_released_archive": 8512,
    "jpeg_files": 763,
    "audio_annotation_records": 10723,
    "missing_referenced_audio_paths": 2211,
    "cross_split_duplicate_groups": 14,
    "file_entries_in_duplicate_groups": 29,
    "image_bounding_boxes": 3875,
    "image_annotation_classes": ["Bird"],
    "noise_filenames_matching_esc50": 2000,
    "note": (
        "Preliminary audit observations from earlier archive inspection. "
        "Treated as historical baselines to verify independently against local raw data."
    ),
}


@dataclass
class AudioRecordManifest:
    file_path: str
    filename: str
    label: str
    sha256: str
    waveform_sha256: str
    duration_sec: float
    sample_rate: int
    channels: int
    rms_energy: float
    peak_amplitude: float
    clipping_ratio: float
    estimated_snr_db: float
    is_corrupt: bool
    source_id: str
    recording_session_id: str
    duplicate_group_id: str
    provenance_info: str
    matches_esc50_pattern: bool
    inclusion_status: str
    exclusion_reason: str
    split: str = "unassigned"


def compute_file_sha256(path: Path) -> str:
    """Compute exact SHA-256 hash of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def read_wav_safe(path: Path) -> Tuple[Optional[np.ndarray], int, int, Optional[str]]:
    """
    Safely read a WAV file without modifying it.
    Returns (waveform_float32_mono, sample_rate, channels, error_reason).
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            if nframes <= 0 or sr <= 0:
                return None, sr, channels, "zero_frames_or_invalid_sr"
            raw_bytes = wf.readframes(nframes)
        if sampwidth == 2:
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return None, sr, channels, f"unsupported_sampwidth_{sampwidth}"
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if samples.size == 0 or not np.all(np.isfinite(samples)):
            return None, sr, channels, "non_finite_or_empty_samples"
        return samples, sr, channels, None
    except Exception as exc:
        return None, 0, 0, f"corrupt_wav: {exc}"


def compute_waveform_sha256(samples: np.ndarray) -> str:
    """Compute SHA-256 of decoded quantized float32 waveform to catch header-only differences."""
    quantized = np.round(samples.astype(np.float32), 5).tobytes()
    return hashlib.sha256(quantized).hexdigest()


def analyze_waveform_quality(samples: np.ndarray, clip_thresh: float = 0.999) -> Dict[str, float]:
    """Compute RMS energy, peak amplitude, clipping fraction, and frame-based SNR estimate."""
    if samples.size == 0:
        return {"rms_energy": 0.0, "peak_amplitude": 0.0, "clipping_ratio": 0.0, "estimated_snr_db": -20.0}
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    peak = float(np.max(np.abs(samples)))
    clip_ratio = float(np.mean(np.abs(samples) >= clip_thresh))
    # Frame-level signal vs noise floor percentile estimation
    frame_len = min(1024, max(64, samples.size // 10))
    n_frames = max(1, samples.size // frame_len)
    framed = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_powers = np.mean(np.square(framed), axis=1) + 1e-12
    signal_p = float(np.percentile(frame_powers, 90))
    noise_p = float(np.percentile(frame_powers, 15))
    snr_db = float(10.0 * np.log10(max(signal_p, 1e-12) / max(noise_p, 1e-12)))
    return {
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "clipping_ratio": round(clip_ratio, 6),
        "estimated_snr_db": round(snr_db, 3),
    }


def infer_provenance_and_source(filepath: Path, label: str) -> Tuple[str, str, str, bool]:
    """
    Inspect filename patterns conservatively. Never fabricate recording sessions or GPS locations.
    """
    fname = filepath.name
    is_esc50 = bool(ESC50_FILENAME_PATTERN.match(fname))
    if is_esc50:
        # ESC-50 format: {fold}-{clip_id}-{take}-{target}.wav
        parts = fname.split("-")
        clip_id = parts[1] if len(parts) > 1 else "unknown"
        return (
            f"esc50_source_{clip_id}",
            f"esc50_clip_{clip_id}",
            "ESC-50_catalogue_filename_match",
            True,
        )
    # Check if filename Explicitly encodes a session prefix like siteA_sess01_clip03.wav
    m = re.match(r"^([a-zA-Z0-9]+_sess\d+)_", fname)
    if m:
        sess = m.group(1)
        return sess, sess, "explicit_session_prefix_in_filename", False
    return "unknown", "unknown", "field_or_unverified_archive_recording", False


def build_and_audit_manifest(
    raw_dir: Path,
    annotations_csv: Optional[Path] = None,
) -> Tuple[List[AudioRecordManifest], Dict[str, Any]]:
    """
    Scan raw directory, inspect optional annotation CSV for missing paths,
    compute SHA-256 and decoded waveform hashes, assign duplicate groups,
    and return structured manifest records + audit summary dictionary.
    """
    raw_dir = Path(raw_dir)
    wav_files = discover_files_recursive(raw_dir, (".wav",))
    jpg_files = discover_files_recursive(raw_dir, (".jpg", ".jpeg"))

    missing_annotation_paths: List[str] = []
    annotation_records_count = 0
    if annotations_csv and annotations_csv.exists():
        with open(annotations_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotation_records_count += 1
                rel_p = row.get("file_path", "") or row.get("filename", "")
                if rel_p and not (raw_dir / rel_p).exists():
                    missing_annotation_paths.append(rel_p)

    records: List[AudioRecordManifest] = []
    sha_to_indices: Dict[str, List[int]] = {}
    wave_sha_to_indices: Dict[str, List[int]] = {}

    for wav_path in wav_files:
        rel_path = str(wav_path.relative_to(raw_dir))
        parent_label = wav_path.parent.name.lower().replace(" ", "_").replace("-", "_")
        if parent_label in ("raw", "audio", ""):
            parent_label = "unknown"

        file_sha = compute_file_sha256(wav_path)
        samples, sr, channels, err = read_wav_safe(wav_path)
        source_id, sess_id, prov, is_esc50 = infer_provenance_and_source(wav_path, parent_label)

        if err is not None or samples is None:
            rec = AudioRecordManifest(
                file_path=rel_path,
                filename=wav_path.name,
                label=parent_label,
                sha256=file_sha,
                waveform_sha256="corrupt",
                duration_sec=0.0,
                sample_rate=sr,
                channels=channels,
                rms_energy=0.0,
                peak_amplitude=0.0,
                clipping_ratio=0.0,
                estimated_snr_db=-20.0,
                is_corrupt=True,
                source_id=source_id,
                recording_session_id=sess_id,
                duplicate_group_id=f"corrupt_{ len(records) }",
                provenance_info=prov,
                matches_esc50_pattern=is_esc50,
                inclusion_status="excluded",
                exclusion_reason=err,
            )
            records.append(rec)
            continue

        wave_sha = compute_waveform_sha256(samples)
        qual = analyze_waveform_quality(samples)
        duration = round(samples.size / float(sr), 4)

        inclusion = "included"
        excl_reason = "none"
        if qual["rms_energy"] < 0.0002:
            inclusion = "flagged_low_quality"
            excl_reason = "near_silent_recording"
        elif qual["clipping_ratio"] > 0.15:
            inclusion = "flagged_low_quality"
            excl_reason = "severe_audio_clipping"

        idx = len(records)
        sha_to_indices.setdefault(file_sha, []).append(idx)
        wave_sha_to_indices.setdefault(wave_sha, []).append(idx)

        rec = AudioRecordManifest(
            file_path=rel_path,
            filename=wav_path.name,
            label=parent_label,
            sha256=file_sha,
            waveform_sha256=wave_sha,
            duration_sec=duration,
            sample_rate=sr,
            channels=channels,
            rms_energy=qual["rms_energy"],
            peak_amplitude=qual["peak_amplitude"],
            clipping_ratio=qual["clipping_ratio"],
            estimated_snr_db=qual["estimated_snr_db"],
            is_corrupt=False,
            source_id=source_id,
            recording_session_id=sess_id,
            duplicate_group_id="",
            provenance_info=prov,
            matches_esc50_pattern=is_esc50,
            inclusion_status=inclusion,
            exclusion_reason=excl_reason,
        )
        records.append(rec)

    # Assign duplicate_group_id using decoded waveform hash (which subsumes exact SHA-256)
    duplicate_groups_count = 0
    files_in_duplicate_groups = 0
    for g_idx, (w_sha, member_indices) in enumerate(wave_sha_to_indices.items()):
        if len(member_indices) > 1:
            duplicate_groups_count += 1
            files_in_duplicate_groups += len(member_indices)
            gid = f"dupgroup_{g_idx:04d}"
        else:
            gid = f"unique_{member_indices[0]:05d}"
        for m_i in member_indices:
            records[m_i].duplicate_group_id = gid

    # Summarize distributions
    class_counts: Dict[str, int] = {}
    sr_counts: Dict[str, int] = {}
    ch_counts: Dict[str, int] = {}
    durations: List[float] = []
    corrupt_count = 0
    esc50_count = 0

    for r in records:
        class_counts[r.label] = class_counts.get(r.label, 0) + 1
        if r.is_corrupt:
            corrupt_count += 1
        else:
            sr_counts[str(r.sample_rate)] = sr_counts.get(str(r.sample_rate), 0) + 1
            ch_counts[str(r.channels)] = ch_counts.get(str(r.channels), 0) + 1
            durations.append(r.duration_sec)
        if r.matches_esc50_pattern:
            esc50_count += 1

    audit_summary = {
        "local_execution_findings": {
            "total_wav_files_scanned": len(wav_files),
            "total_image_files_scanned": len(jpg_files),
            "annotation_records_inspected": annotation_records_count,
            "missing_referenced_audio_paths_detected": len(missing_annotation_paths),
            "missing_paths_list": missing_annotation_paths[:50],
            "corrupt_wav_files": corrupt_count,
            "valid_wav_files": len(wav_files) - corrupt_count,
            "duplicate_waveform_groups": duplicate_groups_count,
            "files_in_duplicate_groups": files_in_duplicate_groups,
            "esc50_filename_matches": esc50_count,
            "class_distribution": class_counts,
            "sample_rate_distribution": sr_counts,
            "channel_distribution": ch_counts,
            "duration_summary_sec": {
                "min": round(float(np.min(durations)), 4) if durations else 0.0,
                "max": round(float(np.max(durations)), 4) if durations else 0.0,
                "mean": round(float(np.mean(durations)), 4) if durations else 0.0,
                "median": round(float(np.median(durations)), 4) if durations else 0.0,
            },
        },
        "preliminary_reported_archive_claims": REPORTED_PRELIMINARY_CLAIMS,
        "synchronization_policy": (
            "Audio-first strictly enforced. Images are never paired with audio clips "
            "because no verified temporal synchronization protocol exists in the archive."
        ),
    }
    return records, audit_summary


def create_group_safe_splits(
    records: List[AudioRecordManifest],
    ratios: Dict[str, float],
    seed: int = 42,
) -> List[AudioRecordManifest]:
    """
    Partition dataset into train, val, calibration, and test splits using group-based
    stratified assignment so that duplicate groups and known recording sessions never
    cross split boundaries.
    """
    rng = np.random.RandomState(seed)
    split_names = ["train", "val", "calibration", "test"]
    target_fracs = [ratios[k] for k in split_names]

    # Build composite group key: if duplicate_group_id is shared OR recording_session_id is known
    # Union-Find across records sharing duplicate_group_id or valid recording_session_id
    n = len(records)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    dup_map: Dict[str, int] = {}
    sess_map: Dict[str, int] = {}
    for idx, rec in enumerate(records):
        if rec.duplicate_group_id:
            if rec.duplicate_group_id in dup_map:
                union(idx, dup_map[rec.duplicate_group_id])
            else:
                dup_map[rec.duplicate_group_id] = idx
        if rec.recording_session_id and rec.recording_session_id != "unknown":
            if rec.recording_session_id in sess_map:
                union(idx, sess_map[rec.recording_session_id])
            else:
                sess_map[rec.recording_session_id] = idx

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    # Group clusters by primary label of their first record for class-balanced group assignment
    label_to_clusters: Dict[str, List[List[int]]] = {}
    for _, member_idxs in sorted(clusters.items()):
        lbl = records[member_idxs[0]].label
        label_to_clusters.setdefault(lbl, []).append(member_idxs)

    for lbl, clist in label_to_clusters.items():
        # Shuffle clusters deterministically
        order = rng.permutation(len(clist))
        shuffled = [clist[i] for i in order]
        # Greedy bucket fill to match target_fracs per class while keeping every cluster intact
        counts = np.zeros(len(split_names), dtype=np.float64)
        total_lbl_items = sum(len(c) for c in shuffled)
        desired = np.array(target_fracs) * max(1, total_lbl_items)

        for cluster in shuffled:
            # Ensure every split gets at least one cluster if len(shuffled) >= 4
            deficits = desired - counts
            chosen_split_idx = int(np.argmax(deficits))
            chosen_split = split_names[chosen_split_idx]
            for m_i in cluster:
                if records[m_i].is_corrupt:
                    records[m_i].split = "excluded"
                else:
                    records[m_i].split = chosen_split
            counts[chosen_split_idx] += len(cluster)

    return records


def verify_split_integrity(records: List[AudioRecordManifest]) -> Dict[str, Any]:
    """
    Verify zero duplicate-group leakage and zero known session leakage across splits.
    """
    valid_records = [r for r in records if not r.is_corrupt and r.split in ("train", "val", "calibration", "test")]
    dup_to_splits: Dict[str, set] = {}
    sha_to_splits: Dict[str, set] = {}
    sess_to_splits: Dict[str, set] = {}

    for r in valid_records:
        dup_to_splits.setdefault(r.duplicate_group_id, set()).add(r.split)
        sha_to_splits.setdefault(r.waveform_sha256, set()).add(r.split)
        if r.recording_session_id != "unknown":
            sess_to_splits.setdefault(r.recording_session_id, set()).add(r.split)

    leaked_dups = {k: list(v) for k, v in dup_to_splits.items() if len(v) > 1}
    leaked_shas = {k: list(v) for k, v in sha_to_splits.items() if len(v) > 1}
    leaked_sess = {k: list(v) for k, v in sess_to_splits.items() if len(v) > 1}

    split_counts: Dict[str, int] = {}
    for r in valid_records:
        split_counts[r.split] = split_counts.get(r.split, 0) + 1

    passed = (len(leaked_dups) == 0) and (len(leaked_shas) == 0) and (len(leaked_sess) == 0)
    return {
        "passed": passed,
        "leaked_duplicate_groups": leaked_dups,
        "leaked_waveform_shas": leaked_shas,
        "leaked_recording_sessions": leaked_sess,
        "split_counts": split_counts,
    }


def save_manifest_csv_and_json(
    records: List[AudioRecordManifest],
    csv_path: Path,
    json_path: Path,
) -> str:
    """Save manifest records to CSV and JSON and return SHA-256 checksum of the CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = list(AudioRecordManifest.__annotations__.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return compute_file_sha256(csv_path)
