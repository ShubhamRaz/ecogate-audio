"""
Reproducible audio preprocessing pipeline for EcoGate-Audio.
Implements mono conversion, resampling, RMS/peak normalization, fixed-window padding/trimming,
Log-Mel spectrogram extraction, MFCC extraction, and train-only augmentations.
Compatible with both NumPy/SciPy environments and PyTorch/Torchaudio when installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fftpack import dct

from src.audit.auditor import read_wav_safe, analyze_waveform_quality


@dataclass
class AudioPreprocessConfig:
    target_sr: int = 32000
    duration_sec: float = 3.0
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 64
    n_mfcc: int = 20
    fmin: float = 50.0
    fmax: float = 14000.0
    target_rms_db: float = -20.0
    log_offset: float = 1e-6


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def create_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 64,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Create a standard triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    if fmax is None:
        fmax = float(sr) / 2.0
    fmax = min(fmax, float(sr) / 2.0)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        if f_center == f_left:
            f_center += 1
        if f_right == f_center:
            f_right += 1
        for k in range(f_left, min(f_center, n_freqs)):
            fb[m - 1, k] = (k - f_left) / float(max(1, f_center - f_left))
        for k in range(f_center, min(f_right, n_freqs)):
            fb[m - 1, k] = (f_right - k) / float(max(1, f_right - f_center))
    return fb


class AudioProcessor:
    """End-to-end deterministic audio preprocessor and acoustic feature extractor."""

    def __init__(self, cfg: Optional[AudioPreprocessConfig] = None) -> None:
        self.cfg = cfg or AudioPreprocessConfig()
        self.target_length = int(round(self.cfg.target_sr * self.cfg.duration_sec))
        self.mel_fb = create_mel_filterbank(
            sr=self.cfg.target_sr,
            n_fft=self.cfg.n_fft,
            n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
        )

    def resample_waveform(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample 1D mono waveform to target_sr using polyphase filtering."""
        if orig_sr == self.cfg.target_sr:
            return waveform.astype(np.float32)
        if orig_sr <= 0 or waveform.size == 0:
            return np.zeros(self.target_length, dtype=np.float32)
        gcd = int(np.gcd(orig_sr, self.cfg.target_sr))
        up = self.cfg.target_sr // gcd
        down = orig_sr // gcd
        resampled = signal.resample_poly(waveform, up, down)
        return resampled.astype(np.float32)

    def normalize_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Apply documented RMS normalization with anti-clipping ceiling."""
        rms = float(np.sqrt(np.mean(np.square(waveform)) + 1e-12))
        if rms < 1e-6:
            return waveform.astype(np.float32)
        target_rms = 10.0 ** (self.cfg.target_rms_db / 20.0)
        scaled = waveform * (target_rms / rms)
        peak = float(np.max(np.abs(scaled)))
        if peak > 0.99:
            scaled = scaled * (0.99 / peak)
        return scaled.astype(np.float32)

    def pad_or_trim(self, waveform: np.ndarray) -> np.ndarray:
        """Create fixed-size window by deterministic center-cropping or zero-padding."""
        n = waveform.size
        if n == self.target_length:
            return waveform
        if n > self.target_length:
            start = (n - self.target_length) // 2
            return waveform[start : start + self.target_length]
        pad_total = self.target_length - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(waveform, (pad_left, pad_right), mode="constant").astype(np.float32)

    def extract_log_mel(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute Log-Mel spectrogram of shape (n_mels, n_frames).
        """
        w = self.pad_or_trim(waveform)
        _, _, zxx = signal.stft(
            w,
            fs=self.cfg.target_sr,
            window="hann",
            nperseg=self.cfg.n_fft,
            noverlap=self.cfg.n_fft - self.cfg.hop_length,
            nfft=self.cfg.n_fft,
            boundary="zeros",
            padded=True,
        )
        power_spec = np.abs(zxx) ** 2
        mel_spec = np.dot(self.mel_fb, power_spec)
        log_mel = np.log(mel_spec + self.cfg.log_offset).astype(np.float32)
        return log_mel

    def extract_mfcc(self, log_mel: np.ndarray, include_deltas: bool = True) -> np.ndarray:
        """
        Extract MFCCs (and optional first/second order deltas) from a Log-Mel spectrogram,
        returning summary statistics vector (mean + std across time frames) for classical ML.
        """
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[: self.cfg.n_mfcc]
        feats = [ np.mean(mfcc, axis=1), np.std(mfcc, axis=1) ]
        if include_deltas:
            delta1 = np.gradient(mfcc, axis=1)
            delta2 = np.gradient(delta1, axis=1)
            feats.extend([
                np.mean(delta1, axis=1),
                np.std(delta1, axis=1),
                np.mean(delta2, axis=1),
                np.std(delta2, axis=1),
            ])
        return np.concatenate(feats, axis=0).astype(np.float32)

    def process_file(self, wav_path: Path) -> Dict[str, Any]:
        """Load, validate, preprocess, and extract Log-Mel, MFCC, and quality metrics."""
        samples, sr, channels, err = read_wav_safe(wav_path)
        if err is not None or samples is None:
            return {
                "valid": False,
                "error": err,
                "log_mel": None,
                "mfcc_vector": None,
                "quality": {"rms_energy": 0.0, "peak_amplitude": 0.0, "clipping_ratio": 1.0, "estimated_snr_db": -20.0},
            }
        qual = analyze_waveform_quality(samples)
        resampled = self.resample_waveform(samples, sr)
        normed = self.normalize_waveform(resampled)
        windowed = self.pad_or_trim(normed)
        log_mel = self.extract_log_mel(windowed)
        mfcc_vec = self.extract_mfcc(log_mel, include_deltas=True)
        return {
            "valid": True,
            "error": None,
            "waveform": windowed,
            "log_mel": log_mel,
            "mfcc_vector": mfcc_vec,
            "quality": qual,
        }


class TrainOnlyAugmentor:
    """
    Configurable train-only waveform and spectrogram augmentations.
    Must ONLY be applied to training partition examples after group splitting.
    """

    def __init__(
        self,
        time_mask_max: int = 20,
        freq_mask_max: int = 8,
        gain_db_range: Tuple[float, float] = (-6.0, 6.0),
        time_shift_max_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.gain_db_range = gain_db_range
        self.time_shift_max_frac = time_shift_max_frac
        self.rng = np.random.RandomState(seed)

    def augment_waveform(
        self,
        waveform: np.ndarray,
        noise_waveform: Optional[np.ndarray] = None,
        snr_db: float = 12.0,
    ) -> np.ndarray:
        """Apply gain variation, circular time shift, and optional background noise mixing."""
        out = waveform.copy()
        gain_db = self.rng.uniform(self.gain_db_range[0], self.gain_db_range[1])
        out = out * (10.0 ** (gain_db / 20.0))

        max_shift = int(out.size * self.time_shift_max_frac)
        if max_shift > 0:
            shift = self.rng.randint(-max_shift, max_shift + 1)
            out = np.roll(out, shift)

        if noise_waveform is not None and noise_waveform.size == out.size:
            sig_p = np.mean(out ** 2) + 1e-12
            noise_p = np.mean(noise_waveform ** 2) + 1e-12
            scale = np.sqrt(sig_p / (noise_p * (10.0 ** (snr_db / 10.0))))
            out = out + scale * noise_waveform

        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def spec_augment(self, log_mel: np.ndarray) -> np.ndarray:
        """Apply SpecAugment frequency and time masking on a copy of log_mel."""
        spec = log_mel.copy()
        n_mels, n_frames = spec.shape
        fill_val = float(np.mean(spec))

        if self.freq_mask_max > 0 and n_mels > self.freq_mask_max:
            f = self.rng.randint(1, self.freq_mask_max + 1)
            f0 = self.rng.randint(0, n_mels - f)
            spec[f0 : f0 + f, :] = fill_val

        if self.time_mask_max > 0 and n_frames > self.time_mask_max:
            t = self.rng.randint(1, self.time_mask_max + 1)
            t0 = self.rng.randint(0, n_frames - t)
            spec[:, t0 : t0 + t] = fill_val

        return spec
