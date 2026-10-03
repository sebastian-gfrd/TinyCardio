"""
Biomedical ECG Signal Augmentation & Rural Noise Simulation
===========================================================
Simulates real-world physical artifacts encountered by low-cost single-lead
sensors (e.g. AD8232, dry electrodes, ungrounded rural clinic power grids):
  1. Respiration Baseline Wander (0.1 - 0.5 Hz sinusoidal fluctuation)
  2. Powerline Grid Interference (50 Hz / 60 Hz electromagnetic hum)
  3. Electromyographic (EMG) Muscle Noise (high-frequency tremors)
  4. Electrode Motion Artifacts & Contact Impedance Jumps
  5. Inter-patient Amplitude Variations & Random Scaling
"""

import numpy as np
from typing import Tuple, Optional


def add_baseline_wander(sig: np.ndarray, fs: float = 250.0, amp_range: Tuple[float, float] = (0.1, 0.4)) -> np.ndarray:
    """Simulates low-frequency baseline drift from patient respiration and chest movement."""
    n = len(sig)
    freq = np.random.uniform(0.15, 0.45)
    phase = np.random.uniform(0, 2 * np.pi)
    amp = np.random.uniform(*amp_range)
    t = np.arange(n) / fs
    wander = amp * np.sin(2 * np.pi * freq * t + phase)
    return sig + wander


def add_powerline_interference(sig: np.ndarray, fs: float = 250.0, power_freq: float = 50.0, amp_range: Tuple[float, float] = (0.02, 0.08)) -> np.ndarray:
    """Simulates 50/60 Hz AC grid electromagnetic hum common with ungrounded generators in rural clinics."""
    n = len(sig)
    phase = np.random.uniform(0, 2 * np.pi)
    amp = np.random.uniform(*amp_range)
    t = np.arange(n) / fs
    hum = amp * np.sin(2 * np.pi * power_freq * t + phase)
    # Add small 2nd harmonic (100 Hz / 120 Hz)
    hum += (amp * 0.25) * np.sin(2 * np.pi * (2 * power_freq) * t + phase)
    return sig + hum


def add_emg_noise(sig: np.ndarray, snr_db_range: Tuple[float, float] = (15.0, 30.0)) -> np.ndarray:
    """Simulates high-frequency muscle tremor noise (EMG) from patient shivering or movement."""
    sig_power = np.mean(sig ** 2)
    if sig_power < 1e-6:
        return sig
    snr_db = np.random.uniform(*snr_db_range)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=len(sig))
    return sig + noise


def add_motion_artifact(sig: np.ndarray, p_jump: float = 0.25, max_jump_amp: float = 0.5) -> np.ndarray:
    """Simulates sudden electrode contact dislocation or skin stretch impedance jump."""
    if np.random.rand() > p_jump:
        return sig
    n = len(sig)
    jump_point = np.random.randint(n // 4, 3 * n // 4)
    jump_amp = np.random.uniform(-max_jump_amp, max_jump_amp)
    sig_modified = sig.copy()
    # Apply exponential decay to simulate electrode recovery
    decay = np.exp(-np.linspace(0, 3, n - jump_point))
    sig_modified[jump_point:] += jump_amp * decay
    return sig_modified


def random_amplitude_scaling(sig: np.ndarray, scale_range: Tuple[float, float] = (0.80, 1.25)) -> np.ndarray:
    """Simulates physiological impedance variation between different patients."""
    scale = np.random.uniform(*scale_range)
    return sig * scale


def augment_ecg_window(window: np.ndarray, fs: float = 250.0, p_apply: float = 0.70) -> np.ndarray:
    """
    Applies a realistic stochastic combination of physical artifacts to a 1D ECG window.
    Designed exclusively for training split augmentation to preserve clean clinical evaluation.
    """
    if np.random.rand() > p_apply:
        return window

    w = window.copy()

    # 1. Respiration baseline drift (60% chance)
    if np.random.rand() < 0.60:
        w = add_baseline_wander(w, fs=fs)

    # 2. Powerline interference (50% chance, alternating 50 Hz and 60 Hz)
    if np.random.rand() < 0.50:
        freq = 50.0 if np.random.rand() < 0.5 else 60.0
        w = add_powerline_interference(w, fs=fs, power_freq=freq)

    # 3. EMG muscle tremors (50% chance)
    if np.random.rand() < 0.50:
        w = add_emg_noise(w)

    # 4. Electrode motion artifact (30% chance)
    if np.random.rand() < 0.30:
        w = add_motion_artifact(w)

    # 5. Amplitude scaling (always variable)
    w = random_amplitude_scaling(w)

    return w
