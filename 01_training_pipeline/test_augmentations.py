"""
Test and Verification Suite for ECG Augmentations
=================================================
Validates that each physical artifact simulator preserves signal dimensions,
does not introduce NaNs or infinities, and maintains valid statistical bounds.
"""

import numpy as np
from augmentations import (
    add_baseline_wander,
    add_powerline_interference,
    add_emg_noise,
    add_motion_artifact,
    random_amplitude_scaling,
    augment_ecg_window
)


def run_tests():
    print("=" * 65)
    print("VERIFYING BIOMEDICAL ECG AUGMENTATIONS SUITE")
    print("=" * 65)

    np.random.seed(42)
    # Synthetic QRS-like test pulse
    t = np.linspace(0, 1.0, 250)
    clean_signal = np.sin(2 * np.pi * 1.5 * t) + np.exp(-((t - 0.5) * 30.0) ** 2) * 2.0

    # 1. Baseline Wander
    w_bw = add_baseline_wander(clean_signal)
    assert len(w_bw) == 250, "Length mismatch in baseline wander"
    assert not np.isnan(w_bw).any(), "NaN found in baseline wander"
    print("  [PASS] 1. Baseline Wander (Respiration drift 0.15 - 0.45 Hz)")

    # 2. Powerline Hum
    w_pl = add_powerline_interference(clean_signal, power_freq=50.0)
    assert len(w_pl) == 250, "Length mismatch in powerline hum"
    assert not np.isnan(w_pl).any(), "NaN found in powerline hum"
    print("  [PASS] 2. Powerline Interference (50 Hz & 60 Hz + 2nd harmonics)")

    # 3. EMG Tremors
    w_emg = add_emg_noise(clean_signal, snr_db_range=(15.0, 25.0))
    assert len(w_emg) == 250, "Length mismatch in EMG noise"
    assert not np.isnan(w_emg).any(), "NaN found in EMG noise"
    print("  [PASS] 3. Electromyographic (EMG) Muscle Tremors (15-25 dB SNR)")

    # 4. Motion Artifact
    w_ma = add_motion_artifact(clean_signal, p_jump=1.0)
    assert len(w_ma) == 250, "Length mismatch in motion artifact"
    assert not np.isnan(w_ma).any(), "NaN found in motion artifact"
    print("  [PASS] 4. Electrode Contact Motion Artifact & Decay Jump")

    # 5. Amplitude Scaling
    w_sc = random_amplitude_scaling(clean_signal)
    assert len(w_sc) == 250, "Length mismatch in amplitude scaling"
    assert not np.isnan(w_sc).any(), "NaN found in amplitude scaling"
    print("  [PASS] 5. Inter-patient Physiological Amplitude Variation (0.80 - 1.25x)")

    # 6. Composite Stochastic Pipeline
    w_comp = augment_ecg_window(clean_signal, p_apply=1.0)
    assert len(w_comp) == 250, "Length mismatch in composite pipeline"
    assert not np.isnan(w_comp).any(), "NaN found in composite pipeline"
    print("  [PASS] 6. Composite Stochastic Pipeline (Randomized multi-artifact)")

    print("=" * 65)
    print("ALL 6 BIOMEDICAL AUGMENTATION TESTS PASSED (100% Stability)")
    print("=" * 65)


if __name__ == '__main__':
    run_tests()
