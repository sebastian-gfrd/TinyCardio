"""
TinyCardio Dataset Ingestion & Preprocessing Pipeline
====================================================
Loads ECG records from PhysioNet datasets (data/nsrdb, data/edb, data/vfdb, data/sddb).
Extracts labeled sliding windows (Class 0: Control/Normal vs Class 1: Arrhythmia/Ischemia/Arrest).
Performs patient-level train/validation/test split to prevent data leakage.
Re-samples NSRDB from 128 Hz to 250 Hz.
Normalizes and exports arrays for GPU/JAX acceleration.
"""

import os
import sys
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to use utils.ecg_loader if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ecg_loader import ECGRecord, load_ecg


def resample_signal(sig: List[float], orig_fs: float, target_fs: float = 250.0) -> np.ndarray:
    """Resamples a signal from orig_fs to target_fs using polyphase anti-aliasing filter."""
    if orig_fs == target_fs:
        return np.array(sig, dtype=np.float32)
    num_target_samples = int(len(sig) * (target_fs / orig_fs))
    return signal.resample(np.array(sig, dtype=np.float32), num_target_samples)


def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, fs: float = 250.0, order: int = 2) -> np.ndarray:
    """Applies a 2nd order Butterworth bandpass filter (0.5 - 40 Hz) matching TinyCardio spec."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Z-score normalizes a 1D window with small epsilon to prevent division by zero."""
    std = np.std(window)
    if std < 1e-4:
        return np.zeros_like(window, dtype=np.float32)
    return ((window - np.mean(window)) / std).astype(np.float32)


def extract_windows_from_record(record_path: str, label: int, window_size: int = 250, step_size: int = 125, max_windows: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an ECG record, extracts Channel 0, applies filtering, and slices into sliding windows.
    Returns:
        X: array of shape (N, window_size, 1)
        y: array of shape (N,)
    """
    needed_samples = (max_windows * step_size) + window_size + 1000
    rec = load_ecg(record_path, n_samples=needed_samples, physical=True)
    if not rec.signals or len(rec.signals[0]) < 250:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)

    ch0 = rec.signals[0]
    fs = rec.header.get('sampling_rate', 250.0)

    # Resample if needed (e.g. nsrdb from 128 Hz to 250 Hz)
    sig_250 = resample_signal(ch0, fs, target_fs=250.0)

    # Bandpass filter (0.5 - 40 Hz)
    sig_filtered = butter_bandpass_filter(sig_250, lowcut=0.5, highcut=40.0, fs=250.0)

    windows = []
    total_len = len(sig_filtered)

    for start in range(0, total_len - window_size + 1, step_size):
        w = sig_filtered[start:start + window_size]
        w_norm = normalize_window(w)
        windows.append(w_norm)
        if len(windows) >= max_windows:
            break

    if not windows:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.expand_dims(np.array(windows, dtype=np.float32), axis=-1)
    y = np.full((len(windows),), fill_value=label, dtype=np.float32)
    return X, y


def build_dataset(base_data_dir: str = 'data', window_size: int = 250, max_windows_per_rec: int = 300) -> Dict[str, np.ndarray]:
    """
    Constructs the train, validation, and test datasets with strict patient-level separation.
    Class 0 (Control): NSRDB
    Class 1 (Pathology/Alert): SDDB, VFDB, EDB
    """
    print("Building TinyCardio dataset with patient-level stratification...")

    # Define records per category
    # Control (nsrdb)
    nsrdb_dir = os.path.join(base_data_dir, 'nsrdb')
    nsrdb_recs = sorted([os.path.splitext(f)[0] for f in os.listdir(nsrdb_dir) if f.endswith('.hea')])
    
    # Pathological (vfdb, sddb, edb)
    vfdb_dir = os.path.join(base_data_dir, 'vfdb')
    vfdb_recs = sorted([os.path.splitext(f)[0] for f in os.listdir(vfdb_dir) if f.endswith('.hea') and not f.endswith('.hea-')])
    
    sddb_dir = os.path.join(base_data_dir, 'sddb')
    sddb_recs = sorted([os.path.splitext(f)[0] for f in os.listdir(sddb_dir) if f.endswith('.hea') and not f.endswith('.hea-')])

    edb_dir = os.path.join(base_data_dir, 'edb')
    edb_recs = sorted([os.path.splitext(f)[0] for f in os.listdir(edb_dir) if f.endswith('.hea')])

    # Patient-level split: 70% Train, 15% Val, 15% Test
    def split_recs(rec_list):
        n = len(rec_list)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return rec_list[:n_train], rec_list[n_train:n_train+n_val], rec_list[n_train+n_val:]

    nsr_tr, nsr_va, nsr_te = split_recs(nsrdb_recs)
    vf_tr, vf_va, vf_te = split_recs(vfdb_recs)
    sd_tr, sd_va, sd_te = split_recs(sddb_recs)
    ed_tr, ed_va, ed_te = split_recs(edb_recs[:30]) # Use a representative subset of EDB

    splits = {
        'train': {'c0': [(os.path.join(nsrdb_dir, r), 0) for r in nsr_tr],
                  'c1': [(os.path.join(vfdb_dir, r), 1) for r in vf_tr] +
                        [(os.path.join(sddb_dir, r), 1) for r in sd_tr] +
                        [(os.path.join(edb_dir, r), 1) for r in ed_tr]},
        'val':   {'c0': [(os.path.join(nsrdb_dir, r), 0) for r in nsr_va],
                  'c1': [(os.path.join(vfdb_dir, r), 1) for r in vf_va] +
                        [(os.path.join(sddb_dir, r), 1) for r in sd_va] +
                        [(os.path.join(edb_dir, r), 1) for r in ed_va]},
        'test':  {'c0': [(os.path.join(nsrdb_dir, r), 0) for r in nsr_te],
                  'c1': [(os.path.join(vfdb_dir, r), 1) for r in vf_te] +
                        [(os.path.join(sddb_dir, r), 1) for r in sd_te] +
                        [(os.path.join(edb_dir, r), 1) for r in ed_te]}
    }

    dataset = {}
    for split_name, categories in splits.items():
        X_list, y_list = [], []
        # Process Class 0
        for path, lbl in categories['c0']:
            X_w, y_w = extract_windows_from_record(path, lbl, window_size=window_size, max_windows=max_windows_per_rec)
            if len(X_w) > 0:
                X_list.append(X_w)
                y_list.append(y_w)
        # Process Class 1
        for path, lbl in categories['c1']:
            X_w, y_w = extract_windows_from_record(path, lbl, window_size=window_size, max_windows=max_windows_per_rec // 2)
            if len(X_w) > 0:
                X_list.append(X_w)
                y_list.append(y_w)

        if X_list:
            X_cat = np.concatenate(X_list, axis=0)
            y_cat = np.concatenate(y_list, axis=0)
            # Shuffle within split
            indices = np.random.permutation(len(y_cat))
            X_cat = X_cat[indices]
            y_cat = y_cat[indices]
            dataset[f'X_{split_name}'] = X_cat
            dataset[f'y_{split_name}'] = y_cat
            print(f"[{split_name.upper()}] Samples: {len(y_cat)} | Class 0: {np.sum(y_cat == 0)} | Class 1: {np.sum(y_cat == 1)}")

    return dataset


if __name__ == '__main__':
    np.random.seed(42)
    output_dir = '01_training_pipeline/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    ds = build_dataset('data', window_size=250, max_windows_per_rec=200)
    out_file = os.path.join(output_dir, 'dataset_250hz.npz')
    np.savez_compressed(out_file, **ds)
    print(f"Dataset successfully built and saved to {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
