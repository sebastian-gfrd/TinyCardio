"""
TinyCardio INT8 Edge Emulator in Python
=======================================
Simulates the exact arithmetic executed by tinyml_infer.c.
Reads the generated C header (include/model_weights.h) and verifies the
classification of normal vs pathological ECG windows.
"""

import os
import re
import numpy as np


def parse_c_header(header_path: str):
    """Parses arrays from model_weights.h."""
    with open(header_path, 'r') as f:
        text = f.read()

    def get_array_i8(name):
        m = re.search(rf'{name}\[\d+\]\s*=\s*\{{([^}}]+)\}};', text, re.DOTALL)
        if not m:
            raise ValueError(f"Array {name} not found in {header_path}")
        vals = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
        return np.array(vals, dtype=np.int8)

    def get_array_f32(name):
        m = re.search(rf'{name}\[\d+\]\s*=\s*\{{([^}}]+)\}};', text, re.DOTALL)
        if not m:
            raise ValueError(f"Array {name} not found in {header_path}")
        vals = [float(x.strip().rstrip('f')) for x in m.group(1).split(',') if x.strip()]
        return np.array(vals, dtype=np.float32)

    def get_scale(name):
        m = re.search(rf'#define\s+{name}\s+([0-9.ef+-]+)', text)
        return float(m.group(1).rstrip('f')) if m else 1.0

    return {
        'conv1_w': get_array_i8('g_tinycardio_conv1_w').reshape(5, 1, 8),
        'conv1_b': get_array_f32('g_tinycardio_conv1_b'),
        'conv2_w': get_array_i8('g_tinycardio_conv2_w').reshape(3, 8, 16),
        'conv2_b': get_array_f32('g_tinycardio_conv2_b'),
        'dense_w': get_array_i8('g_tinycardio_dense_w').reshape(16, 1),
        'dense_b': get_array_f32('g_tinycardio_dense_b'),
        's_conv1': get_scale('TINYCARDIO_SCALE_CONV1'),
        's_conv2': get_scale('TINYCARDIO_SCALE_CONV2'),
        's_dense': get_scale('TINYCARDIO_SCALE_DENSE')
    }


def emulate_c_infer(header_params, window_250: np.ndarray) -> float:
    """Emulates tinyml_predict_float from tinyml_infer.c."""
    x = window_250.copy()
    std = np.std(x)
    x = (x - np.mean(x)) / (std if std > 1e-4 else 1.0)

    # Layer 1: Conv1D (k=5, in=1, out=8, stride=2, SAME)
    out_len1 = 125
    h1 = np.zeros((out_len1, 8), dtype=np.float32)
    w1_f = header_params['conv1_w'] * header_params['s_conv1']
    b1_f = header_params['conv1_b']

    for i in range(out_len1):
        center = i * 2
        start = center - 2
        for oc in range(8):
            s = b1_f[oc]
            for k in range(5):
                idx = start + k
                val = x[idx] if (0 <= idx < 250) else 0.0
                s += val * w1_f[k, 0, oc]
            h1[i, oc] = max(0.0, s)

    # Layer 2: Conv1D (k=3, in=8, out=16, stride=2, SAME)
    out_len2 = 63
    h2 = np.zeros((out_len2, 16), dtype=np.float32)
    w2_f = header_params['conv2_w'] * header_params['s_conv2']
    b2_f = header_params['conv2_b']

    for i in range(out_len2):
        center = i * 2
        start = center - 1
        for oc in range(16):
            s = b2_f[oc]
            for k in range(3):
                idx = start + k
                if 0 <= idx < out_len1:
                    for ic in range(8):
                        s += h1[idx, ic] * w2_f[k, ic, oc]
            h2[i, oc] = max(0.0, s)

    # Layer 3: GAP
    gap = np.mean(h2, axis=0)

    # Layer 4: Dense + Sigmoid
    w_d_f = header_params['dense_w'].ravel() * header_params['s_dense']
    b_d_f = header_params['dense_b'][0]

    logit = float(np.dot(gap, w_d_f) + b_d_f)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
    return prob


if __name__ == '__main__':
    header = '03_edge_serialized/include/model_weights.h'
    params = parse_c_header(header)
    print("=" * 60)
    print("TINYCARDIO INT8 SERIALIZED MODEL C-EMULATOR")
    print("=" * 60)
    print(f"Header: {header} loaded successfully.")
    print(f"Quantized Scales: Conv1={params['s_conv1']:.6f}, Conv2={params['s_conv2']:.6f}, Dense={params['s_dense']:.6f}")

    # Test with real patient data from dataset_250hz.npz
    ds = np.load('01_training_pipeline/processed_data/dataset_250hz.npz')
    X_test = ds['X_test']
    y_test = ds['y_test']

    # Select 5 normal and 5 alert samples
    norm_idx = np.where(y_test == 0)[0][:5]
    alert_idx = np.where(y_test == 1)[0][:5]

    print("\n[VERIFICATION ON REAL UNSEEN PATIENT TEST WINDOWS]")
    print("Normal Control Windows (Expected: LOW RISK < 0.5):")
    for i, idx in enumerate(norm_idx):
        score = emulate_c_infer(params, X_test[idx].ravel())
        print(f"  Sample {i+1}: Risk = {score*100:5.2f}% | Class = {'NORMAL' if score < 0.5 else 'ALERT'}")

    print("\nPathological Alert Windows (Expected: HIGH RISK >= 0.5):")
    for i, idx in enumerate(alert_idx):
        score = emulate_c_infer(params, X_test[idx].ravel())
        print(f"  Sample {i+1}: Risk = {score*100:5.2f}% | Class = {'ALERT' if score >= 0.5 else 'NORMAL'}")

    print("=" * 60)
