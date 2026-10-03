"""
TinyCardio Real-Time Patient Simulation (CPU)
============================================
Simulates continuous telemetry streaming from a patient's single-lead ECG.
Evaluates window by window, computes risk scores, and triggers real-time alerts.
"""

import os
import sys
import time
import numpy as np

# Ensure parent path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ecg_loader import load_ecg
from inference_cpu import TinyCardioCPU


def simulate_patient_monitoring(record_path: str, max_seconds: int = 60, alert_threshold: float = 0.5):
    """Simulates real-time telemetry from an ECG record."""
    print("=" * 65)
    print(f"STARTING TINYCARDIO REAL-TIME MONITORING SIMULATION: {record_path}")
    print("=" * 65)

    rec = load_ecg(record_path, n_samples=None, physical=True)
    fs = rec.header.get('sampling_rate', 250.0)
    ch0 = rec.signals[0]

    # Convert to 250 Hz if needed
    if fs != 250.0:
        from scipy import signal
        num_target = int(len(ch0) * (250.0 / fs))
        sig = signal.resample(ch0, num_target)
    else:
        sig = np.array(ch0, dtype=np.float32)

    weights_p = '02_model_cpu/model_weights/tinycardio_cpu_weights.npz'
    model = TinyCardioCPU(weights_p, alert_threshold=alert_threshold)

    window_size = 250   # 1 second of ECG
    step_size = 125     # 0.5 second shift (50% overlap)
    total_samples = min(len(sig), int(max_seconds * 250))

    alerts_count = 0
    normal_count = 0

    print(f"{'Time (s)':<10} | {'Risk Score':<12} | {'Classification':<24} | {'Status':<10}")
    print("-" * 65)

    for start in range(0, total_samples - window_size + 1, step_size):
        w = sig[start:start + window_size]
        curr_time_sec = start / 250.0

        res = model.predict_window(w)
        score = res['risk_score']
        is_alert = res['is_alert']

        if is_alert:
            alerts_count += 1
            status_tag = "⚠️  [ALERT]"
        else:
            normal_count += 1
            status_tag = "✅  [OK]"

        # Print update every few steps or on alert
        if is_alert or (start % (step_size * 4) == 0):
            print(f"{curr_time_sec:8.1f} s | {score:10.4f}   | {res['classification']:<24} | {status_tag}")

    print("=" * 65)
    print(f"Simulation Finished.")
    print(f"Evaluated Windows: {alerts_count + normal_count} | Normal: {normal_count} | High Risk Alerts: {alerts_count}")
    if alerts_count > 0:
        print(">>> CRITICAL ALERT: Event detected during telemetry window! <<<")
    else:
        print(">>> Telemetry stable. Patient in normal sinus rhythm. <<<")
    print("=" * 65)


if __name__ == '__main__':
    default_rec = 'data/vfdb/418'
    if len(sys.argv) > 1:
        default_rec = sys.argv[1]
    simulate_patient_monitoring(default_rec, max_seconds=30)
