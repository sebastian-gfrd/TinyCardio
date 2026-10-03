"""
TinyCardio Terminal Live ECG Waveform & Risk Monitor
====================================================
Real-time ASCII/ANSI oscilloscope monitor for frontline Community Health Workers (CHWs).
Renders streaming single-lead ECG waveforms and dynamic risk indicators directly
in standard Unix/Linux terminal sessions without requiring graphical desktop environments.
"""

import os
import sys
import time
import math
import numpy as np
from typing import List

# Ensure parent and current directory in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
CPU_DIR = os.path.dirname(__file__)
if CPU_DIR not in sys.path:
    sys.path.insert(0, CPU_DIR)

from utils.ecg_loader import load_ecg
from inference_cpu import TinyCardioCPU


def render_ascii_waveform(samples: np.ndarray, height: int = 10, width: int = 60) -> List[str]:
    """Renders a 1D signal window into an ASCII waveform grid."""
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Resample to width
    indices = np.linspace(0, len(samples) - 1, width).astype(int)
    downsampled = samples[indices]
    
    min_v, max_v = np.min(downsampled), np.max(downsampled)
    span = max_v - min_v if (max_v - min_v) > 1e-4 else 1.0
    
    for col, val in enumerate(downsampled):
        norm = (val - min_v) / span
        row = int((1.0 - norm) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = '█' if (0.45 <= norm <= 0.55) else ('▲' if norm > 0.7 else ('▼' if norm < 0.3 else '•'))
        
    return ["".join(r) for r in grid]


def run_cli_monitor(record_path: str = 'data/vfdb/418', alert_threshold: float = 0.50, max_seconds: int = 40):
    """Executes live CLI monitoring session."""
    rec = load_ecg(record_path, n_samples=None, physical=True)
    fs = rec.header.get('sampling_rate', 250.0)
    ch0 = rec.signals[0]
    
    if fs != 250.0:
        from scipy import signal
        num_target = int(len(ch0) * (250.0 / fs))
        sig = signal.resample(ch0, num_target)
    else:
        sig = np.array(ch0, dtype=np.float32)
        
    weights_path = os.path.join(os.path.dirname(__file__), 'model_weights', 'tinycardio_cpu_weights.npz')
    model = TinyCardioCPU(weights_path, alert_threshold=alert_threshold)
    
    window_size = 250
    step_size = 125
    total_samples = min(len(sig), int(max_seconds * 250))
    
    print("\033[2J\033[H", end="") # Clear screen
    print("=" * 65)
    print("  TINYCARDIO REAL-TIME CLINICAL TRIAGE MONITOR  (CPU ENGINE)  ")
    print(f"  Patient Telemetry Record: {os.path.basename(record_path)}")
    print("=" * 65)
    
    for start in range(0, total_samples - window_size + 1, step_size):
        w = sig[start:start + window_size]
        curr_t = start / 250.0
        res = model.predict_window(w)
        score = res['risk_score']
        is_alert = res['is_alert']
        
        # Color coding
        if score >= 0.75:
            color = "\033[1;31m" # Bold Red
            tag = "CRITICAL ARREST / VF-VT ALERT"
        elif score >= 0.50:
            color = "\033[1;33m" # Yellow
            tag = "ELEVATED ISCHEMIC RISK"
        else:
            color = "\033[1;32m" # Green
            tag = "NORMAL SINUS RHYTHM"
        reset = "\033[0m"
        
        # Bar gauge
        bar_len = 25
        filled = int(score * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        waveform_lines = render_ascii_waveform(w, height=8, width=60)
        
        # Move cursor to top of display area
        print("\033[5;1H", end="")
        print(f"Time: {curr_t:6.1f}s | Latency: {res['latency_ms']:.2f} ms")
        print(f"Risk Score: {color}[{bar}] {score*100:5.1f}%{reset}  Status: {color}{tag}{reset}")
        print("-" * 65)
        for line in waveform_lines:
            print(f"  \033[36m{line}\033[0m")
        print("-" * 65)
        print("  Press Ctrl+C to terminate live telemetry session.")
        
        time.sleep(0.08) # Smooth visual playback
        
    print(f"\nMonitoring of {record_path} completed successfully.\n")


if __name__ == '__main__':
    rec_arg = 'data/vfdb/418'
    if len(sys.argv) > 1:
        rec_arg = sys.argv[1]
    run_cli_monitor(rec_arg, max_seconds=20)
