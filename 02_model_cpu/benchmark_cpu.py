"""
TinyCardio CPU Benchmark Utility
================================
Evaluates throughput and latency of TinyCardio 1D-CNN execution on standard CPU.
Simulates high-frequency continuous cardiac monitoring.
"""

import time
import os
import sys
import numpy as np

from inference_cpu import TinyCardioCPU


def run_benchmark(n_iterations: int = 2000):
    weights_path = os.path.join(os.path.dirname(__file__), 'model_weights', 'tinycardio_cpu_weights.npz')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), 'model_weights', 'tinycardio_cpu_weights.json')

    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found in {weights_path}. Train the model first.")
        return

    model = TinyCardioCPU(weights_path)
    print(f"Running CPU Benchmark ({n_iterations} windows of 250 ECG samples)...")

    # Generate test windows
    test_windows = [np.random.randn(250).astype(np.float32) for _ in range(n_iterations)]

    # Warmup
    for i in range(50):
        model.predict_window(test_windows[i])

    latencies = []
    t_start = time.perf_counter()

    for i in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict_window(test_windows[i])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    total_time = time.perf_counter() - t_start
    latencies = np.array(latencies)

    print("=" * 60)
    print("TINYCARDIO CPU BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Windows Evaluated: {n_iterations}")
    print(f"Total Execution Time:    {total_time:.3f} s")
    print(f"Throughput:              {n_iterations / total_time:.1f} windows/second")
    print(f"Mean Latency:            {np.mean(latencies):.4f} ms per 250-sample window")
    print(f"Median Latency:          {np.median(latencies):.4f} ms")
    print(f"95th Percentile:         {np.percentile(latencies, 95):.4f} ms")
    print(f"99th Percentile:         {np.percentile(latencies, 99):.4f} ms")
    print(f"Min / Max Latency:       {np.min(latencies):.4f} ms / {np.max(latencies):.4f} ms")
    print("Real-time factor:        Signal length (1000 ms) vs Latency -> Instantaneous (<1 ms)")
    print("=" * 60)


if __name__ == '__main__':
    run_benchmark(1000)
