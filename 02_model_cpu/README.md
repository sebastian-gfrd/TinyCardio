# TinyCardio — CPU Inference, Gateway & Triage Module (02_model_cpu)

This directory contains the optimized **CPU execution runtime**, local clinic gateway server, live interactive oscilloscope dashboard, and 2G SMS telemetry hub for the TinyCardio 1D-CNN model. It is designed for rural primary health posts, Community Health Worker (CHW) field laptops, and district emergency dispatch centers.

---

## Directory Contents

* **`model_weights/`**: Contains the trained model weights exported in `.npz` and `.json` formats.
* **`inference_cpu.py`**: Deterministic, ultra-lightweight NumPy inference engine executing 1D convolution, pooling, and sigmoid classification without framework overhead (< 0.8 ms latency per window).
* **`benchmark_cpu.py`**: Performance benchmarking suite measuring latency distribution and continuous window processing throughput.
* **`run_patient_simulation.py`**: Real-time streaming simulator that streams patient ECG signals window by window and triggers autonomous clinical alerts.
* **`cli_monitor.py`**: Interactive ANSI terminal monitor rendering a live ASCII oscilloscope waveform and dynamic risk gauge for headless or SSH remote environments.
* **`gateway_server.py`**: Zero-dependency Python HTTP server serving REST endpoints (`GET /`, `POST /api/predict`, `GET /api/stream_patient`) and an interactive HTML5 Canvas oscilloscope dashboard (`web/index.html`).
* **`sms_telemetry_webhook.py`**: Microservice that ingests incoming 2G SMS payloads (<60 bytes), correlates them with pre-registered rural patient medical records, tracks Golden Hour countdowns, and broadcasts real-time updates via Server-Sent Events (SSE).
* **`simulate_telemetry_event.py`**: Command-line utility to simulate edge node emergency SMS transmission into the triage hub for live hackathon demonstrations.
* **`web/index.html`**: Clean dark-mode clinical dashboard with real-time waveform visualization, BPM detection, and triage level gauges.

---

## Quick Usage

### 1. Evaluate a Test Window in Python
```python
from inference_cpu import TinyCardioCPU
import numpy as np

# Load inference engine (alert threshold = 0.5)
model = TinyCardioCPU('model_weights/tinycardio_cpu_weights.npz', alert_threshold=0.5)

# ECG window of 250 samples (1 second at 250 Hz)
ecg_window = np.random.randn(250)

# Prediction
result = model.predict_window(ecg_window)
print(result)
# {'risk_score': 0.8741, 'classification': 'HIGH_ALERT_PATHOLOGY', 'is_alert': True, 'latency_ms': 0.68}
```

### 2. Run CPU Latency and Throughput Benchmark
```bash
python3 02_model_cpu/benchmark_cpu.py
```
*Measured Performance:* **$< 0.70\text{ ms}$** mean latency per 1-second ECG window on standard x86 CPU (> 1,400 windows/second throughput).

### 3. Simulate Real-Time Patient Streaming
```bash
# Simulate a patient experiencing malignant ventricular fibrillation from VFDB
python3 02_model_cpu/run_patient_simulation.py data/vfdb/418

# Simulate a healthy control subject from NSRDB
python3 02_model_cpu/run_patient_simulation.py data/nsrdb/16265
```

### 4. Launch Rural Clinic Triage Gateway & Web Dashboard
```bash
python3 02_model_cpu/gateway_server.py --port 8080
# Open http://localhost:8080 in any web browser
```

### 5. Launch Terminal Live Oscilloscope (Headless / SSH Mode)
```bash
python3 02_model_cpu/cli_monitor.py data/vfdb/418
```

### 6. Run the 2G SMS Telemetry Hub & Dispatch Webhook
```bash
python3 02_model_cpu/sms_telemetry_webhook.py --port 8090
```
In another terminal, simulate an emergency event transmission:
```bash
python3 02_model_cpu/simulate_telemetry_event.py --device 1024 --event VF --risk 97 --hr 185
```
