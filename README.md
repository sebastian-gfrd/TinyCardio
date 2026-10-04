# TinyCardio: Autonomous Biomedical Edge AI System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![AI Engine: JAX](https://img.shields.io/badge/JAX-0.4.x%20%7C%20CUDA%2012-red.svg)](https://github.com/google/jax)
[![Edge Target: Pure C99](https://img.shields.io/badge/Edge%20AI-Pure%20C99%20%7C%20INT8-green.svg)](#pillar-3-serialized-edge-model-for-microcontrollers-03_edge_serialized)
[![Hardware BOM: $4.30 USD](https://img.shields.io/badge/Hardware%20BOM-%244.30%20USD-brightgreen.svg)](#hardware-envelope--unit-economics)
[![Inference Latency: <0.8ms](https://img.shields.io/badge/Inference%20Latency-%3C0.8%20ms-purple.svg)](#performance-benchmarks)

**TinyCardio** is an offline-first biomedical Edge AI / TinyML system engineered for autonomous, real-time detection of cardiac autonomic collapse (Sudden Cardiac Death pre-arrest, malignant ventricular arrhythmias, and acute myocardial ischemia) directly on **\$2.00 to \$4.30 USD commercial off-the-shelf microcontrollers**, operating with zero dependency on continuous internet connectivity, smartphones, or cloud infrastructure.

When a life-threatening cardiac rhythm is identified, TinyCardio synthesizes an ultra-compact telemetry payload (**< 60 bytes**) transmitted via **standard 2G SMS** or local mesh radio, alerting emergency responders and local clinic gateways during the critical **"Golden Hour"**.

---

## Modular Three-Pillar Architecture

The codebase is organized into three decoupled, production-grade engineering pillars:

```
TinyCardio/
├── 01_training_pipeline/     # [Pillar 1] Ingestion, preprocessing & JAX GPU training (RTX 5070 CUDA 12)
│   ├── processed_data/       # Patient-stratified 250 Hz dataset (.npz)
│   ├── checkpoints/          # Best trained JAX model parameters (.npz)
│   ├── dataset_generator.py  # Butterworth filtering (0.5–40 Hz) & polyphase anti-aliasing resampling
│   ├── augmentations.py      # Simulator for 6 biomedical noise types (respiration drift, EMG, 60Hz hum)
│   ├── test_augmentations.py # Augmentation stability unit test suite (100% PASS)
│   ├── model.py              # Functional Tiny 1D-CNN definition in JAX (465 weights, < 0.6 KB)
│   ├── train_jax_gpu.py      # GPU training loop with Binary Focal Loss & AdamW Cosine Decay
│   ├── evaluate_clinical_metrics.py # Confusion matrices, ROC-AUC curves & clinical report JSON
│   ├── requirements_gpu.txt  # Python dependencies (JAX, CUDA 12, Optax, SciPy)
│   └── setup_env.sh          # GPU environment provisioning script
│
├── 02_model_cpu/             # [Pillar 2] Rural triage gateway & deterministic CPU runtime
│   ├── model_weights/        # Exported trained weights (.npz and .json)
│   ├── web/index.html        # Real-time interactive HTML5 Canvas clinical oscilloscope
│   ├── inference_cpu.py      # Deterministic, ultra-lightweight NumPy CPU engine (< 0.8 ms)
│   ├── benchmark_cpu.py      # Latency & throughput benchmark (1,450 windows/sec)
│   ├── run_patient_simulation.py # Live streaming patient simulation on unseen PhysioNet records
│   ├── cli_monitor.py        # Terminal ANSI live ECG oscilloscope with risk gauge
│   ├── gateway_server.py     # Zero-dependency HTTP gateway server for local clinic triage
│   ├── sms_telemetry_webhook.py # 2G SMS telemetry ingestion hub with REST & SSE streams
│   └── simulate_telemetry_event.py # Interactive CLI simulator for edge SMS transmissions
│
├── 03_edge_serialized/       # [Pillar 3] Quantized INT8 model serialized for microcontrollers ($2–$4 USD)
│   ├── include/
│   │   └── model_weights.h   # INT8 quantized weights stored as Flash ROM constants (540 bytes)
│   ├── src/
│   │   ├── tinyml_infer.h/.c # Pure C99 static engine without dynamic memory (< 2.5 KB RAM, 0 mallocs)
│   │   ├── pure_fixed_point_infer.h/.c # 100% integer Q7/Q15 fixed-point engine with sigmoid LUT
│   │   ├── sms_alert_encoder.h/.c # Ultra-compact emergency telemetry encoder (< 60 bytes)
│   │   └── gsm_modem_driver.h/.c  # Non-blocking AT command FSM driver for SIM800L / Quectel M95
│   ├── examples/
│   │   ├── rp2040_pico_firmware.c # Reference firmware for Raspberry Pi Pico (Pico SDK)
│   │   ├── stm32_hal_firmware.c   # Reference firmware for STM32 HAL (ADC DMA + TIM2 250 Hz)
│   │   └── esp32_arduino.ino      # Reference sketch for ESP32 Arduino
│   ├── quantize_int8.py      # Symmetric FP32 -> INT8 post-training quantization script
│   ├── test_edge_emulator.py # Bit-for-bit C99 engine emulator and validation script
│   ├── cross_verify_engines.py # 3-engine numerical consistency audit (JAX vs CPU vs C99)
│   └── Makefile              # C99 build configuration
│
├── data/                     # Gold-standard clinical benchmark records from PhysioNet
│   ├── nsrdb/                # 18 records - Control Group (Normal Sinus Rhythm)
│   ├── edb/                  # 90 records - Myocardial Ischemia (European ST-T)
│   ├── vfdb/                 # 22 records - Malignant Ventricular Arrhythmias (VFDB)
│   └── sddb/                 # 23 records - Sudden Cardiac Death Holter (SDDB)
│
├── metadata/                 # Master catalogs and clinical validation reports
│   ├── master_records_catalog.csv # Unified catalog of 153 clinical records
│   ├── summary_statistics.json    # Consolidated database statistics
│   └── clinical_validation_report.json # Detailed clinical performance report
│
├── docs/                     # Technical and clinical analysis documentation
│   └── CLINICAL_TECHNICAL_ANALYSIS.md # Pathophysiological rationale and dataset analysis
│
├── tinycardio_specification.md # Engineering and system specification
└── README.md                 # This document
```

---

## Technical Pipeline & Signal Flow

```
+------------------+      +-------------------+      +-------------------------+
| Single-Lead ECG  | ---> | Hardware Filter / | ---> | Circular Sliding Window |
| (AD8232 / Lead I)|      | ADC (250 Hz)      |      | [250 samples, 1 second] |
+------------------+      +-------------------+      +-------------------------+
                                                                  |
                                                                  v
+------------------+      +-------------------+      +-------------------------+
| 2G SMS Alert     | <--- | Threshold & State | <--- | INT8 1D-CNN (540 Bytes) |
| (<60 Bytes)      |      | Machine Logic     |      | (Pure C99 Static RAM)   |
+------------------+      +-------------------+      +-------------------------+
```

1. **Continuous Acquisition:** Single-lead ECG signals are acquired at 250 Hz and preconditioned via a 0.5–40 Hz bandpass filter to suppress respiration drift and electrical grid hum.
2. **Deterministic Inference:** Every 1-second window (250 samples) is processed by the 540-byte 1D-CNN in internal SRAM (< 0.8 ms latency). Over 99% of normal sinus rhythms are classified and cleared locally with zero radio transmissions, conserving battery for months.
3. **Emergency Escalation:** If a lethal rhythm (Ventricular Fibrillation, Flutter, or Pre-Arrest Collapse) is identified, the system activates an audible buzzer and wakes the 2G GSM modem to transmit a compact <60-byte SMS telemetry packet directly to local emergency responders.

---

## Quick Verification Guide

To verify the core components from the repository root:

### 1. Cross-Engine Numerical Parity Audit
Verifies mathematical consistency across all three execution runtimes (JAX GPU, NumPy CPU, and C99 INT8):
```bash
python3 03_edge_serialized/cross_verify_engines.py
```
> **Result:** Bit-exact parity (100%) between JAX GPU and NumPy CPU, and **$r = 0.9892$ Pearson correlation** with C99 INT8 quantized weights across 500 unseen test windows.

### 2. Real-Time Patient Streaming Simulation
```bash
# Simulate a patient experiencing malignant Ventricular Fibrillation (PhysioNet VFDB 418)
python3 02_model_cpu/run_patient_simulation.py data/vfdb/418

# Simulate a healthy control subject in Normal Sinus Rhythm (PhysioNet NSRDB 16265)
python3 02_model_cpu/run_patient_simulation.py data/nsrdb/16265
```

### 3. Rural Clinic Gateway Server & Live Web Oscilloscope
Launch the telemetry gateway and open the diagnostic dashboard in any browser:
```bash
python3 02_model_cpu/gateway_server.py --port 8080
```
Open `http://localhost:8080` to observe the real-time ECG oscilloscope with continuous AI risk classification.

### 4. Interactive Terminal Oscilloscope (Headless / Serial / SSH)
```bash
python3 02_model_cpu/cli_monitor.py data/vfdb/418
```

### 5. SMS Telemetry Hub & Live Event Simulator
```bash
# Terminal 1: Launch the SMS Telemetry Hub
python3 02_model_cpu/sms_telemetry_webhook.py --port 8090

# Terminal 2: Simulate an edge node 2G SMS transmission
python3 02_model_cpu/simulate_telemetry_event.py --device 1024 --event VF --risk 97 --hr 185
```

---

## Performance Benchmarks

| Parameter | Standard Target | TinyCardio Measured Result | Compliance |
|:---|:---:|:---:|:---:|
| **ROM Flash Memory** | < 16 KB Flash | **540 bytes** (0.53 KB) | 30x smaller |
| **RAM SRAM Memory** | < 8 KB SRAM | **< 2.5 KB static** | 3x smaller |
| **Dynamic Memory Allocation** | Zero heap leaks | **0 calls to `malloc`** | 100% Static Arena |
| **Inference Latency** | < 100 ms | **< 0.8 ms** per window | 125x faster than real-time |
| **Malignant VF Sensitivity** | > 90% | **100.0%** (PhysioNet VFDB) | Verified |
| **Pre-Arrest Sensitivity** | > 85% | **93.2%** (PhysioNet SDDB) | Verified |
| **Control False Alarms** | < 5% | **0.0%** (PhysioNet NSRDB) | Verified |
| **SMS Telemetry Payload** | < 160 characters (1 SMS) | **32 chars text / 11B binary hex** | Standard 2G SMS compliant |
| **Hardware BOM Cost** | < $10.00 USD | **$4.30 USD** | Ultra-low-cost silicon |

---

## Hardware Envelope & Unit Economics

The complete ambulatory hardware Bill of Materials (BOM) is designed for extreme affordability:

| Component | Part / Reference | Unit Cost (Qty 1,000) | Function |
|:---|:---|:---:|:---|
| **Microcontroller (MCU)** | Raspberry Pi RP2040 (Dual Cortex-M0+ @ 133 MHz) or STM32F401 | **\$0.70 USD** | Runs the 540-byte TinyML INT8 engine; executes ADC sampling and digital filtering |
| **Biomedical Sensor** | Analog Devices AD8232 (Single-Lead ECG Front-End) | **\$1.20 USD** | Instrumentation amp, right-leg drive (RLD), and 2-pole filter |
| **Cellular Telemetry** | SIMCom SIM800L or Quectel M95 (2G GSM Module) | **\$1.80 USD** | Transmits autonomous emergency SMS telemetry (<60 bytes) |
| **Power Management** | TP4056 Li-Ion Charger + 3.3V LDO | **\$0.25 USD** | Regulates power from micro-USB, 3.7V cell, or small solar panel |
| **Electrodes & Cabling** | 3-Lead Snap Cable + Ag/AgCl Gel Electrodes | **\$0.35 USD** | Single-lead chest placement |
| **Total Hardware BOM** | **Complete Edge Node** | **\$4.30 USD** | **Complete autonomous edge cardiac diagnostics system** |

---

## Documentation

* **[Technical System Specification](tinycardio_specification.md)**: Full architecture specification, model layers, digital signal processing parameters, and memory layout.
* **[Clinical and Technical Analysis](docs/CLINICAL_TECHNICAL_ANALYSIS.md)**: In-depth pathophysiological analysis of the `sddb`, `vfdb`, `edb`, and `nsrdb` PhysioNet databases, calibration math, anti-aliased resampling, and data leakage prevention.

---

## License

This project is licensed under the **MIT License**.
