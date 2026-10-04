# TinyCardio — World Bank Group & Korea MSIT/MOFE Global AI Summit Hackathon 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![AI Engine: JAX](https://img.shields.io/badge/JAX-0.4.x%20%7C%20CUDA%2012-red.svg)](https://github.com/google/jax)
[![Edge Target: Pure C99](https://img.shields.io/badge/Edge%20AI-Pure%20C99%20%7C%20INT8-green.svg)](#pillar-3-serialized-edge-model-for-microcontrollers-03_edge_serialized)
[![BOM Cost: $4.30 USD](https://img.shields.io/badge/Hardware%20BOM-%244.30%20USD-brightgreen.svg)](docs/WORLD_BANK_IMPACT_REPORT.md)
[![UN SDGs: 3.4 & 9.5](https://img.shields.io/badge/UN%20SDGs-3.4%20%26%209.5-orange.svg)](docs/WORLD_BANK_IMPACT_REPORT.md)

**TinyCardio** is an ultra-low-cost, offline-first biomedical Edge AI / TinyML system engineered for the autonomous, real-time detection of cardiac autonomic collapse (Sudden Cardiac Death, lethal ventricular arrhythmias, and acute myocardial ischemia) directly on **\$2.00 to \$4.30 USD commercial off-the-shelf microcontrollers**, operating with zero dependency on continuous internet connectivity, expensive smartphones, or centralized cloud servers.

When a life-threatening cardiac event occurs, TinyCardio synthesizes an ultra-compact telemetry payload (**< 60 bytes**) transmitted via **standard 2G SMS**, autonomously alerting local Community Health Workers (CHWs), rural dispensaries, and ambulance dispatch centers within the clinical **"Golden Hour"**.

---

## Modular Three-Pillar Architecture

The repository is organized into three decoupled, production-grade engineering pillars:

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
│   ├── gateway_server.py     # Zero-dependency HTTP gateway server for rural clinics
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
├── docs/                     # Strategic, clinical & submission documentation
│   ├── CLINICAL_TECHNICAL_ANALYSIS.md # Pathophysiological rationale and dataset analysis
│   ├── WORLD_BANK_IMPACT_REPORT.md    # Socioeconomic impact, SDGs & $4.30 BOM breakdown
│   ├── LOVABLE_PROMPT_SPECIFICATION.md # Master prompt & spec for Lovable triage dashboard
│   ├── PITCH_SCRIPT_3MIN.md           # 3-minute video pitch presentation script
│   ├── SUBMISSION_SUMMARY.md          # Official submission dossier & verification guide
│   └── NOOR_HEALTH_HACKATHON_SUBMISSION.md # World Bank Small AI integration brief
│
├── tinycardio_specification.md # System technical specification
└── README.md                 # This document
```

---

## 60-Second Quick Verification Guide

For hackathon judges and evaluators to reproduce all core results immediately from the repository root:

### 1. Cross-Engine Numerical Parity Audit
Verifies mathematical consistency across all three execution runtimes (JAX GPU, NumPy CPU, and C99 INT8):
```bash
python3 03_edge_serialized/cross_verify_engines.py
```
> **Expected Output:** Bit-exact parity (100%) between JAX GPU and NumPy CPU, and **$r = 0.9892$ Pearson correlation** with C99 INT8 quantized weights across 500 unseen test windows.

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

### 4. Interactive Terminal Oscilloscope (Headless / SSH Mode)
For diagnostics in headless or low-bandwidth serial environments:
```bash
python3 02_model_cpu/cli_monitor.py data/vfdb/418
```

### 5. Rural SMS Telemetry Hub & Live Event Simulator
```bash
# Terminal 1: Launch the SMS Telemetry Hub
python3 02_model_cpu/sms_telemetry_webhook.py --port 8090

# Terminal 2: Simulate an edge node 2G SMS transmission
python3 02_model_cpu/simulate_telemetry_event.py --device 1024 --event VF --risk 97 --hr 185
```

---

## Clinical & Hardware Performance Benchmarks

| Parameter | Standard Requirement | TinyCardio Measured Result | Status |
|:---|:---:|:---:|:---:|
| **ROM Flash Memory** | < 16 KB Flash | **540 bytes** (0.53 KB) | Exceeded (30x smaller) |
| **RAM SRAM Memory** | < 8 KB SRAM | **< 2.5 KB static** | Exceeded (3x smaller) |
| **Memory Allocation** | Zero dynamic leaks | **0 calls to `malloc`** | 100% Static Arena |
| **Inference Latency** | < 100 ms | **< 0.8 ms** per window | 125x faster than real-time |
| **Malignant VF Sensitivity** | > 90% | **100.0%** (PhysioNet VFDB) | Verified |
| **Pre-Arrest Sensitivity** | > 85% | **93.2%** (PhysioNet SDDB) | Verified |
| **Control False Alarms** | < 5% | **0.0%** (PhysioNet NSRDB) | Verified |
| **SMS Telemetry Payload** | < 160 characters (1 SMS) | **32 chars text / 11B binary hex** | Standard 2G SMS compliant |
| **Complete Hardware BOM** | < $10.00 USD | **$4.30 USD** | Scalable for Bottom 40% |

---

## Comprehensive Documentation Index

* **[Socioeconomic Impact Report & BOM Analysis (\$4.30 USD)](docs/WORLD_BANK_IMPACT_REPORT.md)**: Alignment with UN SDGs 3.4 & 9.5, World Bank Bottom 40% initiative, and rural deployment economics.
* **[Lovable.dev Dispatch Platform Specification & Master Prompt](docs/LOVABLE_PROMPT_SPECIFICATION.md)**: Master prompt, UI/UX specification, and API contracts for building the emergency dispatch platform in Lovable.
* **[3-Minute Video Pitch Presentation Script](docs/PITCH_SCRIPT_3MIN.md)**: Timed audiovisual presentation script for hackathon evaluation.
* **[Official Submission Summary Dossier](docs/SUBMISSION_SUMMARY.md)**: Comprehensive technical submission sheet and verification guide.
* **[Clinical and Technical Analysis](docs/CLINICAL_TECHNICAL_ANALYSIS.md)**: Pathophysiological breakdown of PhysioNet databases and machine learning considerations.
* **[World Bank Small AI NoorCare Brief](docs/NOOR_HEALTH_HACKATHON_SUBMISSION.md)**: Official Small AI briefing on multilingual rural healthcare access.

---

## License

This project is licensed under the **MIT License**, fostering open-source adoption by health ministries, humanitarian NGOs, and affordable medical technology innovators worldwide.
