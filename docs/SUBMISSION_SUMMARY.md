# TinyCardio: Official Hackathon Submission Summary
**World Bank Group & Korea MSIT/MOFE Global AI Summit Hackathon 2026**

---

## 1. Project Overview

| Attribute | Details |
|:---|:---|
| **Project Title** | **TinyCardio: Autonomous Edge AI Cardiac Triage for Low-Resource Environments** |
| **Theme / Track** | Artificial Intelligence for Global Health Equity, Frugal Innovation & NCD Prevention |
| **Repository URL** | [https://github.com/sebastian-gfrd/TinyCardio.git](https://github.com/sebastian-gfrd/TinyCardio.git) |
| **Primary SDGs** | **SDG 3.4** (Reduce premature NCD mortality) & **SDG 9.5** (Frugal engineering & innovation in LMICs) |
| **Hardware BOM** | **$4.30 USD** per complete edge unit (MCU + AD8232 + 2G GSM + Power) |
| **Model Footprint**| **540 bytes Flash ROM**, **< 2.5 KB Static SRAM**, **0 mallocs / 0 dynamic memory** |
| **Inference Latency** | **< 0.8 milliseconds** per 1-second ECG window (250 Hz) |
| **Telemetry Payload**| **11 bytes (Binary Hex + CRC8)** or **32 characters (SMS Text)** over 2G GSM |

---

## 2. Architecture & The Three Pillars

TinyCardio was developed strictly adhering to three modular, production-ready pillars:

```
                                  TINYCARDIO ARCHITECTURE
                                  
   +─────────────────────────────────────────────────────────────────────────────────+
   |  [PILAR 1] 01_training_pipeline/                                                |
   |  - Accelerated JAX + CUDA 12 Training on NVIDIA GeForce RTX 5070                |
   |  - 4 PhysioNet DBs: NSRDB (Control), EDB (Ischemia), VFDB (Lethal), SDDB (Arrest)|
   |  - Binary Focal Loss (gamma=2.0, alpha=0.55) & Optax Cosine Decay Scheduler     |
   |  - 6 Biomedical Noise Augmentations (Respiration drift, EMG, 50/60Hz, contact) |
   |  - Results: 87.18% Global Sensitivity, 100% on Malignant VF, 93.2% on Arrest    |
   +────────────────────────────────────────┬────────────────────────────────────────+
                                            │
                                            ▼
   +─────────────────────────────────────────────────────────────────────────────────+
   |  [PILAR 2] 02_model_cpu/                                                        |
   |  - Deterministic NumPy Inference Engine for District Gateways & Rural Laptops   |
   |  - Throughput: 1,411 windows / second (< 0.71 ms / window)                      |
   |  - Real-Time Live Streaming Patient Simulator on Unseen Clinical Records        |
   |  - Zero-Dependency HTTP Gateway Server + HTML5 Canvas Oscilloscope Dashboard    |
   |  - Rich ANSI Interactive Terminal Monitor with real-time risk gauge             |
   +────────────────────────────────────────┬────────────────────────────────────────+
                                            │
                                            ▼
   +─────────────────────────────────────────────────────────────────────────────────+
   |  [PILAR 3] 03_edge_serialized/                                                  |
   |  - INT8 Symmetric Weight Quantization to 540 bytes Flash ROM                    |
   |  - Zero-Allocation Static C99 Inference Engine (< 2.5 KB RAM, malloc-free)      |
   |  - Pure Fixed-Point Integer Engine (Q7/Q15) with 33-entry Sigmoid LUT           |
   |  - AT Command Finite State Machine Driver for SIM800L / Quectel M95 GSM Modems  |
   |  - Turnkey Reference Firmware for RP2040 (Pico SDK), STM32 (HAL), & ESP32      |
   |  - Cross-Engine Parity: 100% Bit-Exact JAX vs CPU, 0.9892 Correlation with INT8 |
   +─────────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Quick-Start Guide for Hackathon Judges (60-Second Verification)

All verification scripts are standalone and pre-configured. Run these commands from the repository root:

### Step 1: Verify Cross-Engine Numerical Parity (Pillar 1 vs 2 vs 3)
```bash
python3 03_edge_serialized/cross_verify_engines.py
```
*Expected Output:* Confirms bit-exact match between JAX GPU and NumPy CPU, and 0.9892 Pearson correlation with C99 INT8 quantized weights.

### Step 2: Run Real-Time Patient Streaming Simulation (Pillar 2)
```bash
# Simulates a patient in Sudden Ventricular Fibrillation (VFDB record 418)
python3 02_model_cpu/run_patient_simulation.py data/vfdb/418

# Simulates a healthy control subject in Normal Sinus Rhythm (NSRDB record 16265)
python3 02_model_cpu/run_patient_simulation.py data/nsrdb/16265
```

### Step 3: Launch Local Diagnostic Gateway & Oscilloscope Web UI (Pillar 2)
```bash
python3 02_model_cpu/gateway_server.py --port 8080
# Open http://localhost:8080 in any browser to see the real-time ECG oscilloscope!
```

### Step 4: Run Clinical Metrics Validation Suite (Pillar 1)
```bash
python3 01_training_pipeline/evaluate_clinical_metrics.py
```
*Expected Output:* Generates full confusion matrix, ROC-AUC curve data, and per-pathology sensitivity breakdown.

---

## 4. Key Performance Indicators & Hardware Benchmarks

| Metric | Target Requirement | TinyCardio Measured Result | Compliance Status |
|:---|:---:|:---:|:---:|
| **ROM Footprint** | < 16 KB Flash | **540 bytes** | **EXCEEDED (30x smaller)** |
| **RAM Footprint** | < 8 KB SRAM | **< 2.5 KB static** | **EXCEEDED (3x smaller)** |
| **Memory Allocation** | Zero `malloc` calls | **100% Static Tensor Arena** | **VERIFIED** |
| **Inference Latency** | < 100 ms | **< 0.8 ms** on CPU | **EXCEEDED (125x faster)** |
| **Lethal VF Sensitivity** | > 90% | **100.0%** (PhysioNet VFDB) | **VERIFIED** |
| **Pre-Arrest Sensitivity** | > 85% | **93.2%** (PhysioNet SDDB) | **VERIFIED** |
| **Control False Alarms** | < 5% | **0.0%** (PhysioNet NSRDB) | **VERIFIED** |
| **Telemetry Payload** | < 160 chars (1 SMS)| **32 chars text / 11 bytes hex**| **EXCEEDED (5x smaller)** |
| **Total Hardware Cost**| < $10.00 USD | **$4.30 USD** | **EXCEEDED (57% cheaper)** |

---

## 5. Repository File Manifest

```
TinyCardio/
├── 01_training_pipeline/
│   ├── processed_data/dataset_250hz.npz       # Preprocessed 250Hz patient-partitioned dataset
│   ├── checkpoints/model_best_jax.npz        # Best trained JAX model parameters
│   ├── dataset_generator.py                  # Bandpass filtering & anti-aliased resampling
│   ├── augmentations.py                      # 6 biomedical noise augmentation models
│   ├── test_augmentations.py                 # Augmentation unit test suite
│   ├── model.py                              # JAX 1D-CNN tiny architecture definition
│   ├── train_jax_gpu.py                      # GPU training loop with Focal Loss & AdamW Cosine
│   ├── evaluate_clinical_metrics.py          # Clinical ROC-AUC and sensitivity validation
│   ├── requirements_gpu.txt                  # Python dependencies
│   └── setup_env.sh                          # CUDA 12 environment installer
│
├── 02_model_cpu/
│   ├── model_weights/                        # Exported weights (NumPy .npz and JSON)
│   ├── web/index.html                        # Zero-dependency HTML5 Canvas Oscilloscope UI
│   ├── inference_cpu.py                      # Deterministic NumPy CPU engine
│   ├── benchmark_cpu.py                      # Latency and throughput micro-benchmark
│   ├── run_patient_simulation.py             # Real-time streaming patient simulator
│   ├── cli_monitor.py                        # Terminal ANSI live ECG oscilloscope
│   └── gateway_server.py                     # HTTP rural triage gateway server
│
├── 03_edge_serialized/
│   ├── include/model_weights.h               # Flash ROM INT8 C header (540 bytes)
│   ├── src/tinyml_infer.h / tinyml_infer.c   # Static C99 inference engine
│   ├── src/pure_fixed_point_infer.h / .c     # Q7/Q15 integer-only inference engine
│   ├── src/sms_alert_encoder.h / .c          # Binary & text 2G SMS encoder with CRC8
│   ├── src/gsm_modem_driver.h / .c           # AT command GSM modem state machine
│   ├── examples/rp2040_pico_firmware.c       # Reference firmware for Raspberry Pi Pico
│   ├── examples/stm32_hal_firmware.c         # Reference firmware for STM32 HAL (DMA+Timer)
│   ├── examples/esp32_arduino.ino            # Reference sketch for ESP32 Arduino
│   ├── quantize_int8.py                      # Float32 to INT8 quantizer
│   ├── test_edge_emulator.py                 # INT8 emulator and unit tests
│   ├── cross_verify_engines.py               # 3-engine numerical parity audit
│   └── Makefile                              # Build configuration
│
├── data/                                     # Raw PhysioNet datasets (nsrdb, edb, vfdb, sddb)
├── metadata/                                 # Catalog CSVs and clinical validation reports
├── docs/
│   ├── ANALISIS_CLINICO_TECNICO.md           # Pathophysiology & clinical considerations
│   ├── WORLD_BANK_IMPACT_REPORT.md           # Socioeconomic impact & BOM economic analysis
│   ├── PITCH_SCRIPT_3MIN.md                  # 3-minute video pitch presentation script
│   └── SUBMISSION_SUMMARY.md                 # This submission summary document
│
├── tinycardio_specification.md               # Project technical specification
└── README.md                                 # Main project documentation
```

---

## 6. Intellectual Property & Ethical Compliance

TinyCardio is released as **open-source public domain software under the MIT License**, designed to foster non-profit humanitarian adoption by health ministries, NGOs, and medical device makers in developing nations. All clinical training datasets are derived from PhysioNet under the Open Access ODC-BY 1.0 license with anonymized patient data.
