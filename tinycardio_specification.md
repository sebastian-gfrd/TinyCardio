# Technical System Specification: TinyCardio
**Paradigm:** Embedded Biomedical Edge AI / TinyML / Ultra-Low-Power Computing  
**Author:** Sebastian  
**Status:** Production-Ready (v1.0.0 Architecture)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Remote Healthcare Monitoring Gap
In remote and rural areas worldwide, access to clinical cardiologists, 12-lead electrocardiography (ECG) stations, and emergency hospital transport is critically scarce. Sudden Cardiac Death (SCD) triggered by malignant ventricular arrhythmias (Ventricular Tachycardia / Ventricular Fibrillation) and acute ischemic episodes often causes irreversible mortality within minutes of onset.

Traditional digital health solutions rely heavily on:
- High-bandwidth continuous streaming of raw bio-signals to cloud-hosted neural networks.
- Expensive high-power diagnostic stations requiring consistent electrical grids.
- Complex user interfaces unsuitable for frontline emergency responders and field staff.

When cellular broadband or internet coverage drops, standard connected systems fail completely.

### 1.2 The TinyCardio Solution
**TinyCardio** is an offline-first, ultra-low-power biomedical machine learning system engineered to detect pre-lethal cardiac autonomic collapse and severe ischemic anomalies directly on low-cost edge microcontrollers ($2–$4 USD). 

Instead of streaming raw analog data, TinyCardio runs continuous, deterministic inference locally using an INT8-quantized 1D Temporal Convolutional Network (1D-CNN) compiled in pure C/C++. If a high-confidence critical precursor pattern is identified, it broadcasts an emergency alert payload compressed down to < 60 bytes via bare-bones 2G SMS or local mesh radios (LoRa/BLE).

---

## 2. Core Architecture & Technical Pipeline

```
+------------------+      +-------------------+      +-------------------------+
| Single-Lead ECG  | ---> | Hardware Filter / | ---> | Circular Sliding Window |
| (AD8232 / PPG)   |      | ADC (128-250 Hz)  |      | [250-500 samples, ~1s]  |
+------------------+      +-------------------+      +-------------------------+
                                                                  |
                                                                  v
+------------------+      +-------------------+      +-------------------------+
| Cellular Alert   | <--- | Threshold & State | <--- | INT8 1D-CNN Inactive    |
| (<60 byte SMS)   |      | Machine Logic     |      | (JAX -> XLA/C++ Engine) |
+------------------+      +-------------------+      +-------------------------+
```

### 2.1 Signal Acquisition and Preprocessing
* **Sampling Rate ($f_s$):** 128 Hz to 250 Hz (matching MIT-BIH and PhysioNet benchmarks).
* **Signal Conditioning:** Real-time digital band-pass filtering (0.5 Hz – 40.0 Hz Butterworth 2nd order) to remove respiration baseline wander and 50/60 Hz electromagnetic grid hum without phase distortion.
* **Windowing:** Fixed sliding window of 250 samples (1 second at 250 Hz).

### 2.2 Model Architecture (Tiny 1D-CNN)
Designed for sub-millisecond execution on bare-metal ARM Cortex-M microcontrollers:
1. **Input Layer:** Shape `(1, 250, 1)` (Single-channel normalized ECG slice).
2. **Conv1D Block 1:** 8 filters, kernel size 5, stride 2, ReLU activation.
3. **Conv1D Block 2:** 16 filters, kernel size 3, stride 2, ReLU activation.
4. **Global Average Pooling (GAP):** Condenses temporal dimension to feature vectors without dense parameter bloat.
5. **Output Dense Layer:** 1 unit with sigmoid activation $\hat{y} \in [0, 1]$, outputting instantaneous risk score.
* **Total Parameters:** 465 trainable weights.
* **Model Size:** 540 bytes (INT8 quantized).

### 2.3 Training and Compilation Stack
* **Framework:** **JAX** utilizing XLA (Accelerated Linear Algebra) with CUDA 12 GPU acceleration.
* **Post-Training Optimization:** Symmetric 8-bit integer quantization (`int8` weights and activations).
* **Deployment Target:** Exported into a single self-contained C/C++ header file (`model_weights.h` and `tinyml_infer.c`) with zero dynamic memory allocation (`malloc`-free) to run directly on standard MCU memory buffers.

---

## 3. Dataset Strategy (PhysioNet Benchmarks)

The project leverages validated open-access records from the PhysioNet repository:

1. **Sudden Cardiac Death Holter Database (`sddb`):**
   * Pre-event interval extraction (extracting windows 1–15 minutes prior to recorded ventricular fibrillation / arrest) labeled as Class `1` (Positive / High Alert).
2. **European ST-T Database (`edb`):**
   * Verified morphological ST-segment deviations and T-wave alterations for acute ischemia detection.
3. **MIT-BIH Malignant Ventricular Ectopy Database (`vfdb`):**
   * Sustained ventricular flutter and fibrillation episodes for malignant arrhythmia triage.
4. **MIT-BIH Normal Sinus Rhythm Database (`nsrdb`):**
   * Normal cardiac behavior serving as Class `0` (Baseline / Control) to suppress false positives during physical movement or daily activity.

---

## 4. Hardware and Deployment Constraints

TinyCardio targets the lowest hardware envelope:

* **Target Microcontroller:** Raspberry Pi Pico (RP2040, dual Cortex-M0+), STM32F401 "Black Pill" (Cortex-M4), or ESP32-C3 (RISC-V).
* **RAM Requirement:** $< 2.5 \text{ KB}$ SRAM (Tensor arena + input buffer).
* **Flash Requirement:** $< 16 \text{ KB}$ ROM (Executable + model weights).
* **Power Budget:** $< 25 \text{ mW}$ in continuous active inference mode; supports battery operation on small LiPo cells or solar trickle charge.
* **BOM Cost:** $<\$5.00\text{ USD}$ total bill of materials for compute and sensor interfacing.

---

## 5. Architectural Engineering Pillars

| Pillar | Directory | Purpose |
| :--- | :--- | :--- |
| **Pillar 1: Training & Ingestion** | `01_training_pipeline/` | Ingestion of PhysioNet records, anti-aliased resampling, biomedical noise augmentation, and JAX GPU training with Focal Loss. |
| **Pillar 2: CPU Runtime & Gateway** | `02_model_cpu/` | High-throughput deterministic NumPy inference engine, streaming patient simulator, web dashboard, and 2G SMS telemetry hub. |
| **Pillar 3: Serialized Edge Firmware** | `03_edge_serialized/` | INT8 C99 static engine, pure fixed-point Q7/Q15 engine, GSM modem driver, and turnkey reference firmware for RP2040, STM32, and ESP32. |

---

## 6. Clinical Safety & Deployment Paradigm

* **Autonomous Edge Triage:** 100% on-device classification without external data streaming.
* **Ultra-Low Bandwidth Telemetry:** Emergency dispatch triggered via compact 2G SMS (<60 bytes) or binary hex payload.
* **Clinical Safety First:** Zero dynamic memory allocations (`malloc`-free) eliminates buffer overruns, heap fragmentation, and runtime crashes on long-term cardiac monitoring.