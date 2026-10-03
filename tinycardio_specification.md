# Project Specification: TinyCardio
**Event:** World Bank Group & Korea MSIT/MOFE Global AI and Digital Summit Hackathon 2026 (Seoul, Republic of Korea)  
**Track:** Health (Rural Triage & Last-Mile Telemedicine)  
**Paradigm:** Small AI / TinyML / Frugal Edge Computing  
**Author:** Sebastian  

---

## 1. Executive Summary & Problem Statement

### 1.1 The Global South Healthcare Gap
In remote and rural areas of low- and middle-income countries (LMICs), access to cardiologists, 12-lead electrocardiography (ECG) stations, and emergency transport is critically scarce. Sudden cardiac death (SCD) triggered by malignant ventricular arrhythmias (Ventricular Tachycardia/Ventricular Fibrillation) and acute ischemic episodes often causes irreversible mortality within minutes of onset.

Traditional digital health solutions rely heavily on:
- High-bandwidth continuous streaming of raw bio-signals to cloud-hosted large models.
- Expensive high-power diagnostic stations requiring consistent electrical grids.
- Complex user interfaces unsuitable for frontline community health workers (CHWs).

When cellular coverage drops, standard connected systems fail entirely.

### 1.2 The TinyCardio Solution
**TinyCardio** is an offline-first, ultra-low-power biomedical machine learning system engineered to detect pre-lethal cardiac autonomic collapse and severe ischemic anomalies directly on low-cost edge microcontrollers ($2–$4 USD). 

Instead of streaming raw analog data, TinyCardio runs continuous, deterministic inference locally using an INT8-quantized 1D Temporal Convolutional Network (1D-CNN) compiled in pure C/C++. If a high-confidence critical precursor pattern is identified, it broadcasts an emergency alert payload compressed down to < 60 bytes via bare-bones 2G SMS or local mesh radios (LoRa/BLE).

---

## 2. Core Architecture & Technical Pipeline

```
+------------------+      +-------------------+      +-------------------------+
| Single-Lead ECG  | ---> | Hardware Filter / | ---> | Circular Sliding Window |
| (AD8232 / PPG)   |      | ADC (128-250 Hz)  |      | [250-500 samples, ~2s]  |
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
* **Windowing:** Fixed sliding window of 250 to 500 samples (~2 seconds) with a 50% overlap.

### 2.2 Model Architecture (Tiny 1D-CNN)
Designed for sub-millisecond execution on bare-metal ARM Cortex-M microcontrollers:
1. **Input Layer:** Shape `(1, 250, 1)` (Single-channel normalized ECG slice).
2. **Conv1D Block 1:** 8 filters, kernel size 5, stride 2, ReLU activation, batch normalization folded during quantization.
3. **Conv1D Block 2:** 16 filters, kernel size 3, stride 2, ReLU activation.
4. **Global Average Pooling (GAP):** Condenses temporal dimension to feature vectors without dense parameter bloat.
5. **Output Dense Layer:** 1 unit with sigmoid activation $\hat{y} \in [0, 1]$, outputting instantaneous risk score.
* **Total Parameters:** $< 12,000$ weights.
* **Model Size:** $\sim 12 \text{ KB}$ (INT8 quantized).

### 2.3 Training and Compilation Stack
* **Framework:** **JAX** utilizing XLA (Accelerated Linear Algebra) targeting CPU execution.
* **Post-Training Optimization:** Symmetric 8-bit integer quantization (`int8` weights and activations).
* **Deployment Target:** Exported into a single self-contained C/C++ header file (`model_weights.h` and `tinyml_infer.c`) with zero dynamic memory allocation (`malloc`-free) to run directly on standard MCU memory buffers.

---

## 3. Dataset Strategy (PhysioNet Benchmarks)

The project leverages validated open-access records from the PhysioNet repository:

1. **Sudden Cardiac Death Holter Database (`sddb`):**
   * Pre-event interval extraction (extracting windows 1–15 minutes prior to recorded ventricular fibrillation / arrest) labeled as Class `1` (Positive / High Alert).
2. **European ST-T Database (`edb`):**
   * Verified morphological ST-segment deviations and T-wave alterations for acute ischemia detection.
3. **MIT-BIH Normal Sinus Rhythm Database (`nsrdb`):**
   * Normal cardiac behavior serving as Class `0` (Baseline / Control) to suppress false positives during physical movement or daily activity.

---

## 4. Hardware and Deployment Constraints

TinyCardio targets the absolute lowest hardware envelope:

* **Target Microcontroller:** Raspberry Pi Pico (RP2040, dual Cortex-M0+), STM32F401 "Black Pill" (Cortex-M4), or ESP32-C3 (RISC-V).
* **RAM Requirement:** $< 32 \text{ KB}$ SRAM (Tensor arena + input buffer).
* **Flash Requirement:** $< 64 \text{ KB}$ ROM (Executable + model weights).
* **Power Budget:** $< 25 \text{ mW}$ in continuous active inference mode; supports battery operation on small LiPo cells or solar trickle charge.
* **BOM Cost:** $<\$5.00\text{ USD}$ total bill of materials for compute and sensor interfacing.

---

## 5. Development Timeline for 48-Hour Hackathon (Oct 3–4, 2026)

| Milestone | Window | Deliverables |
| :--- | :--- | :--- |
| **Stage 1: Ingestion & Pipelines** | Oct 3, 08:00 - 13:00 | Load PhysioNet WFDB records, construct clean windowed training pairs (Class 0/1). |
| **Stage 2: JAX Training & Quantization** | Oct 3, 13:00 - 19:00 | Train 1D-CNN in JAX, evaluate validation ROC-AUC / F1-score, apply INT8 quantization. |
| **Stage 3: C/C++ Serialization** | Oct 3, 19:00 - 23:00 | Export weights to C flat arrays, implement forward-pass inference in pure C without dependencies. |
| **Stage 4: Edge Simulation & Interface** | Oct 4, 08:00 - 13:00 | Build a lightweight dashboard and SMS payload simulator demonstrating real-time offline classification. |
| **Stage 5: Packaging & Pitch** | Oct 4, 13:00 - 18:00 | Write documentation, record 3-minute video pitch, and finalize repository for submission. |

---

## 6. Socioeconomic Impact & Alignment with World Bank Objectives

* **Target Population:** Bottom 40% income communities and remote primary healthcare posts in Latin America, Sub-Saharan Africa, and Southeast Asia.
* **Alignment with SDGs:**
  * **SDG 3.4:** Reduce premature mortality from non-communicable diseases through early detection and management.
  * **SDG 9.5:** Support domestic technology development, research, and innovation in developing countries.
* **Operational Frugality:** Bypasses the need for broadband infrastructure by transforming low-cost silicon into an autonomous cardiac sentry.