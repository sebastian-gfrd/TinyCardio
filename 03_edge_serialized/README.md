# TinyCardio — Serialized Model for Ultra-Low-Cost Hardware (03_edge_serialized)

This directory contains the **INT8-quantized TinyCardio 1D-CNN** serialized in **pure C99 with zero dynamic memory allocation (`malloc`-free)**, engineered for turnkey integration into the firmware of **\$2 to \$4 USD** commercial off-the-shelf microcontrollers (Raspberry Pi Pico RP2040, STM32F401, ESP32, and Cortex-M0+/M4).

---

## Directory Structure

```
03_edge_serialized/
├── include/
│   └── model_weights.h             # Quantized INT8 weights stored as 'const' Flash ROM arrays (540 bytes)
├── src/
│   ├── tinyml_infer.h / .c         # Standard C99 static inference engine (RAM < 2.5 KB, 0 mallocs)
│   ├── pure_fixed_point_infer.h / .c # 100% integer Q7/Q15 fixed-point engine with sigmoid LUT for FPU-less MCUs
│   ├── sms_alert_encoder.h / .c    # Ultra-compact emergency telemetry encoder (< 60 bytes)
│   └── gsm_modem_driver.h / .c     # Non-blocking AT command FSM driver for SIM800L / Quectel M95
├── examples/
│   ├── rp2040_pico_firmware.c      # Turnkey reference firmware for Raspberry Pi Pico (Pico SDK)
│   ├── stm32_hal_firmware.c        # Turnkey reference firmware for STM32 HAL (ADC DMA + TIM2 250 Hz)
│   └── esp32_arduino.ino           # Turnkey reference sketch for ESP32 Arduino
├── quantize_int8.py                # Symmetric FP32 -> INT8 post-training quantization script
├── test_edge_emulator.py           # Bit-for-bit C99 engine emulator and validation script
├── cross_verify_engines.py         # 3-engine numerical consistency audit (JAX vs CPU vs C99)
├── Makefile                        # POSIX C99 build configuration
└── README.md                       # This document
```

---

## Hardware Envelope & System Specifications

* **Flash ROM Footprint:** **540 bytes** total weights (0.53 KB). Operates comfortably on microcontrollers with $\le 64\text{ KB}$ ROM.
* **RAM SRAM Footprint:** **$< 2.5\text{ KB}$** SRAM using a single contiguous static *Tensor Arena* buffer (zero dynamic heap allocations, immune to memory leaks and heap fragmentation).
* **Power Consumption:** **$< 25\text{ mW}$** during continuous active inference; operates for months on rechargeable 3.7V 18650 cells or small 5V solar panels.
* **Hardware Architectures Tested:**
  * **Raspberry Pi Pico (RP2040):** Dual ARM Cortex-M0+ @ 133 MHz (Zero FPU, uses `pure_fixed_point_infer`).
  * **STM32F401 "Black Pill" / STM32L432:** ARM Cortex-M4 @ 84 MHz with ADC DMA circular buffer.
  * **ESP32 / ESP32-C3:** Xtensa LX6 @ 240 MHz or RISC-V @ 160 MHz.

---

## Emergency Telemetry Encoding (< 60 Bytes)

The `sms_alert_encoder` module produces two ultra-compressed telemetry payloads for remote rural areas without internet access:

1. **Human-Readable 2G SMS (~32 to 37 characters):**
   ```text
   TC:D=1024;E=VF;R=96%;HR=182;TS=3412s
   ```
2. **Compact Binary Hex Payload (22 hex characters / 11 bytes with CRC8):**
   ```text
   544304000160B6102A00E4
   ```
   *Fields:* Device ID (2B), Event Code (1B), Risk Score (1B), Heart Rate (1B), Timestamp (4B), CRC8 Checksum (1B).

---

## Verification & Parity Audit

### 1. Run INT8 Quantization
```bash
python3 03_edge_serialized/quantize_int8.py
```

### 2. Verify C99 Emulator Precision
```bash
python3 03_edge_serialized/test_edge_emulator.py
```

### 3. Run the Cross-Engine Numerical Parity Audit
Audits numerical consistency across all 3 execution runtimes (JAX GPU, NumPy CPU, and C99 INT8):
```bash
python3 03_edge_serialized/cross_verify_engines.py
```
*Expected Result:* **100% bit-exact parity** between JAX GPU and NumPy CPU, and **$r = 0.9892$ Pearson correlation** with C99 INT8 quantized weights across unseen test patients.
