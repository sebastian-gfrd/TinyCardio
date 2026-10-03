# TinyCardio — Modelo Serializado para Equipos de Bajo Costo (03_edge_serialized)

Este directorio contiene el modelo **TinyCardio 1D-CNN cuantizado a INT8** y serializado en **C99 puro sin asignación dinámica de memoria (`malloc`-free)**, listo para ser integrado directamente en el firmware de microcontroladores de **\$2 a \$4 USD** (Raspberry Pi Pico RP2040, STM32F401, ESP32-C3 / RISC-V).

---

## Estructura del Directorio

```
03_edge_serialized/
├── include/
│   └── model_weights.h       # Pesos INT8 como arrays 'const' en memoria Flash ROM
├── src/
│   ├── tinyml_infer.h        # Interfaz de inferencia en C
│   ├── tinyml_infer.c        # Motor de convolución 1D y GAP (sin malloc, RAM < 2.5 KB)
│   ├── sms_alert_encoder.h   # Codificador de telemetría de emergencia (< 60 bytes)
│   └── sms_alert_encoder.c   # Formateo de SMS 2G texto y payload binario hex con CRC8
├── quantize_int8.py          # Script de cuantización FP32 -> INT8 y generación del .h
├── main_edge_test.c          # Simulador y banco de pruebas en C
├── Makefile                  # Script de compilación con gcc
└── README.md                 # Esta documentación
```

---

## Restricciones y Especificaciones de Hardware

* **Consumo de Memoria ROM (Flash):** $\sim 0.5\text{ KB}$ a $1.2\text{ KB}$ (cabe holgadamente en chips de 64 KB ROM).
* **Consumo de Memoria RAM:** $< 2.5\text{ KB}$ SRAM (utiliza un buffer estático de trabajo *Tensor Arena*, sin llamadas a `malloc` ni fragmentación del heap).
* **Consumo Energético:** $< 25\text{ mW}$ en inferencia activa continua.
* **Microcontroladores Soportados:**
  * Raspberry Pi Pico (RP2040, dual Cortex-M0+ a 133 MHz).
  * STM32F401 "Black Pill" (ARM Cortex-M4 con FPU a 84 MHz).
  * ESP32-C3 (RISC-V a 160 MHz) o ESP32 original (Xtensa).

---

## Compilación y Ejecución del Simulador en C

Para compilar y verificar el comportamiento del modelo en C en cualquier entorno POSIX:

```bash
cd 03_edge_serialized
make clean
make
./edge_simulator
```

---

## Codificación de Alertas de Emergencia (<60 Bytes)

El módulo `sms_alert_encoder` genera dos formatos de payload para zonas rurales sin internet:

1. **SMS 2G Texto Legible (~32 caracteres):**
   ```
   TC:D=2A1B;E=VF/VT;R=96%;HR=192;T=1845s
   ```
2. **Payload Binario Hexadecimal (22 caracteres / 11 bytes con CRC8):**
   ```
   54432A1B0000073538C053
   ```
Ambos formatos permiten la transmisión instantánea a través de módems GSM/GPRS básicos (SIM800L / A6) o radios de largo alcance **LoRa / BLE Mesh**, garantizando la emisión de la alerta médica incluso con redes móviles colapsadas o de baja señal.
