# TinyCardio — World Bank Group & Korea MSIT/MOFE Global AI Summit Hackathon 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![AI Engine: JAX](https://img.shields.io/badge/JAX-0.4.x%20%7C%20CUDA%2012-red.svg)](https://github.com/google/jax)
[![Edge Target: Pure C99](https://img.shields.io/badge/Edge%20AI-Pure%20C99%20%7C%20INT8-green.svg)](#pilar-3-modelo-serializado-para-microcontroladores-03_edge_serialized)
[![BOM Cost: $4.30 USD](https://img.shields.io/badge/Hardware%20BOM-%244.30%20USD-brightgreen.svg)](docs/WORLD_BANK_IMPACT_REPORT.md)
[![UN SDGs: 3.4 & 9.5](https://img.shields.io/badge/UN%20SDGs-3.4%20%26%209.5-orange.svg)](docs/WORLD_BANK_IMPACT_REPORT.md)

**TinyCardio** es un sistema biomédico de borde (*Edge AI / TinyML*) diseñado para la detección temprana, autónoma y de costo ultra-bajo de colapso autonómico cardíaco (Muerte Súbita Cardíaca, arritmias ventriculares letales e isquemia miocárdica aguda) directamente en microcontroladores de **\$2 a \$4 USD**, operando sin requerir conexión a internet continua, teléfonos inteligentes ni servidores en la nube.

Cuando se detecta un evento letal, TinyCardio genera un paquete de telemetría de emergencia ultra-compacto (**<60 bytes**) transmitido vía **SMS sobre redes celulares 2G**, alertando a personal de salud comunitario (*Community Health Workers*) y dispensarios rurales.

---

## Estructura Modular de Tres Pilares

El proyecto está organizado en tres pilares de ingeniería independientes y complementarios:

```
WorldBank/
├── 01_training_pipeline/     # [Pilar 1] Ingesta, preprocesamiento y entrenamiento JAX con GPU (RTX 5070 CUDA)
│   ├── processed_data/       # Dataset balanceado y estratificado a 250 Hz (.npz)
│   ├── checkpoints/          # Checkpoints de pesos óptimos entrenados (.npz)
│   ├── dataset_generator.py  # Segmentación con filtros Butterworth y remuestreo polifase
│   ├── augmentations.py      # Simulador de 6 tipos de ruido biomédico (respiración, EMG, 60Hz)
│   ├── test_augmentations.py # Batería de pruebas unitarias de aumentación
│   ├── model.py              # Definición de la arquitectura Tiny 1D-CNN en JAX (465 pesos)
│   ├── train_jax_gpu.py      # Bucle acelerado GPU con Focal Loss y Cosine Decay
│   ├── evaluate_clinical_metrics.py # Matriz de confusión, curvas ROC-AUC y métricas clínicas
│   ├── requirements_gpu.txt  # Dependencias JAX + CUDA 12
│   └── setup_env.sh          # Script de aprovisionamiento de entorno GPU
│
├── 02_model_cpu/             # [Pilar 2] Gateway y servidor de triaje rural en CPU
│   ├── model_weights/        # Pesos entrenados exportados (.npz y .json)
│   ├── web/index.html        # Osciloscopio interactivo en tiempo real (HTML5 Canvas)
│   ├── inference_cpu.py      # Motor de inferencia determinista ultraligero en NumPy (<0.8 ms)
│   ├── benchmark_cpu.py      # Pruebas de latencia (1,411 ventanas/segundo de throughput)
│   ├── run_patient_simulation.py # Simulación en streaming en tiempo real de registros PhysioNet
│   ├── cli_monitor.py        # Osciloscopio interactivo para terminal ANSI
│   └── gateway_server.py     # Servidor HTTP ligero para clínicas rurales
│
├── 03_edge_serialized/       # [Pilar 3] Modelo serializado para microcontroladores (Edge / TinyML)
│   ├── include/
│   │   └── model_weights.h   # Pesos cuantizados INT8 como constantes en Flash ROM (540 bytes)
│   ├── src/
│   │   ├── tinyml_infer.h/.c # Motor C99 estático sin asignación dinámica (< 2.5 KB RAM, 0 mallocs)
│   │   ├── pure_fixed_point_infer.h/.c # Motor 100% entero punto fijo (Q7/Q15) para Cortex-M0+
│   │   ├── sms_alert_encoder.h/.c # Codificador de payload SMS 2G (11B hex o 32 caracteres)
│   │   └── gsm_modem_driver.h/.c  # Driver de máquina de estados AT para SIM800L / Quectel M95
│   ├── examples/
│   │   ├── rp2040_pico_firmware.c # Firmware de referencia para Raspberry Pi Pico (Pico SDK)
│   │   ├── stm32_hal_firmware.c   # Firmware de referencia para STM32 HAL (ADC DMA + Timer)
│   │   └── esp32_arduino.ino      # Sketch de referencia para ESP32 Arduino
│   ├── quantize_int8.py      # Conversor de pesos flotantes a INT8 simétrico
│   ├── test_edge_emulator.py # Emulador y verificador de precisión INT8
│   ├── cross_verify_engines.py # Auditoría de paridad numérica entre los tres motores
│   └── Makefile              # Configuración de compilación C99
│
├── data/                     # Bases de datos clínicas de PhysioNet
│   ├── nsrdb/                # 18 registros - Grupo de Control (Normal Sinus Rhythm)
│   ├── edb/                  # 90 registros - Isquemia Miocárdica (European ST-T)
│   ├── vfdb/                 # 22 registros - Arritmias Malignas Ventriculares (VFDB)
│   └── sddb/                 # 23 registros - Paro Cardíaco y Muerte Súbita (SDDB)
│
├── metadata/                 # Catálogos maestros y reportes
│   ├── master_records_catalog.csv # 153 registros indexados
│   ├── summary_statistics.json    # Estadísticas consolidadas
│   └── clinical_validation_report.json # Reporte de validación clínica
│
├── docs/                     # Documentación técnica y estratégica
│   ├── ANALISIS_CLINICO_TECNICO.md # Fundamentación médica y fisiopatológica
│   ├── WORLD_BANK_IMPACT_REPORT.md # Impacto socioeconómico, ODS y desglose BOM ($4.30 USD)
│   ├── PITCH_SCRIPT_3MIN.md        # Guion para el video pitch de 3 minutos
│   └── SUBMISSION_SUMMARY.md       # Resumen de entrega oficial y guía rápida de verificación
│
├── tinycardio_specification.md # Especificación técnica del sistema
└── README.md                 # Este documento
```

---

## Verificación Rápida en 60 Segundos

Para que los evaluadores puedan reproducir los resultados de inmediato:

### 1. Auditoría de Paridad Numérica entre Motores
Verifica la consistencia matemática entre el modelo JAX en GPU, el motor NumPy en CPU y el motor INT8 en C:
```bash
python3 03_edge_serialized/cross_verify_engines.py
```
> **Resultado esperado:** Paridad bit-exacta (100%) entre JAX y CPU, y correlación Pearson de **0.9892** con el motor INT8 en C.

### 2. Simulación de Paciente en Tiempo Real (Streaming)
```bash
# Simular paciente con Fibrilación Ventricular Maligna (PhysioNet VFDB 418)
python3 02_model_cpu/run_patient_simulation.py data/vfdb/418

# Simular paciente de control en Ritmo Sinusal Normal (PhysioNet NSRDB 16265)
python3 02_model_cpu/run_patient_simulation.py data/nsrdb/16265
```

### 3. Servidor Gateway Rural y Osciloscopio Web en Vivo
Inicia el servidor de telemetría y abre la interfaz web en cualquier navegador:
```bash
python3 02_model_cpu/gateway_server.py --port 8080
```
Abre `http://localhost:8080` para observar el osciloscopio interactivo con cálculo de riesgo continuo.

### 4. Monitor de Terminal ANSI en Tiempo Real
Para entornos de diagnóstico sin interfaz gráfica (vía SSH o terminal serie):
```bash
python3 02_model_cpu/cli_monitor.py data/vfdb/418
```

---

## Métricas de Rendimiento Clínico y de Hardware

| Parámetro | Requerimiento / Estándar | Desempeño Medido TinyCardio | Cumplimiento |
|:---|:---:|:---:|:---:|
| **Memoria ROM (Flash)** | < 16 KB | **540 bytes** (0.53 KB) | Superado (30x más compacto) |
| **Memoria RAM (SRAM)** | < 8 KB | **< 2.5 KB estática** | Superado (3x más ligero) |
| **Asignación de Memoria** | Sin fugas | **0 llamadas a `malloc`** | 100% Estático |
| **Latencia de Inferencia** | < 100 ms | **< 0.8 ms** por ventana | 125x más rápido que tiempo real |
| **Sensibilidad FV Letal** | > 90% | **100.0%** (PhysioNet VFDB) | Verificado |
| **Sensibilidad Pre-Paro** | > 85% | **93.2%** (PhysioNet SDDB) | Verificado |
| **Falsas Alarmas (Control)** | < 5% | **0.0%** (PhysioNet NSRDB) | Verificado |
| **Payload Telemetría SMS** | < 160 caracteres | **32 caracteres** (11B binario) | Compatible con 2G SMS estándar |
| **Costo BOM Hardware** | < $10.00 USD | **$4.30 USD** | Viable para poblaciones vulnerables |

---

## Documentación Detallada

* **[Impacto Socioeconómico y Análisis BOM ($4.30 USD)](docs/WORLD_BANK_IMPACT_REPORT.md)**: Alineación con ODS 3.4 y 9.5, iniciativa *Bottom 40%* del Banco Mundial y economía de despliegue rural.
* **[Guion de Video Pitch de 3 Minutos](docs/PITCH_SCRIPT_3MIN.md)**: Estructura audiovisual cronometrada para la sustentación ante el jurado.
* **[Resumen Oficial de Entrega](docs/SUBMISSION_SUMMARY.md)**: Ficha técnica integral de evaluación y reproducibilidad.
* **[Análisis Clínico y Fisiopatológico](docs/ANALISIS_CLINICO_TECNICO.md)**: Desglose cardiológico de las bases de datos y consideraciones biomédicas.

---

## Licencia

Este proyecto se distribuye bajo la licencia **MIT** de código abierto para fomentar la adopción humanitaria por parte de ministerios de salud, ONG y fabricantes de tecnología médica accesible.
