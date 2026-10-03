# Repositorio TinyCardio — World Bank Group & Korea MSIT/MOFE Hackathon 2026

**TinyCardio** es un sistema biomédico de borde (*Edge AI / TinyML*) diseñado para la detección temprana y autónoma de colapso autonómico cardíaco (Muerte Súbita Cardíaca, arritmias ventriculares letales e isquemia aguda) directamente en microcontroladores de **\$2 a \$4 USD** sin depender de conexión a internet continua ni servidores en la nube.

---

## Estructura Modular del Proyecto

El repositorio está estructurado en tres pilares operativos principales, junto con los datos clínicos y la documentación:

```
WorldBank/
├── 01_training_pipeline/     # [Pilar 1] Ingesta, preprocesamiento y entrenamiento JAX con GPU (RTX 5070 CUDA)
│   ├── processed_data/       # Dataset balanceado y estratificado a 250 Hz (.npz)
│   ├── checkpoints/          # Checkpoints de pesos entrenados en formato .npz
│   ├── dataset_generator.py  # Generador de ventanas de 250 muestras con anti-aliasing
│   ├── model.py              # Definición de la arquitectura Tiny 1D-CNN en JAX
│   ├── train_jax_gpu.py      # Bucle de entrenamiento acelerado en GPU NVIDIA RTX 5070
│   ├── requirements_gpu.txt  # Dependencias CUDA 12 (jax[cuda12], optax, scipy)
│   └── setup_env.sh          # Script de configuración del entorno GPU
│
├── 02_model_cpu/             # [Pilar 2] Modelo entrenado listo para ejecución en CPU
│   ├── model_weights/        # Pesos entrenados exportados (.npz y .json)
│   ├── inference_cpu.py      # Motor de inferencia determinista ultraligero en NumPy
│   ├── benchmark_cpu.py      # Pruebas de rendimiento y latencia (< 0.7 ms por ventana)
│   └── run_patient_simulation.py # Simulación en tiempo real de monitoreo clínico
│
├── 03_edge_serialized/       # [Pilar 3] Modelo serializado para equipos de bajo costo (Microcontroladores)
│   ├── include/
│   │   └── model_weights.h   # Pesos INT8 como constantes en memoria Flash (540 bytes)
│   ├── src/
│   │   ├── tinyml_infer.h    # Interfaz C99 estricta y sin asignación dinámica (malloc-free)
│   │   ├── tinyml_infer.c    # Motor de inferencia en C (RAM < 2.5 KB)
│   │   ├── sms_alert_encoder.h # Codificador de payload de emergencia (< 60 bytes)
│   │   └── sms_alert_encoder.c # Generador de SMS 2G y telemetría binaria con CRC8
│   ├── quantize_int8.py      # Conversor de pesos flotantes a INT8 simétrico
│   ├── test_edge_emulator.py # Emulador de inferencia INT8 en Python
│   ├── main_edge_test.c      # Programa de prueba y banco de evaluación en C
│   └── Makefile              # Script de compilación con gcc
│
├── data/                     # Bases de datos originales de PhysioNet
│   ├── nsrdb/                # 18 registros (128 Hz) - Grupo de Control
│   ├── edb/                  # 90 registros (250 Hz) - Isquemia Miocárdica
│   ├── vfdb/                 # 22 registros (250 Hz) - Arritmias Letales (FV/TV)
│   └── sddb/                 # 23 registros (250 Hz) - Paro Cardíaco / Muerte Súbita
│
├── metadata/                 # Catálogos maestros generados
│   ├── master_records_catalog.csv # Catálogo unificado de 153 registros
│   └── summary_statistics.json    # Estadísticas globales consolidadas
│
├── docs/                     # Documentación médica y técnica
│   └── ANALISIS_CLINICO_TECNICO.md # Análisis fisiopatológico y consideraciones de IA
│
├── utils/                    # Utilidades de bajo nivel
│   ├── ecg_loader.py         # Lector nativo en Python de archivos WFDB formato 212
│   └── verify_integrity.py   # Verificación de hashes SHA-256 de los datos
│
├── tinycardio_specification.md # Especificación oficial del proyecto
└── README.md                 # Este documento
```

---

## Guía Rápida de Ejecución

### 1. Entrenar el Modelo con Aceleración GPU (RTX 5070)
```bash
# Ejecutar entrenamiento con JAX y CUDA en la GPU
python3 01_training_pipeline/train_jax_gpu.py
```
*Rendimiento:* Entrena 15 épocas completas sobre más de 5,200 ventanas clínicas en tan solo **~3.5 segundos** en la NVIDIA GeForce RTX 5070.

### 2. Ejecutar Inferencia y Simulación en CPU
```bash
# Medir latencia en CPU
python3 02_model_cpu/benchmark_cpu.py

# Simular paciente en monitoreo continuo con alerta de arritmia maligna
python3 02_model_cpu/run_patient_simulation.py data/vfdb/418

# Simular paciente control en ritmo sinusal normal
python3 02_model_cpu/run_patient_simulation.py data/nsrdb/16265
```

### 3. Serializar y Probar el Modelo para Microcontroladores (Edge / TinyML)
```bash
# Cuantizar a INT8 y generar el archivo include/model_weights.h
python3 03_edge_serialized/quantize_int8.py

# Validar el comportamiento del modelo INT8 sobre pacientes de prueba
python3 03_edge_serialized/test_edge_emulator.py
```
*Huella de Hardware:*
* **ROM Flash:** Solo **540 bytes** (0.53 KB).
* **RAM SRAM:** **$< 2.5\text{ KB}$** en buffer estático (*malloc-free*).
* **Alerta SMS 2G:** Payload binario de **11 bytes** o texto de **32 caracteres** (< 60 bytes).
