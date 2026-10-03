# TinyCardio — Módulo de Inferencia en CPU (02_model_cpu)

Este directorio contiene el motor de inferencia optimizado para ejecución en **CPU** del modelo TinyCardio 1D-CNN, ideal para estaciones de triaje clínico local, ordenadores portátiles de personal de salud comunitario (CHWs) y gateways de telemedicina rural.

---

## Contenido del Directorio

* **`model_weights/`**: Contiene los pesos entrenados del modelo en formatos `.npz` y `.json`.
* **`inference_cpu.py`**: Motor de inferencia determinista y ultraligero que realiza la convolución 1D, pooling y clasificación sigmoide en NumPy estándar.
* **`benchmark_cpu.py`**: Herramienta de pruebas de rendimiento para medir latencia y rendimiento de procesamiento continuo.
* **`run_patient_simulation.py`**: Simulador en tiempo real que procesa señales de ECG de pacientes ventana a ventana y emite alertas automáticas ante arritmias o isquemia.

---

## Uso Rápido

### 1. Evaluar una ventana de prueba en Python
```python
from inference_cpu import TinyCardioCPU
import numpy as np

# Cargar motor de inferencia (umbral de alerta = 0.5)
model = TinyCardioCPU('model_weights/tinycardio_cpu_weights.npz', alert_threshold=0.5)

# Ventana de 250 muestras de ECG (1 segundo a 250 Hz)
ecg_window = np.random.randn(250)

# Predicción
result = model.predict_window(ecg_window)
print(result)
# {'risk_score': 0.8741, 'classification': 'HIGH_ALERT_PATHOLOGY', 'is_alert': True, 'latency_ms': 0.18}
```

### 2. Ejecutar Benchmark de Rendimiento en CPU
```bash
python3 02_model_cpu/benchmark_cpu.py
```
*Latencia típica:* **$< 0.25\text{ ms}$** por ventana de 1 segundo de ECG en CPU moderna (throughput de más de 4,000 ventanas/segundo).

### 3. Simular Monitoreo en Tiempo Real de un Paciente
```bash
# Simular paciente con taquicardia/fibrilación ventricular de VFDB
python3 02_model_cpu/run_patient_simulation.py data/vfdb/418

# Simular paciente de control sano de NSRDB
python3 02_model_cpu/run_patient_simulation.py data/nsrdb/16265
```
