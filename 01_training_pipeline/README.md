# TinyCardio — Pipeline de Ingesta y Entrenamiento con JAX GPU / CUDA (01_training_pipeline)

Este directorio contiene la arquitectura del modelo **TinyCardio 1D-CNN**, la ingesta de las bases de datos de PhysioNet (`nsrdb`, `edb`, `vfdb`, `sddb`) y el pipeline de entrenamiento acelerado por **GPU NVIDIA GeForce RTX 5070 con CUDA 12**.

---

## Archivos y Componentes

* **`dataset_generator.py`**: 
  * Carga registros clínicos de `data/nsrdb`, `data/edb`, `data/vfdb` y `data/sddb`.
  * Convierte las señales a milivoltios físicos y re-muestrea `nsrdb` de 128 Hz a 250 Hz con filtro anti-aliasing.
  * Aplica filtrado pasabanda Butterworth (0.5 – 40 Hz) y segmenta ventanas deslizantes de 250 muestras (1 segundo a 250 Hz).
  * Realiza una partición estratificada **a nivel de paciente** (70% Train, 15% Val, 15% Test) para evitar *data leakage*.
  * Exporta los datos a `processed_data/dataset_250hz.npz`.
* **`model.py`**:
  * Definición funcional de la arquitectura Tiny 1D-CNN en pure JAX (`conv_general_dilated`, ReLU, Global Average Pooling y neurona Densa sigmoide).
  * 465 parámetros entrenables (~0.5 KB de peso total).
* **`train_jax_gpu.py`**:
  * Detecta automáticamente la GPU **NVIDIA GeForce RTX 5070** (`CudaDevice(id=0)`).
  * Compila el entrenamiento en GPU mediante `@jax.jit` y optimización con Adam/Optax.
  * Calcula métricas clínicas clave para triaje: Exactitud, Sensibilidad (Recall de alerta), Especificidad, Precisión y F1-Score.
  * Exporta los mejores pesos entrenados directamente a `02_model_cpu/model_weights/` y `checkpoints/`.
* **`requirements_gpu.txt`** y **`setup_env.sh`**:
  * Paquetes de CUDA 12 (`jax[cuda12]`, `optax`, `scipy`, `numpy`) para habilitar aceleración por hardware en la GPU RTX 5070.

---

## Ejecución Rápida

### 1. Generar el dataset procesado
```bash
python3 01_training_pipeline/dataset_generator.py
```

### 2. Entrenar el modelo con GPU (RTX 5070)
```bash
python3 01_training_pipeline/train_jax_gpu.py
```
*Tiempo de entrenamiento:* $\sim 3.5\text{ segundos}$ para 15 épocas completas aceleradas en la RTX 5070.
