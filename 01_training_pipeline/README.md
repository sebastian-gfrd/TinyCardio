# TinyCardio — Ingestion and Training Pipeline with JAX GPU / CUDA (01_training_pipeline)

This directory contains the **TinyCardio 1D-CNN** model architecture, the ingestion pipeline for PhysioNet benchmark databases (`nsrdb`, `edb`, `vfdb`, `sddb`), and the hardware-accelerated training pipeline powered by **NVIDIA GeForce RTX 5070 GPU with CUDA 12**.

---

## Files and Components

* **`dataset_generator.py`**: 
  * Ingests clinical records from `data/nsrdb`, `data/edb`, `data/vfdb`, and `data/sddb`.
  * Converts raw signals to physical millivolts and resamples `nsrdb` from 128 Hz to 250 Hz using a polyphase anti-aliasing filter.
  * Applies a 2nd-order Butterworth bandpass filter (0.5 – 40 Hz) and segments continuous records into sliding windows of 250 samples (1 second at 250 Hz).
  * Performs strict **patient-level stratified partitioning** (70% Train, 15% Val, 15% Test) to eliminate data leakage.
  * Exports formatted tensors to `processed_data/dataset_250hz.npz`.
* **`augmentations.py`**:
  * Biomedical noise simulator implementing 6 physiological and environmental distortions: respiration baseline wander, 50/60 Hz powerline hum, electromyographic (EMG) muscle tremors, electrode contact motion jumps, and inter-patient amplitude scaling.
* **`test_augmentations.py`**:
  * Unit test suite verifying signal stability, SNR bounds, and numerical properties of the augmentation pipeline.
* **`model.py`**:
  * Functional definition of the Tiny 1D-CNN architecture in pure JAX (`conv_general_dilated`, ReLU activations, Global Average Pooling, and a single Dense sigmoid unit).
  * Exactly 465 trainable parameters (~0.5 KB total parameter footprint).
* **`train_jax_gpu.py`**:
  * Auto-detects the host **NVIDIA GeForce RTX 5070** GPU (`CudaDevice(id=0)`).
  * Compiles the GPU training graph via `@jax.jit` and optimizes with AdamW and Optax cosine decay scheduler.
  * Implements Binary Focal Loss ($\gamma = 2.0, \alpha = 0.55$) to address severe class imbalance.
  * Evaluates clinical triage metrics: Accuracy, Sensitivity (Alert Recall), Specificity, Precision, and F1-Score.
  * Automatically exports the best trained weights to `02_model_cpu/model_weights/` and `checkpoints/`.
* **`evaluate_clinical_metrics.py`**:
  * Comprehensive clinical validation suite generating ROC-AUC curves, confusion matrices, and per-database sensitivity breakdowns exported to `metadata/clinical_validation_report.json`.
* **`requirements_gpu.txt`** & **`setup_env.sh`**:
  * CUDA 12 environment dependencies (`jax[cuda12]`, `optax`, `scipy`, `numpy`) for hardware acceleration.

---

## Quick Execution

### 1. Generate the Processed Dataset
```bash
python3 01_training_pipeline/dataset_generator.py
```

### 2. Verify Augmentation Stability
```bash
python3 01_training_pipeline/test_augmentations.py
```

### 3. Train the Model on GPU (RTX 5070)
```bash
python3 01_training_pipeline/train_jax_gpu.py
```
*Training Performance:* $\sim 3.5\text{ seconds}$ for 15 full epochs accelerated on the RTX 5070.

### 4. Evaluate Clinical Metrics
```bash
python3 01_training_pipeline/evaluate_clinical_metrics.py
```
