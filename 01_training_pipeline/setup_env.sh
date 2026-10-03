#!/usr/bin/env bash
# ==============================================================================
# Setup Environment for TinyCardio Training on NVIDIA RTX 5070 GPU with JAX
# ==============================================================================
set -e

echo "=== Setting up TinyCardio JAX GPU Environment for NVIDIA RTX 5070 ==="

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Training will default to CPU."
fi

# Install dependencies with pip
pip install --break-system-packages -r requirements_gpu.txt

# Verify JAX GPU devices
python3 -c "import jax; print('JAX Backend:', jax.default_backend()); print('Devices:', jax.devices())"

echo "=== Setup Complete. Ready to run: python3 train_jax_gpu.py ==="
