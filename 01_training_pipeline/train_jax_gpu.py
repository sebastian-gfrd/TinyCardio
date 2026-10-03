"""
TinyCardio GPU-Accelerated Training Pipeline with JAX & CUDA
============================================================
Utilizes NVIDIA GeForce RTX 5070 GPU via JAX / XLA CUDA acceleration.
Trains the 1D-CNN with weighted binary cross-entropy loss, evaluates on validation set,
and automatically exports trained weights to 02_model_cpu and 03_edge_serialized.
"""

import os
import sys

# Configure JAX / XLA memory management for NVIDIA RTX 5070 (avoid greedy 90% preallocation)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.40'

import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple

# Import model architecture
from model import init_model_params, forward_pass, predict_probability, count_parameters


def print_device_info():
    """Prints detected execution devices (NVIDIA RTX 5070 or CPU)."""
    devices = jax.devices()
    backend = jax.default_backend()
    print("=" * 65)
    print(f"JAX Default Backend: {backend.upper()}")
    print(f"Available Devices ({len(devices)}): {devices}")
    has_gpu = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)
    if has_gpu:
        print(">>> SUCCESS: NVIDIA RTX 5070 GPU ACCELERATION ACTIVE (CUDA Enabled) <<<")
    else:
        print(">>> Running on CPU (install jax[cuda12] to enable RTX 5070 GPU) <<<")
    print("=" * 65)


def binary_cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, pos_weight: float = 1.2) -> jnp.ndarray:
    """Weighted binary cross entropy loss for handling triage sensitivity."""
    # labels shape: (N, 1), logits shape: (N, 1)
    probs = jax.nn.sigmoid(logits)
    eps = 1e-7
    probs_clamped = jnp.clip(probs, eps, 1.0 - eps)
    bce = -(pos_weight * labels * jnp.log(probs_clamped) + (1.0 - labels) * jnp.log(1.0 - probs_clamped))
    return jnp.mean(bce)


def create_train_step(optimizer):
    """Creates a JIT-compiled training step."""
    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        def loss_fn(p):
            logits = forward_pass(p, x_batch)
            loss = binary_cross_entropy_loss(logits, y_batch)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


@jax.jit
def eval_step(params, x_batch, y_batch):
    """JIT-compiled evaluation step."""
    logits = forward_pass(params, x_batch)
    loss = binary_cross_entropy_loss(logits, y_batch)
    probs = jax.nn.sigmoid(logits)
    preds = (probs >= 0.5).astype(jnp.float32)
    return loss, probs, preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes clinical triage performance metrics."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / max(1, len(y_true))
    sensitivity = tp / max(1, (tp + fn))  # Recall
    specificity = tn / max(1, (tn + fp))
    precision = tp / max(1, (tp + fp))
    f1 = 2 * (precision * sensitivity) / max(1e-7, (precision + sensitivity))

    return {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1)
    }


def train_model(npz_dataset_path: str, epochs: int = 15, batch_size: int = 64, lr: float = 0.001):
    """Main training loop with JAX."""
    print_device_info()

    if not os.path.exists(npz_dataset_path):
        print(f"Dataset not found at {npz_dataset_path}. Generating now...")
        import subprocess
        subprocess.run([sys.executable, '01_training_pipeline/dataset_generator.py'], check=True)

    data = np.load(npz_dataset_path)
    X_train, y_train = data['X_train'], np.expand_dims(data['y_train'], -1)
    X_val, y_val = data['X_val'], np.expand_dims(data['y_val'], -1)
    X_test, y_test = data['X_test'], np.expand_dims(data['y_test'], -1)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    rng = jax.random.PRNGKey(42)
    params = init_model_params(rng)
    print(f"Model parameters: {count_parameters(params)} weights initialized.")

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    train_step = create_train_step(optimizer)

    n_batches = len(X_train) // batch_size
    best_val_f1 = 0.0
    best_params = params

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(X_train))
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]

        train_losses = []
        for b in range(n_batches):
            xb = jnp.array(X_train_shuf[b * batch_size:(b + 1) * batch_size])
            yb = jnp.array(y_train_shuf[b * batch_size:(b + 1) * batch_size])
            params, opt_state, loss = train_step(params, opt_state, xb, yb)
            train_losses.append(float(loss))

        # Evaluation on Validation set
        val_xb = jnp.array(X_val)
        val_yb = jnp.array(y_val)
        val_loss, val_probs, val_preds = eval_step(params, val_xb, val_yb)
        metrics = compute_metrics(np.array(val_yb).ravel(), np.array(val_preds).ravel())

        mean_loss = np.mean(train_losses)
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {mean_loss:.4f} | Val Loss: {float(val_loss):.4f} | "
              f"Acc: {metrics['accuracy']*100:.1f}% | Sens: {metrics['sensitivity']*100:.1f}% | "
              f"Spec: {metrics['specificity']*100:.1f}% | F1: {metrics['f1_score']:.4f}")

        if metrics['f1_score'] > best_val_f1:
            best_val_f1 = metrics['f1_score']
            best_params = params

    elapsed = time.time() - start_train_time
    print("-" * 65)
    print(f"Training completed in {elapsed:.2f} seconds ({elapsed/epochs:.3f} s/epoch).")

    # Evaluate Best Model on Test Set
    test_loss, test_probs, test_preds = eval_step(best_params, jnp.array(X_test), jnp.array(y_test))
    test_metrics = compute_metrics(np.array(y_test).ravel(), np.array(test_preds).ravel())
    print("=" * 65)
    print("FINAL TEST SET PERFORMANCE (UNSEEN PATIENTS):")
    print(f"  Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"  Sensitivity: {test_metrics['sensitivity']*100:.2f}% (Triage Recall)")
    print(f"  Specificity: {test_metrics['specificity']*100:.2f}% (Normal Rhythm Rejection)")
    print(f"  Precision:   {test_metrics['precision']*100:.2f}%")
    print(f"  F1-Score:    {test_metrics['f1_score']:.4f}")
    print("=" * 65)

    # Save checkpoints and export weights
    os.makedirs('01_training_pipeline/checkpoints', exist_ok=True)
    os.makedirs('02_model_cpu/model_weights', exist_ok=True)

    weights_np = {k: np.array(v) for k, v in best_params.items()}
    np.savez('01_training_pipeline/checkpoints/best_model_weights.npz', **weights_np)
    np.savez('02_model_cpu/model_weights/tinycardio_cpu_weights.npz', **weights_np)

    # Also save as JSON for cross-platform inspection
    weights_json = {k: v.tolist() for k, v in weights_np.items()}
    with open('02_model_cpu/model_weights/tinycardio_cpu_weights.json', 'w') as f:
        json.dump(weights_json, f)

    print("Model weights successfully exported to 01_training_pipeline/checkpoints and 02_model_cpu/model_weights.")
    return best_params


if __name__ == '__main__':
    ds_path = '01_training_pipeline/processed_data/dataset_250hz.npz'
    train_model(ds_path, epochs=15, batch_size=64, lr=0.002)
