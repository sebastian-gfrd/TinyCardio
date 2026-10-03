"""
TinyCardio 1D-CNN Model in Pure JAX
===================================
Implementation of the ultra-lightweight 1D-CNN specified in tinycardio_specification.md.
Layers:
  1. Conv1D Block 1: 8 filters, kernel_size=5, stride=2, ReLU
  2. Conv1D Block 2: 16 filters, kernel_size=3, stride=2, ReLU
  3. Global Average Pooling (GAP)
  4. Dense Unit (1 output) + Sigmoid -> Risk score in [0.0, 1.0]

Ultra-frugal parameter footprint: ~465 weights, ~0.5 KB INT8 footprint.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple


def init_model_params(rng_key: jax.Array) -> Dict[str, jnp.ndarray]:
    """Initializes weights and biases for the Tiny 1D-CNN using He/Glorot initialization."""
    k1, k2, k3 = jax.random.split(rng_key, 3)

    # Conv1: input_channels=1, output_channels=8, kernel_size=5
    # JAX conv filter layout: (kernel_width, in_channels, out_channels)
    w_conv1 = jax.random.normal(k1, (5, 1, 8)) * jnp.sqrt(2.0 / 5.0)
    b_conv1 = jnp.zeros((8,))

    # Conv2: input_channels=8, output_channels=16, kernel_size=3
    w_conv2 = jax.random.normal(k2, (3, 8, 16)) * jnp.sqrt(2.0 / (3.0 * 8.0))
    b_conv2 = jnp.zeros((16,))

    # Dense: in_features=16, out_features=1
    w_dense = jax.random.normal(k3, (16, 1)) * jnp.sqrt(2.0 / 16.0)
    b_dense = jnp.zeros((1,))

    return {
        'w_conv1': w_conv1,
        'b_conv1': b_conv1,
        'w_conv2': w_conv2,
        'b_conv2': b_conv2,
        'w_dense': w_dense,
        'b_dense': b_dense
    }


def forward_pass(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass for batch input x of shape (batch, 250, 1).
    Returns logits of shape (batch, 1).
    """
    # 1. Conv1D Block 1 (stride 2, SAME padding)
    # JAX lax.conv_general_dilated takes dimension_numbers: ('NHC', 'WIO', 'NHC')
    # x shape: (N, 250, 1) -> W=250, C=1
    x_conv1 = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=params['w_conv1'],
        window_strides=(2,),
        padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC')
    ) + params['b_conv1']
    h1 = jax.nn.relu(x_conv1)  # shape: (N, 125, 8)

    # 2. Conv1D Block 2 (stride 2, SAME padding)
    x_conv2 = jax.lax.conv_general_dilated(
        lhs=h1,
        rhs=params['w_conv2'],
        window_strides=(2,),
        padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC')
    ) + params['b_conv2']
    h2 = jax.nn.relu(x_conv2)  # shape: (N, 63, 16)

    # 3. Global Average Pooling (along spatial dimension axis 1)
    gap = jnp.mean(h2, axis=1)  # shape: (N, 16)

    # 4. Dense layer -> logits
    logits = jnp.dot(gap, params['w_dense']) + params['b_dense']  # shape: (N, 1)
    return logits


def predict_probability(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Returns probability risk score in [0.0, 1.0]."""
    logits = forward_pass(params, x)
    return jax.nn.sigmoid(logits)


def count_parameters(params: Dict[str, jnp.ndarray]) -> int:
    """Calculates total number of trainable scalar parameters."""
    return sum(p.size for p in params.values())


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    p = init_model_params(key)
    print(f"TinyCardio 1D-CNN initialized successfully.")
    print(f"Total parameters: {count_parameters(p)} weights.")
    dummy_input = jnp.ones((4, 250, 1))
    probs = predict_probability(p, dummy_input)
    print(f"Forward pass output shape: {probs.shape} | Initial probabilities: {probs.ravel()[:4]}")
