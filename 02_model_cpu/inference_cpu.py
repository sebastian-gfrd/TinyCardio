"""
TinyCardio CPU Inference Engine
===============================
Deterministic, ultra-fast CPU inference for trained TinyCardio 1D-CNN.
Designed for local healthcare triage, rural clinic laptops, and telemedicine gateways.
Requires only standard NumPy (or JAX CPU).
"""

import os
import time
import json
import numpy as np
from typing import Dict, Union, Tuple, List


class TinyCardioCPU:
    """CPU Inference Model for TinyCardio."""

    def __init__(self, weights_path: str = '02_model_cpu/model_weights/tinycardio_cpu_weights.npz', alert_threshold: float = 0.5):
        self.weights_path = weights_path
        self.threshold = alert_threshold
        self.weights: Dict[str, np.ndarray] = {}
        self.load_weights()

    def load_weights(self):
        """Loads weights from .npz or .json."""
        if not os.path.exists(self.weights_path):
            # Try searching in default location
            fallback = os.path.join(os.path.dirname(__file__), 'model_weights', 'tinycardio_cpu_weights.npz')
            if os.path.exists(fallback):
                self.weights_path = fallback
            else:
                fallback_json = os.path.join(os.path.dirname(__file__), 'model_weights', 'tinycardio_cpu_weights.json')
                if os.path.exists(fallback_json):
                    self.weights_path = fallback_json

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        if self.weights_path.endswith('.npz'):
            data = np.load(self.weights_path)
            self.weights = {k: data[k] for k in data.files}
        elif self.weights_path.endswith('.json'):
            with open(self.weights_path, 'r') as f:
                data = json.load(f)
            self.weights = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        else:
            raise ValueError(f"Unsupported format: {self.weights_path}")

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def _conv1d_stride2_same(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Pure NumPy 1D Convolution with stride=2 and SAME padding.
        x shape: (length, in_c)
        w shape: (k_w, in_c, out_c)
        b shape: (out_c,)
        """
        length, in_c = x.shape
        k_w, _, out_c = w.shape
        stride = 2

        pad_total = max(0, (length + stride - 1) // stride * stride - length + k_w - stride)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        if pad_total > 0:
            x_padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode='constant')
        else:
            x_padded = x

        out_len = (x_padded.shape[0] - k_w) // stride + 1
        output = np.zeros((out_len, out_c), dtype=np.float32)

        for i in range(out_len):
            start = i * stride
            patch = x_padded[start:start + k_w, :]  # (k_w, in_c)
            # Element-wise multiply across k_w and in_c, then sum
            output[i] = np.tensordot(patch, w, axes=([0, 1], [0, 1])) + b

        return output

    def predict_window(self, window: np.ndarray) -> Dict[str, Union[float, str, bool]]:
        """
        Processes a single window of 250 ECG samples.
        Args:
            window: array of shape (250,) or (250, 1).
        Returns:
            Dictionary with risk_score, classification, alert_flag, and latency_ms.
        """
        t0 = time.perf_counter()

        x = np.asarray(window, dtype=np.float32).reshape(250, 1)

        # Standardize / Z-score normalize
        std = np.std(x)
        if std > 1e-4:
            x = (x - np.mean(x)) / std
        else:
            x = np.zeros_like(x)

        # Layer 1: Conv1D (k=5, s=2) + ReLU
        h1 = self._relu(self._conv1d_stride2_same(x, self.weights['w_conv1'], self.weights['b_conv1']))

        # Layer 2: Conv1D (k=3, s=2) + ReLU
        h2 = self._relu(self._conv1d_stride2_same(h1, self.weights['w_conv2'], self.weights['b_conv2']))

        # Layer 3: Global Average Pooling (GAP)
        gap = np.mean(h2, axis=0)  # shape: (16,)

        # Layer 4: Dense + Sigmoid
        logit = np.dot(gap, self.weights['w_dense'].ravel()) + self.weights['b_dense'][0]
        prob = float(self._sigmoid(logit))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        is_alert = prob >= self.threshold

        return {
            'risk_score': round(prob, 4),
            'classification': 'HIGH_ALERT_PATHOLOGY' if is_alert else 'NORMAL_SINUS_RHYTHM',
            'is_alert': is_alert,
            'latency_ms': round(latency_ms, 3)
        }


if __name__ == '__main__':
    weights_p = '02_model_cpu/model_weights/tinycardio_cpu_weights.npz'
    if not os.path.exists(weights_p):
        print(f"Weights file {weights_p} not found yet. Run train_jax_gpu.py first.")
        sys.exit(0)

    model = TinyCardioCPU(weights_p)
    dummy = np.random.randn(250).astype(np.float32)
    res = model.predict_window(dummy)
    print("TinyCardio CPU Inference Test:")
    for k, v in res.items():
        print(f"  {k}: {v}")
