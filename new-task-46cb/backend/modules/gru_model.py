"""
GRU + Self-Attention model loader and inference engine.

Architecture:
    Input (20, 63)
    → Bidirectional GRU(256, return_sequences=True) + Dropout
    → Bidirectional GRU(128, return_sequences=True) + Dropout
    → Self-Attention (query-key-value dot-product)
    → GlobalAveragePooling1D
    → Dense(256, relu) + BatchNorm + Dropout
    → Dense(128, relu) + BatchNorm + Dropout
    → Dense(num_classes, softmax)

GPU usage: TensorFlow automatically uses RTX 2050 if CUDA 11.8 + cuDNN are installed.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import tensorflow as tf
from loguru import logger
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
try:
    from model_architecture import SelfAttention
    _CUSTOM_OBJECTS = {"SelfAttention": SelfAttention}
except ImportError:
    _CUSTOM_OBJECTS = {}

_model = None
_labels: list[str] = []


def _configure_gpu():
    """
    Best-effort GPU setup that remains compatible with TF1 and TF2.

    The previous implementation assumed tf.config always exists; on
    environments with an older TensorFlow build that attribute is
    missing and caused startup to crash. We now guard the calls and
    fall back to CPU cleanly.
    """
    gpus = []

    # TF 2.x path
    if hasattr(tf, "config") and hasattr(tf.config, "list_physical_devices"):
        try:
            gpus = tf.config.list_physical_devices("GPU")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"GPU probe via tf.config failed: {exc}")

    # TF 1.x fallback (no tf.config)
    if not gpus:
        try:
            from tensorflow.python.client import device_lib

            gpus = [
                d for d in device_lib.list_local_devices() if d.device_type == "GPU"
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"GPU probe via device_lib failed: {exc}")
            gpus = []

    if not gpus:
        logger.warning("No GPU detected - running on CPU")
        return

    # Enable memory growth when the API exists
    try:
        if hasattr(tf, "config") and hasattr(tf.config, "experimental"):
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not enable memory growth: {exc}")

    names = []
    for g in gpus:
        names.append(getattr(g, "name", getattr(g, "physical_device_desc", "GPU")))
    logger.info(f"GPU(s) found: {names}")


def load_model(model_path: str | Path, labels_path: str | Path) -> None:
    global _model, _labels

    _configure_gpu()

    model_path = Path(model_path)
    labels_path = Path(labels_path)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        raise FileNotFoundError(f"Labels not found at {labels_path}")

    logger.info(f"Loading GRU+Attention model from {model_path}")
    _model = tf.keras.models.load_model(str(model_path), custom_objects=_CUSTOM_OBJECTS)
    logger.info(f"Model loaded. Input shape: {_model.input_shape}")

    _labels = labels_path.read_text(encoding="utf-8").strip().splitlines()
    logger.info(f"Loaded {len(_labels)} class labels")


def predict(sequence: np.ndarray) -> tuple[str, float, dict]:
    """
    Run inference on a single sequence.

    Args:
        sequence: np.ndarray shape (20, 63) — one sliding window

    Returns:
        (predicted_label, confidence, {label: probability, ...})
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    x = sequence.astype(np.float32)
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)

    probs = _model.predict(x, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label = _labels[top_idx] if top_idx < len(_labels) else "unknown"

    top5 = np.argsort(probs)[::-1][:5]
    prob_map = {_labels[i]: float(probs[i]) for i in top5 if i < len(_labels)}

    return label, confidence, prob_map


def get_labels() -> list[str]:
    return _labels.copy()


def is_loaded() -> bool:
    return _model is not None
